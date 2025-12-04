"""
Training script for Experiment 11: Dynamic Routing vs Static Baseline

Usage:
    # Train baseline (static layers)
    python run_experiment.py --config baseline
    
    # Train dynamic routing
    python run_experiment.py --config dynamic
    
    # Resume from checkpoint
    python run_experiment.py --config dynamic --resume checkpoints_dynamic/best_model.pt
"""

import torch
import torch.nn as nn
import sys
import os
import time
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

# Fix tokenizer parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add paths - experiments dir first (for local config/models), then root
experiments_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(experiments_dir)
sys.path.insert(0, experiments_dir)  # Local files first
sys.path.insert(1, root_dir)  # Then root

# Import local modules from experiments directory
import config as exp_config
import models as exp_models
from data.loader import quick_dataset
from utils.helpers import set_seed
from torch.utils.data import DataLoader

# Use the functions from our local modules
get_config = exp_config.get_config
create_model = exp_models.create_model





def load_and_cache_data(data_config, experiment_config=None):
    import pickle
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from pathlib import Path
    
    # Check cache first
    cache_dir = Path(__file__).parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / "tokens_cache.pkl"
    
    # For small experiments, we only need ~10M tokens (allows multiple epochs through data)
    # This is much faster to load and sufficient for 1000 training steps
    max_tokens_to_collect = 10_000_000
    
    if cache_file.exists():
        print(f"Loading cached tokens from {cache_file}...")
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
            tokenizer = cached_data['tokenizer']
            tokens = cached_data['tokens']
        print(f"Loaded {len(tokens):,} cached tokens")
        return [], tokenizer, tokens
    
    print(f"No cache found, collecting tokens...")
    print(f"Target: {max_tokens_to_collect:,} tokens (cached for future runs)")
    
    # Setup tokenizer
    tokenizer_name = "HuggingFaceTB/SmolLM-135M"
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset in streaming mode
    print("Loading dataset (streaming)...")
    dataset = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "cosmopedia-v2",
        split="train",
        streaming=True
    )
    
    print("Collecting tokens...")
    tokens = []
    
    # Iterate and tokenize
    for i, item in enumerate(dataset):
        text = item['text']
        batch_tokens = tokenizer(text, add_special_tokens=True, truncation=False)['input_ids']
        tokens.extend(batch_tokens)
        
        if len(tokens) >= max_tokens_to_collect:
            break
            
        if i % 100 == 0:
            print(f"  Collected {len(tokens):,} / {max_tokens_to_collect:,} tokens...", end='\r')
            
    tokens = tokens[:max_tokens_to_collect]
    print(f"\nCollected {len(tokens):,} tokens")
    
    # Save to cache
    print(f"Saving to cache: {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump({'tokenizer': tokenizer, 'tokens': tokens}, f)
    print("✓ Cached for future runs")
    
    return [], tokenizer, tokens




class Trainer:
    """Training manager for Dynamic Routing Experiment"""
    
    def __init__(self, model, config, train_loader, val_loader, device, save_dir=None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.save_dir = Path(save_dir) if save_dir else Path("checkpoints")
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # History
        self.train_history = []
        self.val_history = []
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            return max(0.1, (self.config.max_steps - step) / (self.config.max_steps - self.config.warmup_steps))
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch):
        """Single training step"""
        self.model.train()
        
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(self.device)
        else:
            input_ids = batch.to(self.device)
        
        labels = input_ids.clone()
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=labels, step=self.global_step)
        loss = outputs.loss
        
        # Log auxiliary loss for dynamic routing
        aux_loss = None
        if self.config.use_dynamic_routing and outputs.past_key_values:
            aux_loss = outputs.past_key_values[0]
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item(), aux_loss.item() if aux_loss is not None else 0.0
    
    @torch.no_grad()
    def evaluate(self, max_batches=None):
        """Evaluate on validation set"""
        self.model.eval()
        
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        for i, batch in enumerate(self.val_loader):
            if max_batches and i >= max_batches:
                break
            
            if isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(self.device)
            else:
                input_ids = batch.to(self.device)
            
            labels = input_ids.clone()
            
            outputs = self.model(input_ids=input_ids, labels=labels, step=self.global_step)
            loss = outputs.loss
            logits = outputs.logits
            
            # Calculate accuracy
            predictions = logits.argmax(dim=-1)
            shift_preds = predictions[..., :-1].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            correct = (shift_preds == shift_labels).sum().item()
            
            total_loss += loss.item() * input_ids.numel()
            total_correct += correct
            total_tokens += shift_labels.numel()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'perplexity': perplexity,
        }
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("Starting Training")
        print("="*70)
        
        start_time = time.time()
        running_loss = 0
        running_aux_loss = 0
        steps_since_log = 0
        
        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                if self.global_step >= self.config.max_steps:
                    break
                
                # Training step
                loss, aux_loss = self.train_step(batch)
                running_loss += loss
                running_aux_loss += aux_loss
                steps_since_log += 1
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = running_loss / steps_since_log
                    avg_aux = running_aux_loss / steps_since_log
                    lr = self.scheduler.get_last_lr()[0]
                    
                    log_msg = (f"Step {self.global_step}/{self.config.max_steps} | "
                               f"Loss: {avg_loss:.4f} | "
                               f"LR: {lr:.6f}")
                    
                    if self.config.use_dynamic_routing:
                        log_msg += f" | Aux: {avg_aux:.4f}"
                    
                    print(log_msg)
                    
                    self.train_history.append({
                        'step': self.global_step,
                        'loss': avg_loss,
                        'aux_loss': avg_aux if self.config.use_dynamic_routing else 0.0,
                        'lr': lr,
                    })
                    
                    running_loss = 0
                    running_aux_loss = 0
                    steps_since_log = 0
                
                # Evaluation
                if self.global_step % self.config.eval_interval == 0:
                    val_metrics = self.evaluate(max_batches=self.config.eval_batches)
                    
                    print(f"\n{'='*70}")
                    print(f"Evaluation at step {self.global_step}")
                    print(f"{'='*70}")
                    print(f"Val Loss: {val_metrics['loss']:.4f}")
                    print(f"Val Accuracy: {val_metrics['accuracy']:.4f} ({val_metrics['accuracy']*100:.2f}%)")
                    print(f"Val Perplexity: {val_metrics['perplexity']:.2f}")
                    
                    # Log routing stats for dynamic model
                    if self.config.use_dynamic_routing and hasattr(self.model, 'get_routing_stats'):
                        routing_stats = self.model.get_routing_stats(reset=True)
                        print(f"\nRouting Statistics:")
                        print(f"  Layer 1: GDN={routing_stats['layer_1_gdn_pct']:.1f}%, Softmax={routing_stats['layer_1_attn_pct']:.1f}%")
                        print(f"  Layer 2: GDN={routing_stats['layer_2_gdn_pct']:.1f}%, Softmax={routing_stats['layer_2_attn_pct']:.1f}%")
                        val_metrics.update(routing_stats)
                    
                    print(f"{'='*70}\n")
                    
                    self.val_history.append({
                        'step': self.global_step,
                        **val_metrics
                    })
                    
                    # Save best model
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        self.save_checkpoint('best_model.pt')
                        print(f"✓ New best validation loss: {self.best_val_loss:.4f} (saved)")
        
        total_time = time.time() - start_time
        
        # Save final model
        self.save_checkpoint('final_model.pt')
        
        print(f"\n{'='*70}")
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f}m)")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")
        
        return {
            'total_time': total_time,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint_path = self.save_dir / filename
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path


def main():
    """Main experiment function"""
    parser = argparse.ArgumentParser(description='Train Dynamic Routing vs Baseline')
    parser.add_argument('--config', type=str, default='baseline',
                        choices=['baseline', 'dynamic'],
                        help='Config to use (baseline or dynamic)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = get_config(args.config)
    
    print("="*70)
    print(f"EXPERIMENT 11: Dynamic Routing")
    print(f"Configuration: {args.config}")
    print("="*70)
    
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    print(f"\nUsing device: {device}")
    print(f"Architecture: {config.hidden_size}d, {config.num_hidden_layers} layers")
    print(f"Dynamic routing: {config.use_dynamic_routing}")
    if config.use_dynamic_routing:
        print(f"  Routed layers: {config.routed_layers}")
        print(f"  Fixed GDN layers: {config.fixed_gdn_layers}")
        print(f"  Fixed Attn layers: {config.fixed_attn_layers}")
        print(f"  Load balance alpha: {config.load_balance_alpha}")
    
    # Load data
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)
    
    from dataclasses import dataclass
    @dataclass
    class DataConfig:
        num_documents: int = config.num_documents
        max_tokens: int = config.max_tokens
        vocab_size: int = config.vocab_size
    
    data_config = DataConfig()
    texts, tokenizer, tokens = load_and_cache_data(data_config, config)
    config.vocab_size = len(tokenizer)
    
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Total tokens: {len(tokens):,}")
    
    # Split tokens
    val_split_ratio = 0.1
    val_token_start = int(len(tokens) * (1 - val_split_ratio))
    
    train_tokens = tokens[:val_token_start]
    val_tokens = tokens[val_token_start:]
    
    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens: {len(val_tokens):,}")
    
    # Create data loaders
    from data.streaming_dataset import create_progressive_loaders
    
    train_loader, val_loader = create_progressive_loaders(
        train_tokens, val_tokens,
        config.max_seq_len, config.batch_size,
        None, None
    )
    
    print(f"Train batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    
    # Create model
    print("\n" + "="*70)
    print("Creating Model")
    print("="*70)
    
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    if dtype == torch.float32:
        print("⚠ Warning: bfloat16 not supported")
    
    model = create_model(config)
    model = model.to(device=device, dtype=dtype)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Using dtype: {dtype}")
    
    # Train
    checkpoints_dir = Path(__file__).parent / config.checkpoint_dir
    results_dir = Path(__file__).parent / f"results_{args.config}"
    
    print(f"\n📁 Output directories:")
    print(f"   Checkpoints: {checkpoints_dir}")
    print(f"   Results: {results_dir}")
    
    trainer = Trainer(model, config, train_loader, val_loader, device, save_dir=checkpoints_dir)
    
    # Load checkpoint if resuming
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.global_step = checkpoint['global_step']
        trainer.best_val_loss = checkpoint['best_val_loss']
        print(f"\n✓ Resumed from checkpoint: {args.resume}")
        print(f"   Step: {trainer.global_step}, Best val loss: {trainer.best_val_loss:.4f}")
    
    results = trainer.train()
    
    # Save results
    results_dir.mkdir(exist_ok=True, parents=True)
    
    results_summary = {
        'experiment': 'exp11_dynamic_routing',
        'config_name': args.config,
        'config': {
            'use_dynamic_routing': config.use_dynamic_routing,
            'routed_layers': config.routed_layers,
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
            'learning_rate': config.learning_rate,
            'load_balance_alpha': config.load_balance_alpha if config.use_dynamic_routing else 0.0,
        },
        'results': {
            'total_time': results['total_time'],
            'best_val_loss': results['best_val_loss'],
        },
    }
    
    
    with open(results_dir / 'training_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save detailed history
    with open(results_dir / 'train_history.json', 'w') as f:
        json.dump(results['train_history'], f, indent=2)
    
    with open(results_dir / 'val_history.json', 'w') as f:
        json.dump(results['val_history'], f, indent=2)
    
    print(f"\nResults saved to: {results_dir / 'training_results.json'}")
    
    # Create visualizations for dynamic routing
    if config.use_dynamic_routing and results['val_history']:
        print("\n" + "="*70)
        print("Creating Routing Visualizations")
        print("="*70)
        
        # Extract routing data
        steps = []
        layer_1_gdn = []
        layer_1_attn = []
        layer_2_gdn = []
        layer_2_attn = []
        
        for entry in results['val_history']:
            if 'layer_1_gdn_pct' in entry:
                steps.append(entry['step'])
                layer_1_gdn.append(entry['layer_1_gdn_pct'])
                layer_1_attn.append(entry['layer_1_attn_pct'])
                layer_2_gdn.append(entry['layer_2_gdn_pct'])
                layer_2_attn.append(entry['layer_2_attn_pct'])
        
        if steps:
            # Figure 1: Layer Selection Over Time (Stacked Area)
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Layer 1
            ax1.fill_between(steps, 0, layer_1_gdn, alpha=0.6, color='#2E86AB', label='GDN')
            ax1.fill_between(steps, layer_1_gdn, 100, alpha=0.6, color='#A23B72', label='Softmax Attn')
            ax1.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
            ax1.set_title('Layer 1 Routing: Per-Token Selection', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 100)
            
            # Layer 2
            ax2.fill_between(steps, 0, layer_2_gdn, alpha=0.6, color='#2E86AB', label='GDN')
            ax2.fill_between(steps, layer_2_gdn, 100, alpha=0.6, color='#A23B72', label='Softmax Attn')
            ax2.set_xlabel('Training Step', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
            ax2.set_title('Layer 2 Routing: Per-Token Selection', fontsize=14, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=11)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)
            
            plt.tight_layout()
            routing_plot_path = results_dir / 'layer_selection_over_time.png'
            plt.savefig(routing_plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {routing_plot_path}")
            plt.close()
            
            # Figure 2: Comparative Line Plot
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(steps, layer_1_gdn, marker='o', linewidth=2.5, color='#2E86AB', 
                   label='Layer 1: GDN %', markersize=6)
            ax.plot(steps, layer_2_gdn, marker='s', linewidth=2.5, color='#F18F01', 
                   label='Layer 2: GDN %', markersize=6)
            ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Balanced (50%)')
            
            ax.set_xlabel('Training Step', fontsize=12, fontweight='bold')
            ax.set_ylabel('GDN Selection Percentage (%)', fontsize=12, fontweight='bold')
            ax.set_title('Dynamic Routing: GDN Preference Across Layers', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
            
            plt.tight_layout()
            comparison_plot_path = results_dir / 'routing_comparison.png'
            plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved: {comparison_plot_path}")
            plt.close()
            
            # Figure 3: Final routing distribution (for paper)
            if len(steps) > 0:
                final_step_idx = -1
                final_data = {
                    'Layer 1': [layer_1_gdn[final_step_idx], layer_1_attn[final_step_idx]],
                    'Layer 2': [layer_2_gdn[final_step_idx], layer_2_attn[final_step_idx]]
                }
                
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(final_data))
                width = 0.35
                
                gdn_bars = ax.bar(x - width/2, [d[0] for d in final_data.values()], 
                                 width, label='GDN', color='#2E86AB', alpha=0.8)
                attn_bars = ax.bar(x + width/2, [d[1] for d in final_data.values()], 
                                  width, label='Softmax Attn', color='#A23B72', alpha=0.8)
                
                ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
                ax.set_title(f'Final Routing Distribution (Step {steps[final_step_idx]})', 
                           fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(final_data.keys(), fontsize=11)
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_ylim(0, 100)
                
                # Add percentage labels on bars
                for bars in [gdn_bars, attn_bars]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
                
                plt.tight_layout()
                final_dist_path = results_dir / 'final_routing_distribution.png'
                plt.savefig(final_dist_path, dpi=300, bbox_inches='tight')
                print(f"✓ Saved: {final_dist_path}")
                plt.close()
                
            print("="*70)
    
    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETED!")
    print("="*70)
    print(f"Configuration: {args.config}")
    print(f"Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"Training Time: {results['total_time']:.1f}s ({results['total_time']/60:.1f}m)")
    print(f"\n💾 Saved to: {checkpoints_dir}/best_model.pt")
    print("="*70)


if __name__ == "__main__":
    main()
