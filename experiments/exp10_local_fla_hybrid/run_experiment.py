"""
Training script for Hybrid DeltaNet + Attention model
Experiment 10: Using local FLA clone (flash-linear-attention/fla)

Available Experiments (H100 80GB):
    - h100_deltanet: Pure DeltaNet (baseline)
    - h100_transformer: Pure Transformer (full attention)
    - h100_hybrid_sparse: 17% attention on [5,11] (DEFAULT)
    - h100_hybrid_alternating: 50% attention (every other layer)
    - h100_hybrid_late: 33% attention (last 4 layers)

Note: This experiment uses the local cloned FLA repository instead of pip-installed package

Usage:
    # Default: H100 hybrid sparse
    python run_experiment.py
    
    # Other H100 experiments
    python run_experiment.py --experiment h100_deltanet
    python run_experiment.py --experiment h100_transformer
    python run_experiment.py --experiment h100_hybrid_alternating
    python run_experiment.py --experiment h100_hybrid_late
    
    # Resume from checkpoint
    python run_experiment.py --resume checkpoints/best_model.pt
    
    # Resume and extend training
    python run_experiment.py --resume checkpoints/best_model.pt --extend-steps 5000
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
matplotlib.use('Agg')  # Use non-interactive backend

# Fix tokenizer parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

# Add flash-linear-attention to path for local imports
fla_path = os.path.join(root_dir, 'flash-linear-attention')
if fla_path not in sys.path:
    sys.path.insert(0, fla_path)

from experiments.exp10_local_fla_hybrid.config import (
    ExperimentConfig,
    # H100 experiment variants (primary)
    get_h100_deltanet_only,
    get_h100_transformer_only,
    get_h100_hybrid_sparse,
    get_h100_hybrid_alternating,
    get_h100_hybrid_late,
)
from experiments.exp10_local_fla_hybrid.models import (
    GatedDeltaNetWrapper,
    count_parameters,
)
from data.loader import prepare_lm_dataset
from configs.dataset_config import DataConfig
from utils.helpers import set_seed
from torch.utils.data import DataLoader, random_split


class Trainer:
    """Training manager for Gated DeltaNet"""
    
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
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_checkpoint_path = None
        
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
        
        # Handle different batch types
        if isinstance(batch, dict):
            input_ids = batch['input_ids'].to(self.device)
            if 'labels' in batch:
                labels = batch['labels'].to(self.device)
            else:
                labels = input_ids.clone()
        elif isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(self.device)
            labels = input_ids.clone()
        else:
            input_ids = batch.to(self.device)
            labels = input_ids.clone()
        
        # Forward pass
        outputs = self.model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss.item()
    
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
            
            # Handle different batch types
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(self.device)
                if 'labels' in batch:
                    labels = batch['labels'].to(self.device)
                else:
                    labels = input_ids.clone()
            elif isinstance(batch, (list, tuple)):
                input_ids = batch[0].to(self.device)
                labels = input_ids.clone()
            else:
                input_ids = batch.to(self.device)
                labels = input_ids.clone()
            
            outputs = self.model(input_ids=input_ids, labels=labels)
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
        steps_since_log = 0
        
        while self.global_step < self.config.max_steps:
            for batch in self.train_loader:
                if self.global_step >= self.config.max_steps:
                    break
                
                # Training step
                loss = self.train_step(batch)
                running_loss += loss
                steps_since_log += 1
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = running_loss / steps_since_log
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - start_time
                    steps_per_sec = self.global_step / elapsed
                    
                    print(f"Step {self.global_step}/{self.config.max_steps} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {lr:.6f} | "
                          f"Speed: {steps_per_sec:.2f} steps/s")
                    
                    self.train_history.append({
                        'step': self.global_step,
                        'loss': avg_loss,
                        'lr': lr,
                    })
                    
                    running_loss = 0
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
                    print(f"{'='*70}\n")
                    
                    self.val_history.append({
                        'step': self.global_step,
                        **val_metrics
                    })
                    
                    # Save best model
                    if val_metrics['loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['loss']
                        self.save_checkpoint('best_model.pt', is_best=True)
                        print(f"✓ New best validation loss: {self.best_val_loss:.4f} (saved)")
                
                # Periodic checkpoint saving
                if self.global_step % self.config.save_interval == 0:
                    checkpoint_path = self.save_checkpoint(f'checkpoint_step_{self.global_step}.pt', is_best=False)
                    print(f"💾 Checkpoint saved at step {self.global_step}: {checkpoint_path}")
            
            self.epoch += 1
        
        total_time = time.time() - start_time
        
        # Save final model
        final_checkpoint = self.save_checkpoint('final_model.pt', is_best=False)
        
        print(f"\n{'='*70}")
        print(f"Training completed in {total_time:.2f}s ({total_time/60:.2f}m)")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"\n💾 Models saved:")
        print(f"  Best model: {self.best_checkpoint_path}")
        print(f"  Final model: {final_checkpoint}")
        print(f"{'='*70}\n")
        
        return {
            'total_time': total_time,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_checkpoint': str(self.best_checkpoint_path),
            'final_checkpoint': str(final_checkpoint),
        }
    
    def save_checkpoint(self, filename, is_best=False):
        """Save model checkpoint for training resumption"""
        checkpoint_path = self.save_dir / filename
        
        # Save complete checkpoint including training history and data state
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }
        
        # Save data loader state if available (for progressive loading)
        if hasattr(self.train_loader, 'get_state'):
            checkpoint['train_data_state'] = self.train_loader.get_state()
        if hasattr(self.val_loader, 'get_state'):
            checkpoint['val_data_state'] = self.val_loader.get_state()
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            self.best_checkpoint_path = checkpoint_path
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore training state"""
        print(f"\n{'='*70}")
        print(f"Resuming from checkpoint: {checkpoint_path}")
        print(f"{'='*70}")
        
        # Load checkpoint (allowlist ExperimentConfig for PyTorch 2.6+ safety)
        torch.serialization.add_safe_globals([ExperimentConfig])
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Check if max_steps has changed (extended training)
        old_config = checkpoint['config']
        max_steps_changed = old_config.max_steps != self.config.max_steps
        
        # Restore model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Restore optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore training state first (needed for scheduler)
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint['best_val_loss']
        
        # Handle scheduler: recreate if max_steps changed, otherwise restore state
        if max_steps_changed:
            print(f"  ⚠ max_steps changed ({old_config.max_steps} → {self.config.max_steps})")
            print(f"  Recreating LR scheduler with new schedule...")
            self.scheduler = self._create_scheduler()
            # Fast-forward scheduler to current step
            for _ in range(self.global_step):
                self.scheduler.step()
            print(f"  ✓ Scheduler recreated and fast-forwarded to step {self.global_step}")
        else:
            # Normal resume: restore scheduler state
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Restore history if available
        self.train_history = checkpoint.get('train_history', [])
        self.val_history = checkpoint.get('val_history', [])
        
        # Check for data loader states
        train_data_state = checkpoint.get('train_data_state', None)
        val_data_state = checkpoint.get('val_data_state', None)
        
        if train_data_state:
            print(f"  ✓ Found progressive data state (train consumed: {train_data_state.get('tokens_consumed', 0):,} tokens)")
        
        print(f"✓ Checkpoint loaded successfully!")
        print(f"  Resuming from step: {self.global_step}")
        print(f"  Epoch: {self.epoch}")
        print(f"  Best val loss so far: {self.best_val_loss:.4f}")
        print(f"  Current LR: {self.scheduler.get_last_lr()[0]:.6f}")
        print(f"  Training history entries: {len(self.train_history)}")
        print(f"  Validation history entries: {len(self.val_history)}")
        print(f"{'='*70}\n")
        
        return checkpoint


def plot_training_curves(train_history, val_history, save_path):
    """Plot training and validation curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    if train_history:
        steps = [h['step'] for h in train_history]
        losses = [h['loss'] for h in train_history]
        axes[0, 0].plot(steps, losses, 'b-', linewidth=2, label='Train Loss')
        axes[0, 0].set_xlabel('Step', fontweight='bold')
        axes[0, 0].set_ylabel('Loss', fontweight='bold')
        axes[0, 0].set_title('Training Loss', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    
    # Validation loss
    if val_history:
        steps = [h['step'] for h in val_history]
        losses = [h['loss'] for h in val_history]
        axes[0, 1].plot(steps, losses, 'r-', linewidth=2, marker='o', label='Val Loss')
        axes[0, 1].set_xlabel('Step', fontweight='bold')
        axes[0, 1].set_ylabel('Loss', fontweight='bold')
        axes[0, 1].set_title('Validation Loss', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # Validation accuracy
    if val_history:
        steps = [h['step'] for h in val_history]
        accs = [h['accuracy'] * 100 for h in val_history]
        axes[1, 0].plot(steps, accs, 'g-', linewidth=2, marker='s', label='Val Accuracy')
        axes[1, 0].set_xlabel('Step', fontweight='bold')
        axes[1, 0].set_ylabel('Accuracy (%)', fontweight='bold')
        axes[1, 0].set_title('Validation Accuracy', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    # Validation perplexity
    if val_history:
        steps = [h['step'] for h in val_history]
        ppls = [h['perplexity'] for h in val_history]
        axes[1, 1].plot(steps, ppls, 'm-', linewidth=2, marker='^', label='Val Perplexity')
        axes[1, 1].set_xlabel('Step', fontweight='bold')
        axes[1, 1].set_ylabel('Perplexity', fontweight='bold')
        axes[1, 1].set_title('Validation Perplexity', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")


def main():
    """Main experiment function"""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train Hybrid DeltaNet + Attention model (H100) with local FLA')
    parser.add_argument('--experiment', type=str, default='h100_hybrid_sparse',
                        choices=[
                            'h100_deltanet', 'h100_transformer',
                            'h100_hybrid_sparse', 'h100_hybrid_alternating', 'h100_hybrid_late',
                        ],
                        help='H100 experiment variant (default: h100_hybrid_sparse)')
    parser.add_argument('--resume', type=str, default=None, 
                        help='Path to checkpoint to resume from (e.g., checkpoints/best_model.pt)')
    parser.add_argument('--extend-steps', type=int, default=None,
                        help='Extend training to this many total steps (useful when resuming)')
    args = parser.parse_args()
    
    # H100 Experiment configuration mapping
    EXPERIMENTS = {
        'h100_deltanet': ('H100: Pure DeltaNet', get_h100_deltanet_only),
        'h100_transformer': ('H100: Pure Transformer', get_h100_transformer_only),
        'h100_hybrid_sparse': ('H100: Hybrid Sparse 17%', get_h100_hybrid_sparse),
        'h100_hybrid_alternating': ('H100: Hybrid Alternating 50%', get_h100_hybrid_alternating),
        'h100_hybrid_late': ('H100: Hybrid Late 33%', get_h100_hybrid_late),
    }
    
    # Get experiment config
    exp_name, get_config_fn = EXPERIMENTS[args.experiment]
    config = get_config_fn()
    
    print("="*70)
    print(f"EXPERIMENT 10: {exp_name} (Local FLA Clone)")
    print("="*70)
    
    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    
    # If resuming from checkpoint, load config from checkpoint
    resume_checkpoint_path = None
    if args.resume:
        resume_checkpoint_path = Path(args.resume)
        if not resume_checkpoint_path.exists():
            print(f"\n❌ Error: Checkpoint not found: {resume_checkpoint_path}")
            return
        
        # Load config from checkpoint
        print(f"\nLoading config from checkpoint: {resume_checkpoint_path}")
        torch.serialization.add_safe_globals([ExperimentConfig])
        checkpoint_data = torch.load(resume_checkpoint_path, map_location='cpu', weights_only=False)
        config = checkpoint_data['config']
        print(f"✓ Config loaded from checkpoint")
    
    # Extend training steps if requested
    if args.extend_steps:
        original_max_steps = config.max_steps
        config.max_steps = args.extend_steps
        print(f"\n📈 Extending training: {original_max_steps} → {args.extend_steps} steps")
    
    print(f"\nUsing device: {device}")
    print(f"Configuration: {config.hidden_size}d, {config.num_hidden_layers} layers")
    
    # Load data
    print("\n" + "="*70)
    print("Loading Data")
    print("="*70)
    
    # Create DataConfig for the new data loader
    data_config = DataConfig(
        dataset_path="HuggingFaceTB/smollm-corpus",
        dataset_name="cosmopedia-v2",
        tokenizer_name="HuggingFaceTB/SmolLM-135M",
        seq_length=config.max_seq_len,
        num_samples=config.num_documents,
        streaming=True,
        cache_dir="./hf_cache",
    )
    
    # Load dataset and tokenizer using new prepare_lm_dataset
    dataset, tokenizer = prepare_lm_dataset(data_config, return_tokenizer=True)
    config.vocab_size = tokenizer.vocab_size
    
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Dataset prepared with {config.num_documents:,} samples")
    
    # For streaming datasets, we can't determine exact size, so we'll create a generator
    # Split into train and val - we'll use 90/10 split
    # Since it's streaming, we'll just wrap it in DataLoader and let it handle batching
    
    # For now, use simple DataLoader approach since streaming doesn't support splitting easily
    # We'll create two separate data configs for train and val
    print(f"\nCreating train/val split...")
    
    # Calculate number of samples for train/val
    val_split_ratio = 0.1
    train_samples = int(config.num_documents * (1 - val_split_ratio))
    val_samples = config.num_documents - train_samples
    
    print(f"Train samples: {train_samples:,}")
    print(f"Val samples: {val_samples:,}")
    
    # Create train dataset
    train_config = DataConfig(
        dataset_path=data_config.dataset_path,
        dataset_name=data_config.dataset_name,
        tokenizer_name=data_config.tokenizer_name,
        seq_length=config.max_seq_len,
        num_samples=train_samples,
        streaming=True,
        cache_dir=data_config.cache_dir,
    )
    train_dataset,  _ = prepare_lm_dataset(train_config, return_tokenizer=True)
    
    # Create val dataset  
    val_config = DataConfig(
        dataset_path=data_config.dataset_path,
        dataset_name=data_config.dataset_name,
        tokenizer_name=data_config.tokenizer_name,
        seq_length=config.max_seq_len,
        num_samples=val_samples,
        streaming=True,
        cache_dir=data_config.cache_dir,
    )
    val_dataset, _ = prepare_lm_dataset(val_config, return_tokenizer=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        num_workers=0,  # Streaming datasets work better with 0 workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=0,
    )
    
    print(f"✓ Data loaders created")
    
    # Create model
    print("\n" + "="*70)
    print("Creating Model")
    print("="*70)
    
    # FLA requires bfloat16 for optimal performance
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    if dtype == torch.float32:
        print("⚠ Warning: bfloat16 not supported, FLA may have issues with float32")
    
    model = GatedDeltaNetWrapper(config)
    model.print_info()
    model = model.to(device=device, dtype=dtype)
    print(f"\nUsing dtype: {dtype}")
    
    # Verify FLA is available from local clone
    print("\nChecking FLA availability...")
    try:
        import fla
        print("✓ FLA (Flash Linear Attention) is loaded from LOCAL clone")
        fla_module_path = fla.__file__
        print(f"  FLA module path: {fla_module_path}")
        print(f"  FLA version: {fla.__version__ if hasattr(fla, '__version__') else 'unknown'}")
    except ImportError as e:
        print(f"❌ FLA not found: {e}")
        return
    
    # Train
    # Organize results and checkpoints by H100 experiment variant
    exp_base_dir = Path(__file__).parent
    exp_subdir = args.experiment
    results_dir = exp_base_dir / f"results_{exp_subdir}"
    checkpoints_dir = exp_base_dir / f"checkpoints_{exp_subdir}"
    
    print(f"\n📁 Output directories:")
    print(f"   Checkpoints: {checkpoints_dir}")
    print(f"   Results: {results_dir}")
    
    trainer = Trainer(model, config, train_loader, val_loader, device, save_dir=checkpoints_dir)
    
    # Load checkpoint if resuming
    if resume_checkpoint_path:
        trainer.load_checkpoint(resume_checkpoint_path)
    
    results = trainer.train()
    
    # Save results
    results_dir.mkdir(exist_ok=True, parents=True)
    
    results_summary = {
        'experiment_name': exp_name,
        'experiment_variant': args.experiment if args.experiment else 'h100_hybrid_sparse',
        'fla_source': 'local_clone',
        'config': {
            'hidden_size': config.hidden_size,
            'num_layers': config.num_hidden_layers,
            'num_heads': config.num_attention_heads,
            'max_seq_len': config.max_seq_len,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'max_steps': config.max_steps,
        },
        'model_info': model.get_info(),
        'results': {
            'total_time': results['total_time'],
            'best_val_loss': results['best_val_loss'],
            'final_train_loss': results['train_history'][-1]['loss'] if results['train_history'] else None,
            'final_val_metrics': results['val_history'][-1] if results['val_history'] else None,
        },
        'checkpoints_dir': str(checkpoints_dir),
        'results_dir': str(results_dir),
    }
    
    with open(results_dir / 'training_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to: {results_dir / 'training_results.json'}")
    
    # Plot training curves
    plot_training_curves(
        results['train_history'],
        results['val_history'],
        results_dir / 'training_curves.png'
    )
    
    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📊 Results Summary:")
    print(f"   Experiment: {exp_name}")
    print(f"   FLA Source: Local clone (flash-linear-attention/fla)")
    print(f"   Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"   Training Time: {results['total_time']:.1f}s")
    print(f"\n💾 Saved Files:")
    print(f"   Checkpoints: {checkpoints_dir}/")
    print(f"   Results: {results_dir}/")
    print(f"   Training curves: {results_dir / 'training_curves.png'}")
    print(f"   JSON results: {results_dir / 'training_results.json'}")
    print(f"\n🔬 Next Steps:")
    print(f"   • Run benchmarks: python benchmarks/arc_challenge.py --checkpoint {checkpoints_dir}/best_model.pt")
    print(f"   • Generate text: python inference.py")
    if args.experiment:
        print(f"   • Compare models: python benchmarks/compare_models.py --checkpoints {checkpoints_dir}/best_model.pt ...")
    print("="*70)


if __name__ == "__main__":
    main()
