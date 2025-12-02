"""
Load and use trained Gated DeltaNet model for inference
Using local FLA clone from flash-linear-attention/fla
"""

import torch
import sys
import os
from pathlib import Path

# Fix tokenizer warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

# Add flash-linear-attention to path for local imports
fla_path = os.path.join(root_dir, 'flash-linear-attention')
if fla_path not in sys.path:
    sys.path.insert(0, fla_path)

from experiments.exp10_local_fla_hybrid.models import GatedDeltaNetWrapper
from experiments.exp10_local_fla_hybrid.config import ExperimentConfig
from transformers import AutoTokenizer


def load_model(checkpoint_path, device='cuda', dtype=torch.bfloat16):
    """
    Load a trained model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file (e.g., 'checkpoints/best_model.pt')
        device: Device to load model on
        dtype: Data type for model (must be fp16 or bf16 for Flash Attention)
    
    Returns:
        model: Loaded model ready for inference
        config: Model configuration
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Using dtype: {dtype}")
    
    # Load checkpoint (allowlist ExperimentConfig for PyTorch 2.6+ safety)
    torch.serialization.add_safe_globals([ExperimentConfig])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get config
    config = checkpoint['config']
    
    # Print hybrid config info
    if hasattr(config, 'attn_config') and config.attn_config is not None:
        print(f"✓ Hybrid model detected")
        print(f"  Attention layers: {config.attn_config.get('layers', [])}")
    else:
        print(f"✓ Pure DeltaNet model")
    
    # Create model
    model = GatedDeltaNetWrapper(config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and convert to appropriate dtype
    # IMPORTANT: Flash Attention requires fp16 or bf16!
    model = model.to(device=device, dtype=dtype)
    model.eval()
    
    print(f"✅ Model loaded successfully!")
    print(f"   Steps trained: {checkpoint['global_step']}")
    print(f"   Best val loss: {checkpoint['best_val_loss']:.4f}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   FLA Source: Local clone (flash-linear-attention/fla)")
    
    return model, config


def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0, top_k=50, device='cuda', dtype=torch.bfloat16):
    """
    Generate text from a prompt
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        prompt: Input text prompt
        max_length: Maximum length to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling parameter
        device: Device
        dtype: Data type for computations
    
    Returns:
        generated_text: Generated text string
    """
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    print(f"\nPrompt: {prompt}")
    print(f"Generating {max_length} tokens...")
    
    with torch.no_grad():
        # Use autocast for mixed precision if using fp16/bf16
        with torch.amp.autocast(device_type='cuda', dtype=dtype):
            for _ in range(max_length):
                # Forward pass
                outputs = model(input_ids)
                logits = outputs.logits
                
                # Get logits for last token (convert to float32 for sampling stability)
                next_token_logits = logits[:, -1, :].float() / temperature
                
                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][:, -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Stop if we hit max length or EOS
                if input_ids.shape[1] >= 1024:  # Max context length
                    break
    
    # Decode
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    return generated_text


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference with trained model (Local FLA)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint (default: checkpoints_h100_hybrid_sparse/best_model.pt)')
    parser.add_argument('--experiment', type=str, default='h100_hybrid_sparse',
                        help='Experiment variant to load (default: h100_hybrid_sparse)')
    args = parser.parse_args()
    
    print("="*70)
    print("Hybrid DeltaNet + Attention Inference (Local FLA Clone)")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Choose dtype based on GPU capabilities
    # Flash Attention requires fp16 or bf16
    if torch.cuda.is_available():
        # Use bfloat16 if available (better for training stability)
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print(f"Using bfloat16 (GPU supports it)")
        else:
            dtype = torch.float16
            print(f"Using float16 (GPU doesn't support bfloat16)")
    else:
        dtype = torch.float32
        print(f"Using float32 (CPU mode)")
    
    print(f"Device: {device}")
    
    # Load model
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        # Use default checkpoint from experiment variant
        checkpoint_path = Path(__file__).parent / f"checkpoints_{args.experiment}" / "best_model.pt"
    
    if not checkpoint_path.exists():
        print(f"\n❌ Checkpoint not found: {checkpoint_path}")
        print(f"Please train the model first:")
        print(f"  python run_experiment.py --experiment {args.experiment}")
        return
    
    model, config = load_model(checkpoint_path, device, dtype=dtype)
    
    # Load tokenizer (same as used in training)
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    print(f"✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # Generate text
    print("\n" + "="*70)
    print("Text Generation Examples")
    print("="*70)
    
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In the year 2050",
    ]
    
    for prompt in prompts:
        print("\n" + "-"*70)
        generated = generate_text(
            model, 
            tokenizer, 
            prompt, 
            max_length=50,
            temperature=0.8,
            top_k=40,
            device=device,
            dtype=dtype
        )
        print(f"\nGenerated:\n{generated}")
    
    print("\n" + "="*70)
    print("✅ Inference completed!")
    print("="*70)


if __name__ == "__main__":
    main()
