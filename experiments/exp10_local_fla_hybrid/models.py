"""
Gated DeltaNet Model for Training using FLA
Using FLA's optimized DeltaNet implementation with Triton kernels
Imports from local cloned repository: flash-linear-attention/fla
"""

import torch
import torch.nn as nn
from typing import Optional
import sys
import os

# Add flash-linear-attention to path for local imports
fla_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'flash-linear-attention')
if fla_path not in sys.path:
    sys.path.insert(0, fla_path)

# Use FLA's Gated DeltaNet implementation from local clone (supports hybrid with attention)
from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM


def create_gated_deltanet_model(config):
    """
    Create a DeltaNet model using FLA's optimized implementation
    Supports hybrid models with standard attention layers
    
    Args:
        config: ExperimentConfig with model parameters
    
    Returns:
        DeltaNetForCausalLM model instance
    """
    # Convert ExperimentConfig to GatedDeltaNetConfig
    deltanet_config = GatedDeltaNetConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        
        # DeltaNet specific parameters
        expand_k=getattr(config, 'expand_k', 1.0),  # Key expansion ratio
        expand_v=getattr(config, 'expand_v', 1.0),  # Value expansion ratio
        
        # MLP configuration
        hidden_ratio=getattr(config, 'hidden_ratio', 4),  # MLP expansion ratio
        intermediate_size=config.intermediate_size if hasattr(config, 'intermediate_size') else None,
        
        # Regularization
        norm_eps=config.rms_norm_eps,
        
        # Optimization flags
        fuse_norm=True,  # Use fused normalization
        fuse_cross_entropy=True,  # Use fused cross entropy
        
        # Standard configs
        max_position_embeddings=config.max_position_embeddings,
        initializer_range=0.02,
        use_cache=True,
        
        # Tokenizer configs
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    
    # Add hybrid attention configuration if specified
    if hasattr(config, 'attn_config') and config.attn_config is not None:
        # Set up standard attention layers at specified positions
        deltanet_config.attn = {
            'layers': config.attn_config.get('layers', []),
            'num_heads': config.num_attention_heads,
            'num_kv_heads': config.attn_config.get('num_kv_heads', config.num_attention_heads),
            'window_size': config.attn_config.get('window_size', 2048),
            'qkv_bias': config.attn_config.get('qkv_bias', False),
            'rope_theta': config.attn_config.get('rope_theta', 10000.0),
        }
    
    # Create model using FLA's Gated DeltaNet implementation
    model = GatedDeltaNetForCausalLM(deltanet_config)
    
    return model


def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'total_millions': total_params / 1_000_000,
        'trainable_millions': trainable_params / 1_000_000,
    }


def verify_model_architecture(model, config):
    """
    Verify that the model has the expected architecture
    Returns dict with architecture info and verification status
    """
    # Detect actual model type
    model_class_name = model.__class__.__name__
    is_deltanet = 'DeltaNet' in model_class_name
    
    # Count layers
    actual_num_layers = len(model.model.layers)
    expected_num_layers = config.num_hidden_layers
    
    # Verify layer structure and identify layer types
    layer_types = []
    all_layers_valid = True
    deltanet_layers = []
    attention_layers = []
    
    for i, layer in enumerate(model.model.layers):
        has_mlp = hasattr(layer, 'mlp')
        layer_type = layer.__class__.__name__
        
        # Detect the mixer type (GatedDeltaNet or standard Attention)
        # GatedDeltaNetBlock uses 'attn' attribute (can be either GatedDeltaNet or Attention)
        mixer_type = None
        if hasattr(layer, 'attn'):
            mixer = layer.attn
            mixer_class = mixer.__class__.__name__
            if 'DeltaNet' in mixer_class:
                mixer_type = 'GatedDeltaNet'
                deltanet_layers.append(i)
            elif 'Attention' in mixer_class:
                mixer_type = 'Attention'
                attention_layers.append(i)
            else:
                mixer_type = mixer_class
        elif hasattr(layer, 'mixer'):
            # Fallback for other layer types
            mixer = layer.mixer
            mixer_class = mixer.__class__.__name__
            if 'DeltaNet' in mixer_class:
                mixer_type = 'DeltaNet'
                deltanet_layers.append(i)
            elif 'Attention' in mixer_class:
                mixer_type = 'Attention'
                attention_layers.append(i)
            else:
                mixer_type = mixer_class
        
        layer_info = {
            'idx': i,
            'type': layer_type,
            'mixer_type': mixer_type,
            'has_mlp': has_mlp,
            'valid': mixer_type is not None and has_mlp,
        }
        layer_types.append(layer_info)
        
        if not (mixer_type is not None and has_mlp):
            all_layers_valid = False
    
    # Check if model is hybrid
    is_hybrid = len(attention_layers) > 0 and len(deltanet_layers) > 0
    
    # Overall verification status
    verification_passed = (
        is_deltanet and
        actual_num_layers == expected_num_layers and
        all_layers_valid
    )
    
    info = {
        'verification_passed': verification_passed,
        'model_type': model_class_name,
        'is_deltanet': is_deltanet,
        'is_hybrid': is_hybrid,
        'num_layers': actual_num_layers,
        'expected_num_layers': expected_num_layers,
        'layers_match': actual_num_layers == expected_num_layers,
        'all_layers_valid': all_layers_valid,
        'deltanet_layers': deltanet_layers,
        'attention_layers': attention_layers,
        'layer_types': layer_types,
    }
    
    return info


class GatedDeltaNetWrapper(nn.Module):
    """
    Wrapper for FLA's DeltaNet model with convenience methods
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = create_gated_deltanet_model(config)
        self.param_info = count_parameters(self.model)
        self.arch_info = verify_model_architecture(self.model, config)
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass through the model"""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def get_info(self):
        """Get model information"""
        return {
            'parameters': self.param_info,
            'architecture': self.arch_info,
            'config': {
                'hidden_size': self.config.hidden_size,
                'num_layers': self.config.num_hidden_layers,
                'num_heads': self.config.num_attention_heads,
                'max_seq_len': self.config.max_position_embeddings,
                'vocab_size': self.config.vocab_size,
            }
        }
    
    def print_info(self):
        """Print model information"""
        info = self.get_info()
        
        print("="*70)
        print("Gated DeltaNet Model Information (FLA Local Clone)")
        print("="*70)
        
        print("\nParameters:")
        print(f"  Total: {info['parameters']['total']:,} ({info['parameters']['total_millions']:.2f}M)")
        print(f"  Trainable: {info['parameters']['trainable']:,} ({info['parameters']['trainable_millions']:.2f}M)")
        
        print("\nConfiguration:")
        for key, value in info['config'].items():
            print(f"  {key}: {value}")
        
        print("\nArchitecture:")
        arch = info['architecture']
        verification_status = "✓ PASSED" if arch['verification_passed'] else "✗ FAILED"
        print(f"  Verification: {verification_status}")
        print(f"  Model Type: {arch['model_type']}")
        print(f"  Is DeltaNet: {'✓' if arch['is_deltanet'] else '✗'}")
        
        # Show hybrid status
        if arch.get('is_hybrid', False):
            print(f"  Model Mode: HYBRID (DeltaNet + Standard Attention)")
            print(f"  DeltaNet Layers: {len(arch['deltanet_layers'])} layers -> {arch['deltanet_layers']}")
            print(f"  Attention Layers: {len(arch['attention_layers'])} layers -> {arch['attention_layers']}")
        else:
            print(f"  Model Mode: Pure DeltaNet")
        
        print(f"  Number of layers: {arch['num_layers']} (expected: {arch['expected_num_layers']})")
        
        print("\nLayer Types:")
        for layer_info in arch['layer_types']:
            status = "✓" if layer_info['valid'] else "✗"
            mixer = layer_info.get('mixer_type', 'unknown')
            mlp_status = "mlp" if layer_info['has_mlp'] else "no-mlp"
            print(f"  Layer {layer_info['idx']}: {layer_info['type']} (mixer={mixer}, {mlp_status}) {status}")
        
        print("\nFLA Optimizations:")
        print("  ✓ Fused normalization (RMSNorm)")
        print("  ✓ Fused cross entropy loss")
        print("  ✓ Triton-optimized kernels")
        if arch.get('is_hybrid', False):
            print("  ✓ Hybrid architecture with standard attention layers")
        else:
            print("  ✓ Chunk-based DeltaNet computation")
        
        print("  ✓ Using LOCAL FLA clone (flash-linear-attention/fla)")
        
        print("="*70)
