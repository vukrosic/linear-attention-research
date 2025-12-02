# Experiment 10 vs Experiment 7 - Comparison

## Overview
Both experiments test hybrid DeltaNet architectures, but differ in how they import the FLA library.

## Key Differences

| Aspect | Experiment 7 | Experiment 10 |
|--------|--------------|---------------|
| **FLA Source** | Pip-installed package | Local cloned repository |
| **Import Method** | `from fla.models import ...` | Path modification + `from fla.models import ...` |
| **Installation** | `pip install git+https://github.com/fla-org/flash-linear-attention` | Use existing `flash-linear-attention/` clone |
| **Development** | Requires reinstall for changes | Changes immediately available |
| **Use Case** | Production training | Development & debugging |

## Import Differences

### Experiment 7 (models.py)
```python
# Use FLA's Gated DeltaNet implementation (supports hybrid with attention)
from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
```

### Experiment 10 (models.py)
```python
# Add flash-linear-attention to path for local imports
fla_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'flash-linear-attention')
if fla_path not in sys.path:
    sys.path.insert(0, fla_path)

# Use FLA's Gated DeltaNet implementation from local clone (supports hybrid with attention)
from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
```

## When to Use Each

### Use Experiment 7 When:
- Running production training
- Using stable FLA release
- Don't need to modify FLA source
- Want cleaner dependencies

### Use Experiment 10 When:
- Debugging FLA internals
- Testing FLA modifications
- Developing new FLA features
- Want to avoid pip installation
- Need to track FLA source changes

## Files That Changed

### Identical Files (can be symlinked):
- `config.py` (copied directly)

### Modified Files:
- `models.py` - Added sys.path modification for local imports
- `run_experiment.py` - Added sys.path modification, updated documentation
- `requirements.txt` - Removed FLA from dependencies
- `README.md` - New documentation explaining local import usage
- `__init__.py` - Updated description

## Expected Results

Both experiments should produce **identical results** when using the same FLA code:
- Same model architectures
- Same training dynamics
- Same performance metrics

The only difference is the import source, not the code being executed.

## Migration Between Experiments

### From Exp 7 → Exp 10:
```bash
# Ensure FLA clone exists
git clone https://github.com/fla-org/flash-linear-attention
cd experiments/exp10_local_fla_hybrid
python run_experiment.py  # Will use local clone
```

### From Exp 10 → Exp 7:
```bash
# Install FLA from pip
pip install git+https://github.com/fla-org/flash-linear-attention
cd experiments/exp7_hybrid_deltanet_ablation
python run_experiment.py  # Will use pip package
```

## Verification

Check which FLA is being used:
```python
import sys
sys.path.insert(0, 'flash-linear-attention')  # For exp10
import fla
print(f"FLA location: {fla.__file__}")
```

**Exp 7:** `/path/to/site-packages/fla/__init__.py`  
**Exp 10:** `/path/to/flash-linear-attention/fla/__init__.py`
