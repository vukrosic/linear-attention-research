# Blueberry LLM

**Open Superintelligence Lab** - Open research for everyone. We publish all of our research for the sake of accelerating science. Learn real AI research from a real research lab.

## Quick Start

```bash
pip install -r requirements.txt

python train_moe.py
```

## About

Purpose of this repository is to research better, faster, smarter LLMs.

This repository contains cutting-edge language model experiments and architectures. We believe scientists do their best work when given freedom to explore, so this is a space for your independent research and discovery.

Fork this repository, create a new experiment in `experiments/` folder, then create a pull request to merge it back.

## Experiments

### Experiment 11: Dynamic Routing

**Research Question**: Can a model dynamically choose between linear attention (GDN) and softmax attention per-token at each layer to achieve better performance than static layer assignments?

#### 🚀 Quick Start

**1. Install Dependencies**
```bash
cd experiments
pip install -r requirements.txt
```

**2. Train Baseline (Static)**
```bash
python run_experiment.py --config baseline
```
This trains a 4-layer model: [GDN, GDN, GDN, Softmax]

**3. Train Dynamic Routing**
```bash
python run_experiment.py --config dynamic
```
This trains a 4-layer model: [GDN, ROUTED, ROUTED, Softmax]

**4. Compare Results**
```bash
python compare_experiments.py
```

#### 📊 Expected Runtime

- **H100**: ~15-20 minutes per config (1000 steps)
- **A100**: ~25-30 minutes per config
- **RTX 4090**: ~40-50 minutes per config

#### 🔧 Troubleshooting

**Routing Collapse**: If dynamic routing always chooses one mechanism (>90%), try:
```bash
python run_experiment.py --config dynamic_aggressive
```

**Out of Memory**: Reduce batch size in `config.py`:
```python
batch_size=24  # Instead of 48
```

#### 📈 Success Indicators

**Dynamic routing success:**
- Validation loss < Baseline loss (even by 1-2%)
- Routing is balanced (~40-60% split per layer)
- Load balance loss decreases over training

**Routing collapse indicators:**
- One mechanism used >90% of the time
- Load balance loss stays high
- Dynamic worse than baseline


## Getting Started

1. **Fork this repository** - Click the "Fork" button at the top right of this page to create your own copy
2. Clone your fork: `git clone --recursive https://github.com/vukrosic/linear-attention-research.git`
   - **If you forgot `--recursive`**: Run `git submodule update --init --recursive` to initialize the submodules
3. Install dependencies: `pip install -r requirements.txt`
4. Read `CONTRIBUTING.md` for contribution guidelines
5. Create your own experiment and merge it
6. Explore the `experiments/` folder for ongoing research and inspiration
7. Once you finish with your research, create a pull request to merge it back to this repo

## Philosophy

We don't prescribe what to research. Instead, we provide:
- Freedom to explore interesting ideas
- Infrastructure to test hypotheses
- A collaborative environment for learning

## Structure

- **`experiments/`** - Research experiments with their own documentation
- **`models/`** - Model architectures and implementations (DeepSeek, Qwen3-Next)
- **`training/`** - Training scripts and utilities
- **`configs/`** - Configuration files

## Contributing

See `CONTRIBUTING.md` for guidelines on how to contribute to this project.