# Research Paper Completion Guide

This guide walks you through completing the research paper after running experiments.

## 📋 Checklist

### Before Training
- [x] Code implemented (baseline + dynamic routing)
- [x] Config files created
- [x] Paper template written
- [ ] Review PAPER.md placeholders
- [ ] Review FIGURE_SPECS.md

### After Training Both Models
- [ ] Baseline trained successfully
- [ ] Dynamic routing trained successfully  
- [ ] No routing collapse (check routing stats)
- [ ] Results saved in `results_baseline/` and `results_dynamic/`

### Generating Figures
- [ ] Run `python generate_figures.py`
- [ ] Review auto-generated figures (2 & 3)
- [ ] Create Figure 1 manually (architecture diagram)
- [ ] Create Figure 4 manually (architecture comparison)
- [ ] Optional: Create Figure 5 (token-level routing heatmap)

### Filling Results
- [ ] Complete Table 1 (main results)
- [ ] Complete Table 2 (routing statistics)
- [ ] Complete Table 3 (computational efficiency)
- [ ] Write Discussion section based on results
- [ ] Update Abstract with key findings
- [ ] Update Conclusion

### Polishing
- [ ] Check all figure references
- [ ] Verify all placeholders filled
- [ ] Proofread entire paper
- [ ] Check math notation consistency
- [ ] Verify citations format
- [ ] Final PDF generation

---

## Step-by-Step Instructions

### Step 1: Train Both Models

```bash
cd experiments/exp11_dynamic_routing

# Train baseline (15-20 min on H100)
python run_experiment.py --config baseline

# Train dynamic routing (15-20 min on H100)
python run_experiment.py --config dynamic

# Compare results
python compare_experiments.py
```

**What to check:**
- Training completed without errors
- Validation loss decreasing (not diverging)
- For dynamic: routing stats show balance (40-60% split)
- Checkpoints saved in respective directories

### Step 2: Extract Results

Open the results files:
```bash
cat results_baseline/training_results.json
cat results_dynamic/training_results.json
```

**Fill in PAPER.md Table 1:**

From the JSON files, extract:
- `results.best_val_loss` → Val Loss column
- Calculate perplexity: `exp(val_loss)`
- `results.final_val_metrics.accuracy` → Accuracy column
- Calculate improvement: `((baseline - dynamic) / baseline) * 100`

**Fill in PAPER.md Table 2:**

From dynamic model's final routing stats (should be in results or logs):
```python
# In run_experiment.py, we logged routing stats
# Look for: "Layer 1 - GDN: X%, Softmax: Y%"

Layer 1 GDN%: ____%
Layer 1 Softmax%: ____%  
Layer 2 GDN%: ____%
Layer 2 Softmax%: ____%
```

Calculate entropy:
```python
gdn_pct = 0.XX  # Convert percentage to decimal
attn_pct = 1 - gdn_pct
entropy = -(gdn_pct * np.log(gdn_pct) + attn_pct * np.log(attn_pct))
# Max entropy for binary = 0.693
```

Status:
- 40-60%: "Balanced"
- >90% one side: "Collapsed"

**Fill in PAPER.md Table 3:**

From training logs or results:
- Training time: Check terminal output or results JSON
- Tokens/second: `(total_tokens_trained) / (training_time_seconds)`
- Overhead: `dynamic_time / baseline_time`

### Step 3: Generate Automatic Figures

```bash
python generate_figures.py
```

This creates:
- `figures/figure2_training_curves.png` (4 subplots)
- `figures/figure3_routing_distribution.png` (2 subplots)

**Review these figures:**
- Do training curves look reasonable?
- Did dynamic model converge?
- Did routing stay balanced or collapse?
- Are labels readable?

### Step 4: Create Manual Figures

#### Figure 1: Architecture Diagram

Use the specifications in `FIGURE_SPECS.md` section "Figure 1".

**Recommended tools:**
1. **Draw.io** (https://app.diagrams.net/)
   - Free, web-based
   - Import/export PNG, SVG, PDF
   - Template: "Flowchart"

2. **Figma** (https://figma.com)
   - Professional design tool
   - Good for precise layouts

3. **TikZ** (LaTeX)
   - If you're comfortable with LaTeX
   - Beautiful vector graphics

**Quick steps in Draw.io:**
1. Create new diagram
2. Add rectangles for each layer
3. Color code: Blue for GDN, Orange for Softmax, Green for Router
4. Add arrows showing data flow
5. Add text annotations
6. Export as PNG (300 DPI) and PDF

#### Figure 4: Architecture Comparison

Create side-by-side comparison:
- Left: Baseline (simple 4-layer stack)
- Right: Dynamic (with router and branching)

Use same tool as Figure 1 for consistency.

### Step 5: Write Discussion Section

Based on your results, choose the appropriate discussion:

#### If Dynamic Routing Won (Lower Val Loss):

```markdown
## 6. Discussion

Our results demonstrate that dynamic per-token routing between linear
and softmax attention achieves [X%] improvement over static layer 
assignment. This validates our hypothesis that different tokens benefit
from different attention mechanisms.

### 6.1 Performance Analysis

The dynamic model achieved a validation loss of [Y.YYY], compared to
[X.XXX] for the baseline, representing a [Z%] improvement. This suggests
that token-level heterogeneity in attention requirements exists and can
be exploited.

### 6.2 Routing Patterns

Analysis of routing statistics reveals:
- Layer 1: [%GDN] tokens routed to GDN, [%Softmax] to Softmax
- Layer 2: [%GDN] tokens routed to GDN, [%Softmax] to Softmax

[Interpret what this means - did it learn meaningful patterns?]

### 6.3 Load Balancing Success

The load balancing mechanism successfully prevented routing collapse,
maintaining [X%] balanced distribution across both layers. This 
demonstrates that α=0.01 is sufficient for our architecture.
```

#### If Baseline Won (Lower Val Loss):

```markdown
## 6. Discussion

Our results show that the static baseline achieves comparable or better
performance than dynamic routing. The baseline achieved [X.XXX] validation
loss versus [Y.YYY] for dynamic routing, a difference of [Z%].

### 6.1 Why Static Assignment Suffices

Several factors may explain this result:

1. **Routing Overhead:** The router network adds [X%] parameters and
   computational overhead that may outweigh adaptive benefits
   
2. **Layer-Level Granularity:** Prior work [5] found that layer-level
   decisions suffice; token-level may be unnecessarily fine-grained
   
3. **Training Complexity:** The load balancing constraint may limit
   the router's ability to find optimal solutions

### 6.2 Routing Collapse Analysis

[If collapse occurred:]
Despite load balancing, routing collapsed to [GDN/Softmax] for [XX%]
of tokens. This suggests α=0.01 was insufficient. Future work should
explore stronger balancing (α=0.05) or adaptive α.
```

### Step 6: Update Abstract

Fill in the results summary in the abstract:

```markdown
[BEFORE:]
[RESULTS PLACEHOLDER: Dynamic routing achieves X% improvement...]

[AFTER - if won:]
Dynamic routing achieves 3.2% improvement over baseline (val loss 4.156
vs 4.294) while maintaining 48/52% balanced routing distribution across
layers. This demonstrates that token-specific attention selection can
improve language modeling performance.

[AFTER - if lost/tied:]
While dynamic routing achieved balanced token distribution (47/53% split),
it performed comparably to static baseline (val loss 4.294 vs 4.287).
This suggests that layer-level attention decisions may be sufficient
granularity for 4-layer models.
```

### Step 7: Final Checks

- [ ] All `[RESULTS PLACEHOLDER]` tags removed
- [ ] All `[TABLE X HERE]` have actual tables
- [ ] All `[FIGURE X HERE]` reference existing figures
- [ ] Math notation consistent ($, $$, or LaTeX)
- [ ] Citations numbered correctly [1], [2], etc.
- [ ] Figure captions written
- [ ] Table captions written
- [ ] Acknowledgments section (if needed)

### Step 8: Generate PDF

If using Markdown → PDF:

**Option 1: Pandoc**
```bash
pandoc PAPER.md -o PAPER.pdf \
    --from markdown \
    --template=template.tex \
    --pdf-engine=xelatex \
    --bibliography=references.bib \
    --citeproc
```

**Option 2: VSCode Markdown PDF Extension**
1. Install "Markdown PDF" extension
2. Open PAPER.md
3. Right-click → "Markdown PDF: Export (pdf)"

**Option 3: Copy to LaTeX**
1. Convert markdown to LaTeX
2. Use Overleaf or local LaTeX installation
3. Compile to PDF

---

## Quick Reference: File Locations

```
experiments/exp11_dynamic_routing/
├── PAPER.md                    # Main paper (fill this!)
├── FIGURE_SPECS.md             # Figure specifications
├── generate_figures.py         # Auto-generate some figures
├── results_baseline/
│   └── training_results.json   # Baseline results
├── results_dynamic/
│   └── training_results.json   # Dynamic results
└── figures/                    # Generated figures go here
    ├── figure1_architecture.png       # Manual
    ├── figure2_training_curves.png    # Auto-generated
    ├── figure3_routing_dist.png       # Auto-generated
    └── figure4_comparison.png         # Manual
```

---

## Example Results Fill-In

### Before:
```markdown
| Model | Val Loss ↓ | Perplexity ↓ | Accuracy ↑ |
|-------|------------|--------------|------------|
| Baseline | **X.XXX** | **XX.XX** | **XX.X%** |
| Dynamic | **X.XXX** | **XX.XX** | **XX.X%** |
```

### After:
```markdown
| Model | Val Loss ↓ | Perplexity ↓ | Accuracy ↑ |
|-------|------------|--------------|------------|
| Baseline | 4.294 | 73.2 | 31.4% |
| **Dynamic** | **4.156** | **64.0** | **32.1%** |
| Δ | +3.2% | +12.6% | +0.7 pp |
```

---

## Troubleshooting

**Q: Routing collapsed to >95% one mechanism**
- Try `--config dynamic_aggressive` (α=0.05)
- Or manually edit `config.py` to increase `load_balance_alpha`

**Q: Training diverged (loss → NaN)**
- Lower learning rate
- Check for numerical instability in Gumbel-Softmax temperature
- Verify gradient clipping is enabled

**Q: Figures not generating**
- Check that results JSON files exist
- Verify training history was saved
- Check `generate_figures.py` data extraction code matches your results format

**Q: Can't reproduce results**
- Verify same random seed (42)
- Check data loading is deterministic
- Ensure same GPU type (results may vary across hardware)

---

## Contact / Questions

If you're uncertain about any step:
1. Review FIGURE_SPECS.md for detailed figure instructions
2. Check SUMMARY.md for architecture overview
3. Look at exp7 results as reference for formatting

Good luck with your research! 🚀
