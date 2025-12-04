# Figure Specifications for Paper

This document provides detailed specifications for creating all figures in the research paper.

## Figure 1: Architecture Diagram (MOST IMPORTANT)

### Layout
- **Canvas:** 1200px width × 800px height
- **Style:** Clean, professional diagram with boxes and arrows
- **Colors:** 
  - GDN blocks: Light blue (#4A90E2) with darker border (#2E5C8A)
  - Softmax blocks: Light orange (#F5A623) with darker border (#D68910)
  - Router: Light green (#7ED321) with darker border (#5FA118)
  - Background: White or very light gray (#F8F9FA)

### Components (from top to bottom):

**1. Input Layer:**
- Box: "Input Embeddings" (gray, 150px wide × 50px tall)
- Text: "x ∈ ℝ^(B×L×768)" in small font below

**2. Layer 0 (Fixed GDN):**
- Single blue box: "Layer 0: GDN" (200px wide × 80px tall)
- Label: "O(n) complexity" in small italics
- Annotation: "Fixed - stability" on right side
- Arrow down to Router

**3. Router Network:**
- Green box: "Router Network" (250px wide × 60px tall)
- Equation inside: "R: ℝ^768 → ℝ^(2×2)"
- Label below: "Parallel routing for Layers 1 & 2"
- Two arrows out: one to Layer 1, one to Layer 2
- Each arrow labeled with routing weights: "r₁[0], r₁[1]" and "r₂[0], r₂[1]"

**4. Layer 1 (Routed):**
- Two parallel boxes side-by-side (150px wide × 80px each):
  - Left: Blue "GDN₁" with "O(n)" label
  - Right: Orange "Softmax₁" with "O(n²)" label
- Between them: text "vs"
- Below both: Box labeled "Weighted Sum: h₁ = r₁[0]·GDN + r₁[1]·Attn"
- Routing weights shown as percentages: "40%" above GDN, "60%" above Softmax (example values)

**5. Layer 2 (Routed):**
- Same structure as Layer 1
- Different example percentages: "55%" above GDN, "45%" above Softmax

**6. Layer 3 (Fixed Softmax):**
- Single orange box: "Layer 3: Softmax" (200px wide × 80px tall)
- Label: "O(n²) complexity"  
- Annotation: "Fixed - global context" on right side

**7. Output:**
- Box: "LM Head → Logits" (150px wide × 50px tall)
- Text: "∈ ℝ^(B×L×50257)" below

### Arrows:
- Solid arrows for data flow
- Dashed arrows for routing control
- Arrow thickness: 3px for main flow, 2px for routing

### Annotations:
- Add small box in corner showing:
  - Total layers: 4
  - Routed layers: 2 (Layers 1-2)
  - Fixed layers: 2 (Layers 0, 3)
  - Parameters: ~50M

---

## Figure 2: Training Curves (2×2 Grid)

### Specifications:
- **Size:** 1400px × 1000px total
- **Grid:** 2 rows × 2 columns
- **Padding:** 50px between subplots

### Subplot 1 (Top-Left): Training Loss
```
X-axis: "Training Steps" (0, 200, 400, 600, 800, 1000)
Y-axis: "Training Loss" (auto-scale based on data)
Lines:
  - Baseline: Solid blue (#4A90E2), linewidth=2
  - Dynamic: Dashed orange (#F5A623), linewidth=2
Legend: Top-right corner
Grid: Light gray dotted lines
Title: "Training Loss Over Time"
```

### Subplot 2 (Top-Right): Validation Loss
```
X-axis: "Training Steps" (0, 50, 100, ..., 1000, same as eval intervals)
Y-axis: "Validation Loss"
Lines: Same style as Subplot 1
Markers: Circles at each evaluation point
Best point: Yellow star (large) marking lowest loss
Title: "Validation Loss (Primary Metric)"
```

### Subplot 3 (Bottom-Left): Validation Accuracy
```
X-axis: "Training Steps"
Y-axis: "Next-Token Accuracy (%)"
Lines: Same style
Title: "Validation Accuracy"
Y-axis range: Should show percentage (e.g., 20-40%)
```

### Subplot 4 (Bottom-Right): Load Balance Loss
```
X-axis: "Training Steps"
Y-axis: "Load Balance Loss"
Lines:
  - Baseline: Flat gray line at 0.0 (N/A)
  - Dynamic: Orange line showing actual load balance loss
Title: "Load Balancing Loss (Dynamic Only)"
Annotation: "Target: Decreasing trend indicates successful balancing"
Y-axis: Start at 0, auto-scale upward
```

**Overall Figure Title:** "Training Dynamics: Baseline vs Dynamic Routing"

---

## Figure 3: Routing Distribution Over Time

### Specifications:
- **Size:** 1200px × 600px
- **Layout:** 2 horizontal subplots (stacked vertically)

### Subplot 1: Layer 1 Routing
```
X-axis: "Training Steps" (0-1000)
Y-axis: "Routing Percentage (%)" (0-100)
Two stacked area plots OR two lines:
  - Bottom/First: GDN percentage (blue gradient/line)
  - Top/Second: Softmax percentage (orange gradient/line)
  - They should sum to 100%
Horizontal reference lines:
  - 50% (black dotted line, labeled "Balanced")
  - 90% and 10% (red dotted lines, labeled "Collapse threshold")
Title: "Layer 1: Routing Distribution Evolution"
Y-grid: Every 10%
```

### Subplot 2: Layer 2 Routing
```
Same format as Layer 1
Title: "Layer 2: Routing Distribution Evolution"
```

**Interpretation Guide (in caption):**
- "Converging to 50/50: Balanced routing (ideal)"
- "Converging to >90% one side: Routing collapse (undesired)"
- "Fluctuating: Continued exploration (depends on temperature annealing)"

---

## Figure 4: Architecture Comparison (Side-by-Side)

### Layout:
- **Size:** 1400px × 600px
- **Split:** 50/50 vertical split

### Left Side: Baseline (Static)
```
Title: "Baseline: Static Assignment (75% GDN, 25% Softmax)"

Stack of 4 boxes (equal height):
- Layer 0: Blue "GDN" (200px wide)
- Layer 1: Blue "GDN"
- Layer 2: Blue "GDN"  
- Layer 3: Orange "Softmax"

Annotations:
- Total FLOPs: ~X GFLOPS (calculate based on architecture)
- No routing overhead
- Static layer assignment
```

### Right Side: Dynamic
```
Title: "Dynamic Routing: Adaptive per Token"

Similar stack but with differences:
- Layer 0: Blue "GDN (fixed)"
- Layer 1: Half-blue, half-orange "GDN | Softmax (routed)" with small router icon
- Layer 2: Same as Layer 1
- Layer 3: Orange "Softmax (fixed)"

Annotations:
- Total FLOPs: ~X GFLOPS (similar to baseline at 50/50 routing)
- Router overhead: ~2% parameters
- Dynamic routing: 2 layers
```

**Central Comparison:**
- Arrow between them labeled "vs"
- Box below comparing:
  - Parameters: ~50M (both)
  - Routing: None vs Token-level (Layers 1-2)
  - Complexity: Fixed vs Adaptive

---

## Figure 5 (Optional): Token-Level Routing Heatmap

### Specifications:
- **Size:** 1000px × 400px per example
- **Show:** 3-5 diverse example sentences

### Format:
```
For each sentence:
- Tokens displayed horizontally (words/subwords)
- 2 rows below each token:
  - Row 1: Layer 1 routing (blue=GDN, orange=Softmax)
  - Row 2: Layer 2 routing (blue=GDN, orange=Softmax)
- Color intensity based on confidence (darker = more confident)

Example layout:
     The    quick   brown   fox    jumps   over    the     lazy    dog
L1: [Blue] [Blue] [Orange][Blue] [Orange][Blue] [Blue] [Orange][Orange]
L2: [Blue][Orange][Orange][Blue]  [Blue] [Blue][Orange][Orange][Orange]

Color legend:
- Solid blue: >80% GDN
- Solid orange: >80% Softmax
- Mixed (gradient): 40-60% range
```

**Analysis to include:**
- Do common words (the, a) route consistently to GDN?
- Do rare words/entities route to Softmax?
- Do adjacent tokens show similar routing?
- Layer-specific patterns?

---

## Table Specifications

### Table 1: Main Results
```
| Model              | Val Loss ↓ | Perplexity ↓ | Accuracy ↑ | Collapse? |
|--------------------|------------|--------------|------------|-----------|
| Baseline (Static)  | 4.XXX      | XX.XX        | XX.X%      | N/A       |
| Dynamic Routing    | 4.YYY      | YY.YY        | YY.Y%      | No/Yes    |
| Δ Improvement      | +Z.Z%      | +Z.Z%        | +Z.Z pp    | -         |

Format:
- Bold best results
- Use ↓ for metrics where lower is better
- Use ↑ for metrics where higher is better
- Show percentage improvement in bottom row
- pp = percentage points (for accuracy)
```

### Table 2: Routing Statistics
```
| Layer   | GDN %  | Softmax % | Entropy | Status     |
|---------|--------|-----------|---------|------------|
| Layer 1 | XX.X%  | XX.X%     | 0.XXX   | Balanced   |
| Layer 2 | YY.Y%  | YY.Y%     | 0.YYY   | Balanced   |

Status indicators:
- "Balanced" if 40-60% for both  
- "Collapsed→GDN" if >90% GDN
- "Collapsed→Softmax" if >90% Softmax
- "Slightly skewed" if 60-90%

Entropy:
- Maximum (perfectly balanced): 0.693 for binary choice
- Closer to 0.693 = more balanced
- Closer to 0.0 = collapsed
```

### Table 3: Computational Efficiency
```
| Model     | Tokens/sec ↑ | Time ↓  | Overhead |
|-----------|--------------|---------|----------|
| Baseline  | X,XXX        | XX min  | 1.0×     |
| Dynamic   | Y,YYY        | YY min  | X.XX×    |

Calculate overhead = Dynamic_time / Baseline_time
Expected: 1.05-1.15× (small router overhead)
```

---

## Color Palette (Consistent across all figures)

```
Primary Colors:
- GDN/Linear:     #4A90E2 (blue)
- Softmax/Attn:   #F5A623 (orange)
- Router:         #7ED321 (green)

Supporting Colors:
- Baseline line:  #4A90E2 (solid)
- Dynamic line:   #F5A623 (dashed)
- Grid:           #E0E0E0 (light gray)
- Background:     #F8F9FA (very light gray)
- Text:           #333333 (dark gray)
- Emphasis:       #FFD700 (gold, for best results star)

Danger/Warning:
- Collapse threshold: #E74C3C (red)
```

---

## Tools for Creating Figures

**Recommended:**
1. **Architecture diagrams:** Draw.io, Figma, or TikZ (LaTeX)
2. **Training curves:** matplotlib, seaborn (Python)
3. **Heatmaps:** seaborn.heatmap()
4. **Tables:** LaTeX booktabs package

**Python Example for Figure 2:**
```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Load data from training logs
baseline_train = np.load('results_baseline/train_history.npy')
dynamic_train = np.load('results_dynamic/train_history.npy')

# Subplot 1: Training Loss
axes[0, 0].plot(baseline_train['steps'], baseline_train['loss'], 
                'b-', linewidth=2, label='Baseline')
axes[0, 0].plot(dynamic_train['steps'], dynamic_train['loss'], 
                'orange', linestyle='--', linewidth=2, label='Dynamic')
axes[0, 0].set_xlabel('Training Steps')
axes[0, 0].set_ylabel('Training Loss')
axes[0, 0].set_title('Training Loss Over Time')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# ... repeat for other subplots ...

plt.tight_layout()
plt.savefig('figure2_training_curves.png', dpi=300, bbox_inches='tight')
```

---

## Figure Checklist

Before finalizing each figure:

- [ ] All text is readable (minimum 10pt font)
- [ ] Colors match the defined palette
- [ ] Axes are labeled with units
- [ ] Legend is present and clear
- [ ] Resolution is high enough for publication (300 DPI minimum)
- [ ] Figure caption explains what to observe
- [ ] Consistent style across all figures
- [ ] Color-blind friendly (test with Coblis simulator)
- [ ] File formats: PNG for raster, PDF/SVG for vector
