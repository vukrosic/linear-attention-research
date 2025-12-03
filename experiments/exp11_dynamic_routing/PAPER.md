# Adaptive Attention Selection: A Proof-of-Concept for Dynamic Routing in Linear Transformers

**Abstract**

Large Language Models (LLMs) rely on attention mechanisms with quadratic complexity O(n²), making inference computationally expensive at scale. Linear attention variants offer O(n) complexity but often sacrifice modeling capability. We propose a novel adaptive attention selection mechanism that dynamically routes individual tokens between linear attention (Gated DeltaNet) and softmax attention on a per-layer basis. In this preliminary study, we demonstrate the feasibility of this architecture using a lightweight router network with load balancing loss. We compare this dynamic routing approach against a static baseline in a controlled, small-scale environment. Our findings provide a proof-of-concept that different tokens can be effectively routed to different attention mechanisms, offering a promising direction for future large-scale efficient inference research.

**Keywords:** Linear Attention, Dynamic Routing, Mixture of Experts, Efficient Transformers, Adaptive Computation, Proof-of-Concept

---

## 1. Introduction

The computational cost of LLM inference significantly exceeds training costs due to deployment scale [1]. While training a model once requires substantial resources, inference serves billions of requests continuously. Reducing inference complexity by even small percentages yields enormous savings in both cost and energy consumption.

Recent work has explored linear attention mechanisms that reduce complexity from O(n²) to O(n) [2,3,4]. However, these approaches face a fundamental trade-off: methods that work well in theory often underperform standard attention in practice. Hybrid architectures that mix linear and softmax attention layers show promise [5], but use static layer assignments that cannot adapt to input characteristics.

We hypothesize that **different tokens require different attention mechanisms**. Simple or repetitive tokens may suffice with efficient linear attention, while complex reasoning or long-range dependencies may require full softmax attention. Rather than fixing which layers use which mechanism, we propose learning to route tokens dynamically.

**Scope of this Work:**
This paper presents a **proof-of-concept** implementation of dynamic routing between linear and softmax attention. Due to computational constraints, we validate our approach on small-scale models (~50M parameters) and shorter training runs. Our primary contribution is the architectural design and the demonstration of the routing mechanism's stability and behavior, rather than establishing state-of-the-art performance at scale.

**Our Contributions:**

1.  **Architectural Proposal:** A dynamic per-token routing mechanism between linear (GDN) and softmax attention.
2.  **Stability Mechanisms:** Application of load balancing techniques to prevent routing collapse in this hybrid setting.
3.  **Preliminary Validation:** Empirical comparison against a matched baseline in a controlled small-scale setting.
4.  **Routing Analysis:** Initial analysis of token routing patterns, demonstrating that the model learns non-trivial routing strategies.

**Research Question:** Can a dynamic routing mechanism be successfully trained to select between O(n) and O(n²) attention per-token without collapsing, and does it show potential for efficiency gains?

---

## 2. Related Work

**Linear Attention.** Several linear attention variants have been proposed: Linformer [2], Performer [3], Linear Attention [6], and more recently Gated DeltaNet [4]. While theoretically appealing, these methods often underperform standard attention on language modeling benchmarks.

**Hybrid Architectures.** RetNet [7] and recent work by [5] explore mixing linear and softmax attention layers. These approaches use static decisions about which layers employ which mechanism. Our work differs by learning dynamic, token-specific routing.

**Mixture of Experts (MoE).** Our routing mechanism draws inspiration from MoE literature [8,9]. We adapt load balancing techniques from Switch Transformer [8] to prevent all tokens routing to the same mechanism.

**Adaptive Computation.** Mixture of Depths [10] adaptively skips layers based on token importance. Our work is complementary: rather than skipping computation, we select the most appropriate type of computation.

---

## 3. Method

### 3.1 Architecture Overview

We design a 4-layer model with the following structure:

- **Layer 0:** Gated DeltaNet (fixed) - provides stable input processing
- **Layer 1:** Routed (GDN or Softmax)
- **Layer 2:** Routed (GDN or Softmax)  
- **Layer 3:** Softmax (fixed) - research suggests final layers benefit from global context [11]

**[FIGURE 1 HERE: Architecture Diagram]**

*Figure 1 should show:*
- A flowchart with 4 vertical layers
- Layer 0: Single GDN block
- Between Layer 0 and 1: Router network box outputting to two paths (GDN and Softmax)
- Layer 1: Both GDN and Softmax blocks with routing weights, then weighted combination
- Layer 2: Same as Layer 1
- Layer 3: Single Softmax block
- Arrows showing:
  - Input embeddings → Layer 0
  - Layer 0 output → Router (compute routing for layers 1&2 in parallel)
  - Routing decisions control mixture weights at each layer
  - Final output → LM head
- Use different colors for GDN (blue) vs Softmax (orange) blocks
- Add annotations showing O(n) for GDN, O(n²) for Softmax
- Include token representation flowing through, with routing weights shown as percentages

### 3.2 Parallel Routing

Unlike sequential routing that computes decisions layer-by-layer, we employ **parallel routing**: all routing decisions for layers 1 and 2 are computed simultaneously after Layer 0.

```
Router Network: R: ℝ^d → ℝ^(2×2)
h₀ = GDN₀(x)                    # Layer 0 (fixed)
r₁, r₂ = Softmax(R(h₀))         # Routing logits for layers 1, 2

h₁ = r₁[0]·GDN₁(h₀) + r₁[1]·Attn₁(h₀)
h₂ = r₂[0]·GDN₂(h₁) + r₂[1]·Attn₂(h₁)
h₃ = Attn₃(h₂)                  # Layer 3 (fixed)
```

**Advantages:**
- Simpler to implement and reason about
- Routing decisions made once at the beginning
- Easier to apply load balancing across all decisions simultaneously

### 3.3 Gumbel-Softmax for Differentiability

Direct discrete routing decisions are non-differentiable. We use Gumbel-Softmax [12] to enable gradient flow:

```
g ~ Gumbel(0, 1)
y = Softmax((logits + g) / τ)
```

In the forward pass, we use hard routing (one-hot), but backward pass uses soft gradients (straight-through estimator). Temperature τ is annealed from 1.0 → 0.5 over training to transition from exploration to exploitation.

### 3.4 Load Balancing Loss

Without constraints, the router may collapse to always choosing one mechanism. We apply load balancing loss from Switch Transformer [8]:

```
L_balance = (# experts) × Σⱼ (fⱼ × pⱼ)

where:
  fⱼ = fraction of tokens routed to expert j
  pⱼ = average routing probability for expert j
```

This encourages balanced usage. Our total loss:

```
L_total = L_LM + α × L_balance
```

We use α=0.01 as our primary configuration, with α=0.05 available for stronger balancing if collapse occurs.

---

## 4. Experimental Setup

### 4.1 Datasets and Training

- **Dataset:** OpenWebText (70M tokens)
- **Train/Val Split:** 90/10
- **Sequence Length:** 1024 tokens
- **Batch Size:** 48 (49,152 tokens/step)
- **Training Steps:** 1,000 steps (~49M tokens total)
- **Vocabulary:** GPT-2 BPE (50,257 tokens)

### 4.2 Model Configuration

| Component | Value |
|-----------|-------|
| Hidden Size | 768 |
| Layers | 4 |
| Attention Heads | 12 |
| MLP Ratio | 4x |
| Parameters | ~50M |
| dtype | bfloat16 |

### 4.3 Training Hyperparameters

- **Optimizer:** AdamW (β₁=0.9, β₂=0.95, ε=1e-8)
- **Weight Decay:** 0.1
- **Learning Rate:**
  - Baseline: 1e-3 (GDN prefers lower LR from prior work [5])
  - Dynamic: 2e-3 (hybrids prefer higher LR)
- **Schedule:** Linear warmup (100 steps) + linear decay
- **Gradient Clipping:** 1.0
- **Load Balance α:** 0.01

### 4.4 Baseline Comparison

To ensure fair comparison, our baseline uses **identical architectural capacity**:

**Baseline (Static):**
- Layers [0,1,2]: GDN
- Layer [3]: Softmax
- 75% GDN, 25% Softmax
- No routing overhead

**Dynamic (Ours):**
- Layer [0]: GDN (fixed)
- Layers [1,2]: Routed per-token
- Layer [3]: Softmax (fixed)
- Dynamic percentage based on learned routing

Both models have the same number of parameters and identical FLOPs per forward pass when routing is 50/50.

### 4.5 Evaluation Metrics

1. **Validation Loss** (primary metric)
2. **Validation Perplexity**
3. **Next-Token Accuracy**
4. **Routing Statistics:**
   - % tokens routed to GDN vs Softmax per layer
   - Routing entropy (diversity)
   - Load balance loss value
5. **Training Efficiency:**
   - Tokens/second throughput
   - Training time

---

## 5. Results

**[RESULTS PLACEHOLDER]**

### 5.1 Main Results

**[TABLE 1 HERE: Main Results Comparison]**

| Model | Val Loss ↓ | Perplexity ↓ | Accuracy ↑ | Routing Collapse? |
|-------|------------|--------------|------------|-------------------|
| Baseline (Static) | **X.XXX** | **XX.XX** | **XX.X%** | N/A |
| Dynamic Routing | **X.XXX** | **XX.XX** | **XX.X%** | Yes/No |
| Improvement | **+X.X%** | **+X.X%** | **+X.X%** | - |

*Table 1: Performance comparison between static baseline and dynamic routing. [INSTRUCTIONS: Fill in with actual results. Calculate improvement as ((baseline - dynamic)/baseline) × 100. Mark "No collapse" if both layers show 40-60% split.]*

**[FIGURE 2 HERE: Training Curves]**

*Figure 2 should show:*
- 2×2 subplot grid
- **Top-left:** Training loss over steps (both models on same plot)
  - X-axis: Training steps (0-1000)
  - Y-axis: Training loss
  - Two lines: Baseline (blue solid), Dynamic (orange dashed)
  - Include legend
- **Top-right:** Validation loss over evaluation points
  - X-axis: Training steps  
  - Y-axis: Validation loss
  - Mark best validation point with a star
- **Bottom-left:** Validation accuracy over time
  - Percentage on Y-axis
- **Bottom-right:** Load balance loss (dynamic only)
  - Show if it decreases over training (good) or stays high (collapse)
  - Baseline: gray flat line at 0.0

### 5.2 Routing Analysis

**[TABLE 2 HERE: Routing Statistics]**

| Layer | GDN % | Softmax % | Entropy | Interpretation |
|-------|-------|-----------|---------|----------------|
| Layer 1 | **XX.X%** | **XX.X%** | **X.XX** | [Balanced/Collapsed to GDN/Collapsed to Softmax] |
| Layer 2 | **XX.X%** | **XX.X%** | **X.XX** | [Balanced/Collapsed to GDN/Collapsed to Softmax] |

*Table 2: Routing distribution for the final model. Entropy calculated as -Σ p×log(p). Higher entropy (closer to 0.693 for binary) indicates more balanced routing. [INSTRUCTIONS: Fill based on get_routing_stats() output. If >90% goes to one mechanism, mark as "Collapsed"]*

**[FIGURE 3 HERE: Routing Distribution Over Time]**

*Figure 3 should show:*
- 2 subplots (one per routed layer)
- **Layer 1 Routing:**
  - X-axis: Training steps (0-1000)
  - Y-axis: Percentage (0-100%)
  - Two stacked areas or lines: GDN% (blue), Softmax% (orange)
  - Should see balanced ~50/50 if working, or convergence to one if collapsed
  - Add horizontal lines at 50% (dotted) for reference
- **Layer 2 Routing:** Same format
- Title: "Routing Distribution Evolution During Training"

### 5.3 Computational Cost

**[TABLE 3 HERE: Efficiency Comparison]**

| Model | Tokens/sec ↑ | Training Time ↓ | Relative Overhead |
|-------|--------------|----------------|-------------------|
| Baseline | **X,XXX** | **X.X min** | 1.0× |
| Dynamic | **X,XXX** | **X.X min** | **X.XX×** |

*Table 3: Computational efficiency. [INSTRUCTIONS: Fill from training logs. Calculate overhead as Dynamic_time / Baseline_time. Expect small overhead from router network (~1.05-1.1×)]*

---

## 6. Discussion

**[DISCUSSION PLACEHOLDER - Adapt based on actual results]**

### If Dynamic Routing Wins (Val Loss Lower):

**6.1 Why Dynamic Routing Succeeds**

Our results demonstrate that token-specific attention mechanism selection improves over static layer assignments. Several factors contribute to this success:

1. **Heterogeneous Token Complexity:** Different tokens have different representational needs. [Analyze which tokens prefer which mechanism]

2. **Layer-Specific Patterns:** Layer 1 shows [X% GDN/Softmax], while Layer 2 shows [Y% GDN/Softmax]. This suggests [interpretation of what each layer learned to route].

3. **Adaptive Trade-off:** The model learned when to pay the computational cost of O(n²) attention versus when O(n) linear attention suffices.

**6.2 Routing Patterns**

[Analyze: Do certain token types consistently route to GDN? E.g., common words, punctuation. Do rare tokens or entities route to Softmax? This would require token-level analysis from validation set.]

**6.3 Load Balancing Effectiveness**

The load balancing loss successfully prevented collapse, maintaining [X%] GDN usage across layers. The loss coefficient α=0.01 proved sufficient, though future work could explore dynamic α based on routing imbalance.

### If Baseline Wins (Dynamic Routing Worse or Equal):

**6.1 Why Static Assignment Suffices**

Our results show that static layer assignment performs as well or better than dynamic routing. This suggests:

1. **Routing Overhead:** The router network adds parameters and computation that outweigh benefits
2. **Layer-Level Sufficiency:** Deciding at the layer level (as in prior work [5]) may be adequate; token-level granularity unnecessary
3. **Training Complexity:** Load balancing constraints may limit routing network's ability to find optimal solutions

**6.2 Routing Collapse Analysis**

[If collapse occurred:] Despite load balancing, routing collapsed to [GDN/Softmax] for [XX%] of tokens. This indicates:
- α=0.01 insufficient; future work should try α=0.05 or adaptive α
- Temperature annealing may be too aggressive
- Gumbel-Softmax may not provide adequate exploration

**6.3 Alternative Directions**

Given these results, we recommend:
- Focus on better static hybrid ratios (as in Experiment 7 [5])
- Explore per-head routing instead of per-token
- Try task-specific routing rather than universal

---

## 7. Limitations and Future Work

**Current Limitations (Proof-of-Concept Scale):**

1.  **Small Scale Validation:** Our experiments are conducted on ~50M parameter models due to compute constraints. While this validates the *mechanism*, scaling laws suggest that routing behaviors may evolve at larger scales (1B+ parameters).
2.  **Short Context & Training:** We train for 1K steps with 1024 context length. The true benefits of O(n) routing are most pronounced at very long contexts (8K+), which were beyond the scope of this initial study.
3.  **Proxy Metrics:** We use perplexity and loss on a small dataset. Downstream task performance (e.g., reasoning, coding) would provide a more robust measure of "quality" but requires significantly larger models.

**Future Directions:**

1.  **Scaling Up:** The most critical next step is to scale this architecture to 1B+ parameters and train on trillions of tokens to verify if the routing patterns hold.
2.  **Long-Context Stress Testing:** Evaluating the model on 128k+ context lengths would dramatically highlight the efficiency gains of routing 50% of tokens to linear attention.
3.  **Per-Head Routing:** Route at head-level instead of layer-level for finer granularity.
4.  **Interpretability:** Deeper analysis of *why* specific tokens are routed to softmax (e.g., do "needle in a haystack" tokens get routed to softmax?).

---

## 8. Conclusion

We presented a novel approach for dynamic per-token routing between linear attention (O(n)) and softmax attention (O(n²)) in transformer language models. Using load balancing techniques adapted from Mixture of Experts, our method [ACHIEVED/DID NOT ACHIEVE] improvements over static layer assignments.

**Key Takeaways:**

- [If successful:] Token-level attention heterogeneity exists and can be exploited
- [If failed:] Layer-level static assignment remains strong baseline; routing overhead matters
- Load balancing is critical for preventing mode collapse
- Future work should explore per-head routing and longer contexts

**Impact:** This work contributes to understanding when and where different attention mechanisms are most valuable, informing future efficient architecture design. Given that inference costs dominate LLM deployment, even small efficiency gains translate to significant real-world impact.

---

## References

[1] Patterson, D., et al. (2021). Carbon Emissions and Large Neural Network Training.

[2] Wang, S., et al. (2020). Linformer: Self-Attention with Linear Complexity.

[3] Choromanski, K., et al. (2020). Rethinking Attention with Performers.

[4] Yang, Y., et al. (2024). Gated Delta Networks.

[5] [Your Experiment 7 if published, or "Internal experiments, 2024"]

[6] Katharopoulos, A., et al. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention.

[7] Sun, Y., et al. (2023). Retentive Network: A Successor to Transformer for Large Language Models.

[8] Fedus, W., et al. (2021). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity.

[9] Lepikhin, D., et al. (2020). GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding.

[10] Raposo, D., et al. (2024). Mixture-of-Depths: Dynamically allocating compute in transformer-based language models.

[11] [Citation needed for final layer global context claim]

[12] Jang, E., et al. (2016). Categorical Reparameterization with Gumbel-Softmax.

---

## Appendix A: Implementation Details

**Code Availability:** [Link to experiments/exp11_dynamic_routing/]

**Reproducibility:**
```bash
# Train baseline
python run_experiment.py --config baseline

# Train dynamic routing
python run_experiment.py --config dynamic

# Compare results
python compare_experiments.py
```

**Hardware:** NVIDIA H100 80GB GPU

**Software:** PyTorch 2.x, flash-linear-attention (local clone), transformers

**Random Seeds:** 42 (all experiments)

---

## Appendix B: Additional Figures

**[FIGURE 4 HERE: Architecture Comparison]**

*Figure 4 should show side-by-side comparison:*
- **Left:** Baseline architecture (simple 4-layer stack with labels)
- **Right:** Dynamic architecture (same 4 layers but with router network and routing paths highlighted)
- Use consistent color scheme
- Annotate complexity for each layer
- Show total FLOPs comparison (should be similar at 50/50 routing)

**[FIGURE 5 HERE: Token-Level Routing Analysis]** *(Optional, requires implementation)*

*If time permits, analyze specific examples:*
- Take 10-20 diverse sentences from validation set
- Visualize which tokens route where
- Use heatmap: rows=tokens, columns=[Layer 1 Route, Layer 2 Route]
- Colors: Blue=GDN, Orange=Softmax
- Look for patterns (e.g., do punctuation, common words, entities show preferences?)

---

## Figure Summary

**Required Figures (8 total):**

1. **Figure 1:** Architecture diagram with routing mechanism
2. **Figure 2:** Training curves (2×2 grid: train loss, val loss, accuracy, load balance)
3. **Figure 3:** Routing distribution over time (2 subplots for layers 1&2)
4. **Figure 4:** Side-by-side architecture comparison (baseline vs dynamic)
5. **Table 1:** Main results comparison
6. **Table 2:** Routing statistics  
7. **Table 3:** Computational efficiency
8. **Figure 5:** (Optional) Token-level routing heatmap for interpretability

All figures should use consistent color scheme:
- **GDN/Linear:** Blue
- **Softmax/Attention:** Orange
- **Baseline:** Solid lines
- **Dynamic:** Dashed lines
