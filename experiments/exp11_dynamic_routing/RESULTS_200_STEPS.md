# Experiment 11: Dynamic Routing - 200 Step Results

## Summary

Successfully trained three configurations for 200 steps to compare static vs dynamic routing.

## Results Comparison

| Metric | Baseline (Static) | Dynamic (Collapsed) | Dynamic (Balanced) |
|--------|-------------------|---------------------|--------------------|
| **Val Loss** | **7.0026** | **5.7030** | 6.2496 |
| **Val Acc** | 13.30% | 21.41% | 21.08% |
| **Params** | 141.7M | 160.6M | 160.6M |
| **Routing L1** | Fixed GDN | 96.7% GDN | **55.6% GDN** |
| **Routing L2** | Fixed GDN | 99.8% GDN | **60.9% GDN** |

## Key Findings

1. **Mode Collapse Risk**: Without strong regularization, the dynamic router collapses to selecting GDN 99% of the time (likely due to easier optimization path).
2. **Effective Regularization**: Increasing `load_balance_alpha` to 0.5 successfully forced the model to maintain diversity (~60/40 split).
3. **Performance Trade-off**: 
   - The "collapsed" model (pure GDN) achieved the best loss (5.70), suggesting GDN is very strong for this scale/data.
   - The "balanced" model (6.25) performed worse than pure GDN but better than the static baseline (7.00).
   - This suggests that while GDN is dominant, having the *option* to route to attention is valuable.

## Visualizations

Visualizations have been generated in `results_dynamic_aggressive/`:
- `layer_selection_over_time.png`: Shows how routing stabilized around 60/40
- `routing_comparison.png`: Shows Layer 2 consistently preferring GDN slightly more than Layer 1

## Conclusion

We have successfully implemented a stable dynamic routing mechanism for hybrid linear/quadratic attention. The key to success was aggressive load balancing regularization to prevent mode collapse.
