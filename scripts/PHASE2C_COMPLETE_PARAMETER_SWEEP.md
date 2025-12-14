# Phase 2C: Complete Parameter Sweep Analysis
## Comprehensive Evaluation of All Original Suggested Parameters

---

## Summary: Parameter Coverage

### ✅ Parameters Now Being Tested (Phase 2C vs Phase 2B)

| Parameter | Original Ranges | Phase 2B Tested | Phase 2C (NEW) | Status |
|-----------|-----------------|-----------------|----------------|--------|
| **n_colonies** | {2,3,4,5} | ✅ {2,3,4,5} | ✅ {2,3,4,5} | COMPLETE |
| **local_patience** | {3,5,7,10} | ❌ {5,10,15} | ✅ {3,5,7,10} | **FIXED** |
| **sharing_frequency** | {5,10,15,20} | ❌ NOT TESTED | ✅ {5,10,15,20} | **NEW** |
| **sharing_ratio** | {0.05,0.1,0.15,0.2} | ❌ NOT TESTED | ✅ {0.05,0.1,0.15,0.2} | **NEW** |
| **initial_mu** | {1e-4,1e-3,1e-2,1e-1} | ❌ {1e-4,1e-2,1e-1} | ✅ {1e-4,1e-3,1e-2,1e-1} | **FIXED** |
| **lm_max_iterations** | {10,25,50,100} | ❌ NOT TESTED | ✅ {10,25,50,100} | **NEW** |

---

## Phase 2C Experimental Design

### Configuration Space
- **Total Factorial Combinations**: 4 × 4 × 4 × 4 × 4 × 4 = **4,096 configurations**
- **Sampling Strategy**: Random sampling for computational efficiency
- **Current Plan**: 100 random configs × 2 runs each = 200 total evaluations
- **Dataset**: Heart disease (920 samples, 35 features, binary classification)
- **Train/Test Split**: 70/30 stratified split
- **Evaluation Metric**: Accuracy

### Experiment Parameters (FIXED across all configs)
```
Shared ACOR Parameters:
  - n_ants: 2
  - n_samples: 230 per colony
  - q: 0.01
  - xi: 0.95
  - max_iter: 100 (global)
  - patience: 15 (global stagnation threshold)

FNN Architecture:
  - Input dimension: 35
  - Hidden dimension: 6 (ReLU)
  - Output dimension: 1 (Sigmoid)
```

### Tunable Parameters (SWEEP SPACE)
```
Hybrid-Specific Parameters:
  1. n_colonies ∈ {2, 3, 4, 5}           [4 values]
  2. local_patience ∈ {3, 5, 7, 10}      [4 values] ← LM trigger threshold
  3. sharing_frequency ∈ {5, 10, 15, 20} [4 values] ← How often colonies share
  4. sharing_ratio ∈ {0.05, 0.1, 0.15, 0.2} [4 values] ← What % of solutions share
  5. initial_mu ∈ {1e-4, 1e-3, 1e-2, 1e-1} [4 values] ← LM Jacobian damping
  6. lm_max_iterations ∈ {10, 25, 50, 100} [4 values] ← LM refinement budget
```

---

## Key Improvements Over Phase 2B

### Phase 2B Limitations (Identified):
1. ❌ **Incomplete local_patience**: Tested {5,10,15} instead of original {3,5,7,10}
2. ❌ **Missing sharing_frequency**: Never tested {5,10,15,20}
3. ❌ **Missing sharing_ratio**: Never tested {0.05,0.1,0.15,0.2}
4. ❌ **Incomplete initial_mu**: Missing 1e-3 value
5. ❌ **Missing lm_max_iterations**: Never tested {10,25,50,100}
6. ✅ **LM was disabled**: All tests were pure ACOR (local_patience=100)

### Phase 2C Fixes:
1. ✅ **Complete local_patience range**: {3,5,7,10}
2. ✅ **Add sharing_frequency**: {5,10,15,20}
3. ✅ **Add sharing_ratio**: {0.05,0.1,0.15,0.2}
4. ✅ **Complete initial_mu**: {1e-4,1e-3,1e-2,1e-1}
5. ✅ **Add lm_max_iterations**: {10,25,50,100}
6. ✅ **LM Enabled**: Full hybrid ACOR-LM evaluation

---

## Expected Insights from Phase 2C

### Questions to Answer:

1. **sharing_frequency Impact**
   - Does more frequent sharing help or hurt?
   - Optimal balance between exploration and exploitation?

2. **sharing_ratio Impact**
   - How much solution exchange is needed?
   - Diminishing returns with higher ratios?

3. **lm_max_iterations Impact**
   - Is 10 iterations enough for LM refinement?
   - Do more iterations (50, 100) improve convergence?

4. **local_patience Refined**
   - Should we trigger LM frequently (lp=3) or rarely (lp=10)?
   - Optimal balance with sharing parameters?

5. **Parameter Interactions**
   - Do certain parameter combinations synergize?
   - Are there dominant parameters vs interactions?

---

## Comparison with Phase 2B Results

### Phase 2B Key Finding:
- **n_colonies=5 achieved 67.77%** (best pure ACOR)
- **local_patience and initial_mu had ZERO effect** (LM disabled)
- **Conclusion**: n_colonies dominates when LM is off

### Phase 2C Hypothesis:
- With LM enabled, **local_patience and initial_mu should show significant effect**
- **sharing_frequency and sharing_ratio** may be critical for colony coordination
- **lm_max_iterations** likely affects convergence quality vs computation trade-off
- Overall accuracy should be **79-82%+** with proper LM configuration (vs 67.77% without LM)

---

## Execution Plan

### Stage 1: Initial Sweep (Current)
- **Configs**: 100 random samples from 4,096
- **Runs**: 2 per config
- **Time**: ~15-20 minutes
- **Purpose**: Quick sensitivity analysis

### Stage 2: Parameter Refinement (If needed)
- **Focus**: Top 5-10 best configurations from Stage 1
- **Runs**: 5-10 per config with full CV
- **Time**: ~30-60 minutes
- **Purpose**: Statistical validation

### Stage 3: Full Grid Search (Optional)
- **Coverage**: All 4,096 configs (if computational budget allows)
- **Runs**: 1-2 per config
- **Time**: ~2-6 hours
- **Purpose**: Complete parameter space characterization

---

## Success Criteria

✅ **Phase 2C will be successful if:**
1. All 6 parameters are tested with complete ranges
2. At least one configuration achieves ≥80% accuracy
3. Parameter effects are quantified (dominant vs secondary factors)
4. Improvement over Phase 2B baseline (67.77%) demonstrated
5. Clear recommendation for optimal hybrid parameters

---

## Expected Outputs

1. **`scripts/phase2c_results.csv`**
   - All evaluated configurations with their performance metrics
   - Mean accuracy ± std deviation
   - Success/error status

2. **Phase 2C Analysis Document**
   - Parameter sensitivity analysis
   - Best configuration identified
   - Comparison with Phase 2B (ACOR-only)
   - Recommendations for final thesis experiments

3. **Visualizations (if needed)**
   - Parameter impact heatmaps
   - Accuracy distribution by parameter
   - Best configurations ranking

---

## Technical Notes

### Why Phase 2C Tests Hybrid ACOR-LM (Unlike Phase 2B):
- Phase 2B disabled LM to test pure colony dynamics
- Phase 2C enables full hybrid to understand LM parameter effects
- Sharing parameters only affect multi-colony operation
- LM parameters (local_patience, initial_mu, lm_max_iterations) critical for hybrid

### Computational Efficiency:
- Heart dataset (920 samples) is smaller than breast cancer
- Train/test split (70/30) avoids k-fold overhead
- 2 runs per config balances precision vs speed
- Random sampling covers parameter space efficiently

### Reproducibility:
- Fixed random seed (42) for all experiments
- Identical FNN architecture across all runs
- Consistent objective function (cross-entropy loss)
- Same train/test split for fair comparison

---

## Status: READY FOR EXECUTION ✅

All original parameters from the comprehensive parameter list are now included in Phase 2C.
Script is prepared and ready to run the complete factorial sweep.
