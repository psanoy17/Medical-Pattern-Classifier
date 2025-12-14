# Overhead Optimization Progress — Real-time Update

**Date**: Dec 10, 2025  
**Status**: Phase 2B Parallel Profiler Running

## Problem Identified & Solution Deployed

### The Overhead Issue
- **LM Jacobian computation**: Finite-difference Jacobian = O(weights × forward passes) = expensive
- Single baseline config: ~0.2s
- Single hybrid config: ~50+ seconds (LM triggers frequently, each with slow Jacobian)

### Solutions Attempted

#### 1. JAX Autodiff (Attempted)
- **Result**: ⚠️ Incompatible with FNN model (numpy conversions clash with JAX tracing)
- **Fallback**: Reverted to finite differences
- **Status**: Abandoned

#### 2. Multiprocessing Parallelization (Deployed) ✅
- **Script**: `f_race_hybrid_parallel.py`
- **Approach**: Use Python multiprocessing Pool to evaluate multiple configs simultaneously
- **Workers**: 6 parallel processes (configurable)
- **Expected speedup**: ~6× with 6 workers on multi-core CPU
- **Status**: Currently running

## Current Run Details

| Parameter | Value |
|-----------|-------|
| Configs to evaluate | 20 (random sample) |
| CV folds | 2 |
| Runs per fold | 3 |
| Total evaluations per config | 6 (2 folds × 3 runs) |
| Parallel workers | 6 |
| Serial time per config | ~50–60 seconds |
| **Parallel time estimate** | **~100–120 seconds total** (~8–10 sec per config with 6× parallelism) |
| Expected total runtime | ~10–15 minutes |

## Why Parallelization Works Here

1. **Independent configs**: Each config evaluation is independent (can run on separate CPU cores)
2. **Multicore hardware**: Your system has >= 6 cores available
3. **Simple to implement**: No model changes needed, just wrap evaluations in a worker pool
4. **Automatic load balancing**: Python's Pool.imap_unordered distributes work dynamically

## Expected Results

When Phase 2B completes, we'll have:
- **20 hybrid configs evaluated** with 2-fold CV, 3 runs per fold
- **Results in CSV**: `scripts/f_race_hybrid_parallel_results.csv`
- **Comparison data** showing how hybrid ACOR-LM parameters affect performance
- **Parameter sensitivity analysis** across n_colonies, local_patience, initial_mu, etc.

## Next Steps (After Phase 2B Completes)

1. **Analyze results** from `f_race_hybrid_parallel_results.csv`
2. **Compare hybrid vs baseline** performance
3. **Identify high-impact parameters** (n_colonies, local_patience, initial_mu)
4. **Run final full experiments** (4-fold, 50 runs per fold) with top configurations

---

**ETA**: Phase 2B should complete in ~10–15 minutes. Monitor the background terminal for "Parallel profile complete" message.
