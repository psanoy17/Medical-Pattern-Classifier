# Phase 2C: Full Factorial Parameter Sweep - Comprehensive Results

**Generated:** 2025-12-14 21:21:31  
**Total Configurations per Dataset:** 4,096  
**Runs per Configuration:** 2  
**Total Evaluations per Dataset:** 8,192

---

## Executive Summary

### Overall Performance Comparison

| Dataset | Successful | Failed | Mean Acc | Std Acc | Min Acc | Max Acc | Range |
|---------|-----------|--------|----------|---------|---------|---------|-------|
| Heart | 4096 | 0 | 0.7921 | 0.0193 | 0.7572 | 0.8351 | 0.0779 |
| Cancer | 4096 | 0 | 0.9664 | 0.0069 | 0.9452 | 0.9786 | 0.0333 |
| Diabetes | 4096 | 0 | 0.7619 | 0.0081 | 0.7468 | 0.7857 | 0.0390 |

---

## Heart Dataset Results

### Best Configuration (Accuracy: 0.8351 � 0.0091)

```
n_colonies:         4
local_patience:     10
sharing_frequency:  20
sharing_ratio:      0.05
initial_mu:         1e-02
lm_max_iterations:  10
```

### Top 10 Configurations

| Rank | Accuracy | n_col | lp | sf | sr | mu | lm_iter |
|------|----------|-------|----|----|----|----|--------|
| 1 | 0.8351�0.0091 | 4 | 10 | 20 | 0.05 | 1e-02 | 10 |
| 2 | 0.8351�0.0091 | 4 | 10 | 15 | 0.15 | 1e-01 | 10 |
| 3 | 0.8351�0.0091 | 4 | 7 | 15 | 0.10 | 1e-03 | 10 |
| 4 | 0.8351�0.0091 | 4 | 7 | 10 | 0.20 | 1e-01 | 10 |
| 5 | 0.8351�0.0091 | 4 | 10 | 15 | 0.05 | 1e-02 | 10 |
| 6 | 0.8351�0.0091 | 4 | 7 | 15 | 0.20 | 1e-03 | 10 |
| 7 | 0.8351�0.0091 | 4 | 7 | 10 | 0.20 | 1e-03 | 10 |
| 8 | 0.8351�0.0091 | 4 | 10 | 15 | 0.20 | 1e-04 | 10 |
| 9 | 0.8351�0.0091 | 4 | 5 | 10 | 0.05 | 1e-03 | 10 |
| 10 | 0.8351�0.0091 | 4 | 5 | 10 | 0.10 | 1e-02 | 10 |

### Parameter Sensitivity Analysis

#### n_colonies

| Value | Mean Acc | Std Acc | Count |
|-------|----------|---------|-------|
| 5 | 0.8049 | 0.0074 | 1024 |
| 4 | 0.8025 | 0.0119 | 1024 |
| 3 | 0.7841 | 0.0214 | 1024 |
| 2 | 0.7770 | 0.0162 | 1024 |

#### local_patience

| Value | Mean Acc | Std Acc | Count |
|-------|----------|---------|-------|
| 7 | 0.7937 | 0.0198 | 1024 |
| 10 | 0.7925 | 0.0199 | 1024 |
| 5 | 0.7912 | 0.0186 | 1024 |
| 3 | 0.7911 | 0.0186 | 1024 |

#### sharing_frequency

| Value | Mean Acc | Std Acc | Count |
|-------|----------|---------|-------|
| 15 | 0.7945 | 0.0196 | 1024 |
| 10 | 0.7937 | 0.0198 | 1024 |
| 20 | 0.7911 | 0.0186 | 1024 |
| 5 | 0.7893 | 0.0185 | 1024 |

#### sharing_ratio

| Value | Mean Acc | Std Acc | Count |
|-------|----------|---------|-------|
| 0.05 | 0.7921 | 0.0193 | 1024 |
| 0.1 | 0.7921 | 0.0193 | 1024 |
| 0.15 | 0.7921 | 0.0193 | 1024 |
| 0.2 | 0.7921 | 0.0193 | 1024 |

#### initial_mu

| Value | Mean Acc | Std Acc | Count |
|-------|----------|---------|-------|
| 0.0001 | 0.7926 | 0.0192 | 1024 |
| 0.01 | 0.7922 | 0.0188 | 1024 |
| 0.1 | 0.7919 | 0.0198 | 1024 |
| 0.001 | 0.7918 | 0.0192 | 1024 |

#### lm_max_iterations

| Value | Mean Acc | Std Acc | Count |
|-------|----------|---------|-------|
| 10 | 0.8068 | 0.0205 | 1024 |
| 25 | 0.7915 | 0.0162 | 1024 |
| 50 | 0.7851 | 0.0156 | 1024 |
| 100 | 0.7851 | 0.0156 | 1024 |

### Worst Configuration (Accuracy: 0.7572 � 0.0145)

```
n_colonies:         2
local_patience:     5
sharing_frequency:  5
sharing_ratio:      0.15
initial_mu:         1e-03
lm_max_iterations:  10
```

---

## Cancer Dataset Results

### Best Configuration (Accuracy: 0.9786 � 0.0024)

```
n_colonies:         3
local_patience:     3
sharing_frequency:  10
sharing_ratio:      0.15
initial_mu:         1e-01
lm_max_iterations:  10
```

### Top 10 Configurations

| Rank | Accuracy | n_col | lp | sf | sr | mu | lm_iter |
|------|----------|-------|----|----|----|----|--------|
| 1 | 0.9786�0.0024 | 3 | 3 | 10 | 0.15 | 1e-01 | 10 |
| 2 | 0.9786�0.0024 | 3 | 7 | 20 | 0.10 | 1e-01 | 10 |
| 3 | 0.9786�0.0024 | 3 | 5 | 15 | 0.20 | 1e-01 | 10 |
| 4 | 0.9786�0.0024 | 3 | 3 | 10 | 0.10 | 1e-01 | 10 |
| 5 | 0.9786�0.0024 | 3 | 7 | 20 | 0.20 | 1e-01 | 10 |
| 6 | 0.9786�0.0024 | 3 | 5 | 15 | 0.15 | 1e-01 | 10 |
| 7 | 0.9786�0.0024 | 3 | 3 | 10 | 0.20 | 1e-01 | 10 |
| 8 | 0.9786�0.0024 | 3 | 3 | 10 | 0.05 | 1e-01 | 10 |
| 9 | 0.9786�0.0024 | 3 | 7 | 20 | 0.15 | 1e-01 | 10 |
| 10 | 0.9786�0.0024 | 3 | 5 | 15 | 0.05 | 1e-01 | 10 |

### Parameter Sensitivity Analysis

#### n_colonies

| Value | Mean Acc | Std Acc | Count |
|-------|----------|---------|-------|
| 4 | 0.9707 | 0.0036 | 1024 |
| 5 | 0.9703 | 0.0025 | 1024 |
| 3 | 0.9655 | 0.0077 | 1024 |
| 2 | 0.9592 | 0.0053 | 1024 |

#### local_patience

| Value | Mean Acc | Std Acc | Count |
|-------|----------|---------|-------|
| 3 | 0.9679 | 0.0065 | 1024 |
| 5 | 0.9667 | 0.0065 | 1024 |
| 7 | 0.9662 | 0.0074 | 1024 |
| 10 | 0.9648 | 0.0070 | 1024 |

#### sharing_frequency

| Value | Mean Acc | Std Acc | Count |
|-------|----------|---------|-------|
| 20 | 0.9679 | 0.0065 | 1024 |
| 15 | 0.9674 | 0.0075 | 1024 |
| 10 | 0.9662 | 0.0074 | 1024 |
| 5 | 0.9641 | 0.0056 | 1024 |

#### sharing_ratio

| Value | Mean Acc | Std Acc | Count |
|-------|----------|---------|-------|
| 0.05 | 0.9664 | 0.0069 | 1024 |
| 0.1 | 0.9664 | 0.0069 | 1024 |
| 0.15 | 0.9664 | 0.0069 | 1024 |
| 0.2 | 0.9664 | 0.0069 | 1024 |

#### initial_mu

| Value | Mean Acc | Std Acc | Count |
|-------|----------|---------|-------|
| 0.01 | 0.9673 | 0.0070 | 1024 |
| 0.0001 | 0.9665 | 0.0072 | 1024 |
| 0.001 | 0.9663 | 0.0072 | 1024 |
| 0.1 | 0.9656 | 0.0061 | 1024 |

#### lm_max_iterations

| Value | Mean Acc | Std Acc | Count |
|-------|----------|---------|-------|
| 25 | 0.9684 | 0.0051 | 1024 |
| 100 | 0.9668 | 0.0053 | 1024 |
| 50 | 0.9667 | 0.0053 | 1024 |
| 10 | 0.9638 | 0.0100 | 1024 |

### Worst Configuration (Accuracy: 0.9452 � 0.0167)

```
n_colonies:         2
local_patience:     10
sharing_frequency:  15
sharing_ratio:      0.10
initial_mu:         1e-04
lm_max_iterations:  10
```

---

## Diabetes Dataset Results

### Best Configuration (Accuracy: 0.7857 � 0.0108)

```
n_colonies:         5
local_patience:     3
sharing_frequency:  5
sharing_ratio:      0.15
initial_mu:         1e-03
lm_max_iterations:  10
```

### Top 10 Configurations

| Rank | Accuracy | n_col | lp | sf | sr | mu | lm_iter |
|------|----------|-------|----|----|----|----|--------|
| 1 | 0.7857�0.0108 | 5 | 3 | 5 | 0.15 | 1e-03 | 10 |
| 2 | 0.7857�0.0108 | 5 | 10 | 20 | 0.10 | 1e-03 | 10 |
| 3 | 0.7857�0.0108 | 5 | 10 | 20 | 0.15 | 1e-03 | 10 |
| 4 | 0.7857�0.0108 | 5 | 7 | 10 | 0.15 | 1e-03 | 10 |
| 5 | 0.7857�0.0108 | 5 | 10 | 15 | 0.20 | 1e-03 | 10 |
| 6 | 0.7857�0.0108 | 5 | 10 | 20 | 0.05 | 1e-03 | 10 |
| 7 | 0.7857�0.0108 | 5 | 3 | 5 | 0.05 | 1e-03 | 10 |
| 8 | 0.7857�0.0108 | 5 | 7 | 10 | 0.10 | 1e-03 | 10 |
| 9 | 0.7857�0.0108 | 5 | 10 | 15 | 0.10 | 1e-03 | 10 |
| 10 | 0.7857�0.0108 | 5 | 5 | 10 | 0.10 | 1e-03 | 10 |

### Parameter Sensitivity Analysis

#### n_colonies

| Value | Mean Acc | Std Acc | Count |
|-------|----------|---------|-------|
| 5 | 0.7684 | 0.0048 | 1024 |
| 4 | 0.7605 | 0.0107 | 1024 |
| 2 | 0.7599 | 0.0076 | 1024 |
| 3 | 0.7588 | 0.0034 | 1024 |

#### local_patience

| Value | Mean Acc | Std Acc | Count |
|-------|----------|---------|-------|
| 10 | 0.7632 | 0.0082 | 1024 |
| 7 | 0.7619 | 0.0081 | 1024 |
| 5 | 0.7618 | 0.0081 | 1024 |
| 3 | 0.7607 | 0.0079 | 1024 |

#### sharing_frequency

| Value | Mean Acc | Std Acc | Count |
|-------|----------|---------|-------|
| 5 | 0.7641 | 0.0080 | 1024 |
| 10 | 0.7619 | 0.0081 | 1024 |
| 15 | 0.7609 | 0.0080 | 1024 |
| 20 | 0.7607 | 0.0079 | 1024 |

#### sharing_ratio

| Value | Mean Acc | Std Acc | Count |
|-------|----------|---------|-------|
| 0.05 | 0.7619 | 0.0081 | 1024 |
| 0.1 | 0.7619 | 0.0081 | 1024 |
| 0.15 | 0.7619 | 0.0081 | 1024 |
| 0.2 | 0.7619 | 0.0081 | 1024 |

#### initial_mu

| Value | Mean Acc | Std Acc | Count |
|-------|----------|---------|-------|
| 0.1 | 0.7643 | 0.0084 | 1024 |
| 0.01 | 0.7626 | 0.0060 | 1024 |
| 0.001 | 0.7608 | 0.0090 | 1024 |
| 0.0001 | 0.7599 | 0.0081 | 1024 |

#### lm_max_iterations

| Value | Mean Acc | Std Acc | Count |
|-------|----------|---------|-------|
| 10 | 0.7624 | 0.0085 | 1024 |
| 25 | 0.7619 | 0.0076 | 1024 |
| 50 | 0.7616 | 0.0082 | 1024 |
| 100 | 0.7616 | 0.0082 | 1024 |

### Worst Configuration (Accuracy: 0.7468 � 0.0152)

```
n_colonies:         4
local_patience:     10
sharing_frequency:  10
sharing_ratio:      0.05
initial_mu:         1e-03
lm_max_iterations:  25
```

---

## Cross-Dataset Insights

### Best Configurations Summary

| Dataset | Accuracy | n_col | lp | sf | sr | mu | lm_iter |
|---------|----------|-------|----|----|----|----|--------|
| Heart | 0.8351 | 4 | 10 | 20 | 0.05 | 1e-02 | 10 |
| Cancer | 0.9786 | 3 | 3 | 10 | 0.15 | 1e-01 | 10 |
| Diabetes | 0.7857 | 5 | 3 | 5 | 0.15 | 1e-03 | 10 |

## Key Findings

### Heart

- **Performance Range:** 0.0779 (0.7572 to 0.8351)
- **Success Rate:** 4096/4096 (100.0%)
- **Best Parameter Values (by average accuracy):**
  - `n_colonies`: 5 (avg acc: 0.8049)
  - `local_patience`: 7 (avg acc: 0.7937)
  - `sharing_frequency`: 15 (avg acc: 0.7945)
  - `sharing_ratio`: 0.05 (avg acc: 0.7921)
  - `initial_mu`: 0.0001 (avg acc: 0.7926)
  - `lm_max_iterations`: 10 (avg acc: 0.8068)

### Cancer

- **Performance Range:** 0.0333 (0.9452 to 0.9786)
- **Success Rate:** 4096/4096 (100.0%)
- **Best Parameter Values (by average accuracy):**
  - `n_colonies`: 4 (avg acc: 0.9707)
  - `local_patience`: 3 (avg acc: 0.9679)
  - `sharing_frequency`: 20 (avg acc: 0.9679)
  - `sharing_ratio`: 0.05 (avg acc: 0.9664)
  - `initial_mu`: 0.01 (avg acc: 0.9673)
  - `lm_max_iterations`: 25 (avg acc: 0.9684)

### Diabetes

- **Performance Range:** 0.0390 (0.7468 to 0.7857)
- **Success Rate:** 4096/4096 (100.0%)
- **Best Parameter Values (by average accuracy):**
  - `n_colonies`: 5 (avg acc: 0.7684)
  - `local_patience`: 10 (avg acc: 0.7632)
  - `sharing_frequency`: 5 (avg acc: 0.7641)
  - `sharing_ratio`: 0.05 (avg acc: 0.7619)
  - `initial_mu`: 0.1 (avg acc: 0.7643)
  - `lm_max_iterations`: 10 (avg acc: 0.7624)


---

## Methodology

- **Parameter Space:** 4 � 4 � 4 � 4 � 4 � 4 = 4,096 configurations
- **Sampling Strategy:** Complete enumeration (all 4,096 configs tested)
- **Evaluation:** Train/test split (70/30) with 2 independent runs per config
- **Fixed Parameters:**
  - `n_ants`: 2
  - `n_samples`: 230
  - `q`: 0.01
  - `xi`: 0.95
  - `max_iter`: 100
  - `patience`: 15
  - `hidden_dim`: 6

