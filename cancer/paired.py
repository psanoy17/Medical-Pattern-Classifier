import pandas as pd
import numpy as np
from scipy import stats

# Load data
filename = 'cancer_kfold_paired_comparison.csv'
try:
    df = pd.read_csv(filename)
except:
    print("CSV file not found. Run the cross-validation script first.")
    exit()

print("="*105)
print(f"{'METRIC':<12} | {'BASELINE (Mean ± SD)':<22} | {'HYBRID (Mean ± SD)':<22} | {'DIFF':<8} | {'t-val':<8} | {'p-val':<8} | {'SIG?'}")
print("="*105)

metrics = ['Accuracy', 'Precision', 'Recall', 'F1_score']

for metric in metrics:
    # 1. Get raw data columns
    base_col = f'Baseline_{metric}'
    hyb_col = f'Hybrid_{metric}'
    diff_col = f'Difference_{metric}'
    
    # 2. Calculate Mean and SD for display
    base_mean = df[base_col].mean()
    base_sd = df[base_col].std()
    
    hyb_mean = df[hyb_col].mean()
    hyb_sd = df[hyb_col].std()
    
    # 3. Perform T-Test Stats
    differences = df[diff_col]
    n = len(differences)
    d_bar = differences.mean()
    sd_diff = differences.std()
    
    t_val = d_bar / (sd_diff / np.sqrt(n))
    
    # Calculate p-value (two-tailed)
    df_degrees = n - 1
    p_val = stats.t.sf(np.abs(t_val), df_degrees) * 2  
    
    # Determine significance star
    sig_label = "Yes" if p_val < 0.05 else "No"
    
    # 4. Print formatted row for Thesis Table
    print(f"{metric:<12} | "
          f"{base_mean:.4f} ± {base_sd:.4f}      | "
          f"{hyb_mean:.4f} ± {hyb_sd:.4f}      | "
          f"{d_bar:+.4f}   | "
          f"{t_val:.4f}   | "
          f"{p_val:.4f}   | "
          f"{sig_label}")

print("="*105)
print("Note: If p-val is 0.0000, report it as 'p < 0.001' in your thesis.")