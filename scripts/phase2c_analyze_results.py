"""
Phase 2C: Comprehensive Analysis of Full 4096-Config Sweeps
Analyzes results from heart, cancer, and diabetes datasets
"""

import pandas as pd
import numpy as np
from datetime import datetime

def analyze_dataset(df, dataset_name):
    """Analyze results for a single dataset"""
    
    # Filter successful runs
    successful = df[df['status'] == 'SUCCESS'].copy()
    
    if len(successful) == 0:
        return None
    
    # Basic statistics
    accuracies = successful['mean_accuracy'].values
    
    # Best configuration
    best_idx = successful['mean_accuracy'].idxmax()
    best_config = successful.loc[best_idx]
    
    # Worst configuration
    worst_idx = successful['mean_accuracy'].idxmin()
    worst_config = successful.loc[worst_idx]
    
    # Top 10 configurations
    top10 = successful.nlargest(10, 'mean_accuracy')
    
    # Parameter impact analysis
    param_cols = ['n_colonies', 'local_patience', 'sharing_frequency', 
                  'sharing_ratio', 'initial_mu', 'lm_max_iterations']
    
    param_stats = {}
    for param in param_cols:
        grouped = successful.groupby(param)['mean_accuracy'].agg(['mean', 'std', 'count'])
        param_stats[param] = grouped
    
    return {
        'dataset': dataset_name,
        'total_configs': len(df),
        'successful': len(successful),
        'failed': len(df) - len(successful),
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'min_accuracy': np.min(accuracies),
        'max_accuracy': np.max(accuracies),
        'accuracy_range': np.max(accuracies) - np.min(accuracies),
        'best_config': best_config,
        'worst_config': worst_config,
        'top10': top10,
        'param_stats': param_stats
    }


def generate_report(results_dict):
    """Generate comprehensive markdown report"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Phase 2C: Full Factorial Parameter Sweep - Comprehensive Results

**Generated:** {timestamp}  
**Total Configurations per Dataset:** 4,096  
**Runs per Configuration:** 2  
**Total Evaluations per Dataset:** 8,192

---

## Executive Summary

"""
    
    # Summary table
    report += "### Overall Performance Comparison\n\n"
    report += "| Dataset | Successful | Failed | Mean Acc | Std Acc | Min Acc | Max Acc | Range |\n"
    report += "|---------|-----------|--------|----------|---------|---------|---------|-------|\n"
    
    for dataset in ['Heart', 'Cancer', 'Diabetes']:
        if dataset in results_dict and results_dict[dataset]:
            r = results_dict[dataset]
            report += f"| {dataset} | {r['successful']} | {r['failed']} | "
            report += f"{r['mean_accuracy']:.4f} | {r['std_accuracy']:.4f} | "
            report += f"{r['min_accuracy']:.4f} | {r['max_accuracy']:.4f} | "
            report += f"{r['accuracy_range']:.4f} |\n"
    
    report += "\n---\n\n"
    
    # Detailed results per dataset
    for dataset in ['Heart', 'Cancer', 'Diabetes']:
        if dataset not in results_dict or not results_dict[dataset]:
            continue
            
        r = results_dict[dataset]
        report += f"## {dataset} Dataset Results\n\n"
        
        # Best configuration
        report += f"### Best Configuration (Accuracy: {r['best_config']['mean_accuracy']:.4f} ± {r['best_config']['std_accuracy']:.4f})\n\n"
        report += "```\n"
        report += f"n_colonies:         {r['best_config']['n_colonies']}\n"
        report += f"local_patience:     {r['best_config']['local_patience']}\n"
        report += f"sharing_frequency:  {r['best_config']['sharing_frequency']}\n"
        report += f"sharing_ratio:      {r['best_config']['sharing_ratio']:.2f}\n"
        report += f"initial_mu:         {r['best_config']['initial_mu']:.0e}\n"
        report += f"lm_max_iterations:  {r['best_config']['lm_max_iterations']}\n"
        report += "```\n\n"
        
        # Top 10 configurations
        report += "### Top 10 Configurations\n\n"
        report += "| Rank | Accuracy | n_col | lp | sf | sr | mu | lm_iter |\n"
        report += "|------|----------|-------|----|----|----|----|--------|\n"
        
        for idx, (_, row) in enumerate(r['top10'].iterrows(), 1):
            report += f"| {idx} | {row['mean_accuracy']:.4f}±{row['std_accuracy']:.4f} | "
            report += f"{row['n_colonies']} | {row['local_patience']} | "
            report += f"{row['sharing_frequency']} | {row['sharing_ratio']:.2f} | "
            report += f"{row['initial_mu']:.0e} | {row['lm_max_iterations']} |\n"
        
        report += "\n"
        
        # Parameter sensitivity analysis
        report += "### Parameter Sensitivity Analysis\n\n"
        
        for param in ['n_colonies', 'local_patience', 'sharing_frequency', 
                      'sharing_ratio', 'initial_mu', 'lm_max_iterations']:
            report += f"#### {param}\n\n"
            report += "| Value | Mean Acc | Std Acc | Count |\n"
            report += "|-------|----------|---------|-------|\n"
            
            stats = r['param_stats'][param].sort_values('mean', ascending=False)
            for value, row in stats.iterrows():
                report += f"| {value} | {row['mean']:.4f} | {row['std']:.4f} | {int(row['count'])} |\n"
            
            report += "\n"
        
        # Worst configuration
        report += f"### Worst Configuration (Accuracy: {r['worst_config']['mean_accuracy']:.4f} ± {r['worst_config']['std_accuracy']:.4f})\n\n"
        report += "```\n"
        report += f"n_colonies:         {r['worst_config']['n_colonies']}\n"
        report += f"local_patience:     {r['worst_config']['local_patience']}\n"
        report += f"sharing_frequency:  {r['worst_config']['sharing_frequency']}\n"
        report += f"sharing_ratio:      {r['worst_config']['sharing_ratio']:.2f}\n"
        report += f"initial_mu:         {r['worst_config']['initial_mu']:.0e}\n"
        report += f"lm_max_iterations:  {r['worst_config']['lm_max_iterations']}\n"
        report += "```\n\n"
        
        report += "---\n\n"
    
    # Cross-dataset insights
    report += "## Cross-Dataset Insights\n\n"
    
    report += "### Best Configurations Summary\n\n"
    report += "| Dataset | Accuracy | n_col | lp | sf | sr | mu | lm_iter |\n"
    report += "|---------|----------|-------|----|----|----|----|--------|\n"
    
    for dataset in ['Heart', 'Cancer', 'Diabetes']:
        if dataset in results_dict and results_dict[dataset]:
            r = results_dict[dataset]
            bc = r['best_config']
            report += f"| {dataset} | {bc['mean_accuracy']:.4f} | "
            report += f"{bc['n_colonies']} | {bc['local_patience']} | "
            report += f"{bc['sharing_frequency']} | {bc['sharing_ratio']:.2f} | "
            report += f"{bc['initial_mu']:.0e} | {bc['lm_max_iterations']} |\n"
    
    report += "\n"
    
    # Key findings
    report += "## Key Findings\n\n"
    
    for dataset in ['Heart', 'Cancer', 'Diabetes']:
        if dataset in results_dict and results_dict[dataset]:
            r = results_dict[dataset]
            report += f"### {dataset}\n\n"
            report += f"- **Performance Range:** {r['accuracy_range']:.4f} "
            report += f"({r['min_accuracy']:.4f} to {r['max_accuracy']:.4f})\n"
            report += f"- **Success Rate:** {r['successful']}/{r['total_configs']} "
            report += f"({100*r['successful']/r['total_configs']:.1f}%)\n"
            
            # Identify best parameter values
            best_params = {}
            for param in ['n_colonies', 'local_patience', 'sharing_frequency', 
                          'sharing_ratio', 'initial_mu', 'lm_max_iterations']:
                stats = r['param_stats'][param]
                best_value = stats['mean'].idxmax()
                best_params[param] = (best_value, stats.loc[best_value, 'mean'])
            
            report += f"- **Best Parameter Values (by average accuracy):**\n"
            for param, (value, acc) in best_params.items():
                report += f"  - `{param}`: {value} (avg acc: {acc:.4f})\n"
            
            report += "\n"
    
    report += "\n---\n\n"
    report += "## Methodology\n\n"
    report += "- **Parameter Space:** 4 × 4 × 4 × 4 × 4 × 4 = 4,096 configurations\n"
    report += "- **Sampling Strategy:** Complete enumeration (all 4,096 configs tested)\n"
    report += "- **Evaluation:** Train/test split (70/30) with 2 independent runs per config\n"
    report += "- **Fixed Parameters:**\n"
    report += "  - `n_ants`: 2\n"
    report += "  - `n_samples`: 230\n"
    report += "  - `q`: 0.01\n"
    report += "  - `xi`: 0.95\n"
    report += "  - `max_iter`: 100\n"
    report += "  - `patience`: 15\n"
    report += "  - `hidden_dim`: 6\n"
    report += "\n"
    
    return report


def main():
    print("Loading results...")
    
    results_dict = {}
    
    # Load cancer results
    try:
        cancer_df = pd.read_csv('scripts/phase2c_cancer_results.csv')
        results_dict['Cancer'] = analyze_dataset(cancer_df, 'Cancer')
        print(f"✓ Cancer: {len(cancer_df)} configs")
    except Exception as e:
        print(f"✗ Cancer: {e}")
    
    # Load diabetes results
    try:
        diabetes_df = pd.read_csv('scripts/phase2c_diabetes_results.csv')
        results_dict['Diabetes'] = analyze_dataset(diabetes_df, 'Diabetes')
        print(f"✓ Diabetes: {len(diabetes_df)} configs")
    except Exception as e:
        print(f"✗ Diabetes: {e}")
    
    # Load heart results
    try:
        heart_df = pd.read_csv('scripts/phase2c_full_4096_results.csv')
        results_dict['Heart'] = analyze_dataset(heart_df, 'Heart')
        print(f"✓ Heart: {len(heart_df)} configs")
    except Exception as e:
        print(f"✗ Heart: {e}")
    
    print("\nGenerating comprehensive report...")
    report = generate_report(results_dict)
    
    # Save report
    output_path = 'PHASE2C_FULL_4096_COMPREHENSIVE_REPORT.md'
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\n✓ Report saved: {output_path}")
    print(f"  Total datasets analyzed: {len(results_dict)}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    
    for dataset in ['Heart', 'Cancer', 'Diabetes']:
        if dataset in results_dict and results_dict[dataset]:
            r = results_dict[dataset]
            print(f"\n{dataset}:")
            print(f"  Best Accuracy: {r['max_accuracy']:.4f}")
            print(f"  Mean Accuracy: {r['mean_accuracy']:.4f} ± {r['std_accuracy']:.4f}")
            print(f"  Range: {r['accuracy_range']:.4f}")


if __name__ == '__main__':
    main()
