"""
Evaluate MORAL fairness and performance metrics from final ranking files.
"""

import argparse
import glob
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

def load_ranking_files(pattern: str = "three_classifiers_*_final_ranking.pt") -> Dict:
    """Load all ranking files matching a pattern"""
    files = glob.glob(pattern)
    print(f"Found {len(files)} ranking files:")
    
    results = {}
    for file in sorted(files):
        try:
            # Parse run info from filename
            # Pattern: three_classifiers_{dataset}_{method}_{model}_{run}_final_ranking.pt
            parts = Path(file).stem.split('_')
            if len(parts) >= 6:
                dataset = parts[2]
                method = parts[3]  # Usually "MORAL"
                model = parts[4]   # Usually "GAE" or "NCN"
                run = parts[5]     # Run number
                
                key = f"{dataset}_{method}_{model}"
                if key not in results:
                    results[key] = []
                
                # Load the data
                predictions, labels = torch.load(file)
                results[key].append((predictions.numpy(), labels.numpy()))
                
                print(f"  {file}: {predictions.shape} predictions, {labels.shape} labels")
                
        except Exception as e:
            print(f"  Error loading {file}: {e}")
    
    return results

def calculate_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict:
    """Calculate various evaluation metrics"""
    # Sort by predictions (descending)
    sorted_indices = np.argsort(-predictions)
    sorted_labels = labels[sorted_indices]
    
    # Calculate metrics at different cutoff points
    metrics = {}
    
    # Top-K precision
    for k in [10, 20, 50, 100]:
        if len(sorted_labels) >= k:
            metrics[f'P@{k}'] = np.mean(sorted_labels[:k])
    
    # Average Precision (AP)
    precisions = []
    recalls = []
    num_positives = np.sum(labels)
    
    for i in range(1, len(sorted_labels) + 1):
        if num_positives > 0:
            precisions.append(np.mean(sorted_labels[:i]))
            recalls.append(np.sum(sorted_labels[:i]) / num_positives)
    
    if precisions and recalls:
        # Average Precision
        metrics['AP'] = np.trapz(precisions, recalls)
        
        # AUC (simplified)
        metrics['AUC'] = np.trapz(precisions, dx=1/len(precisions))
    
    # NDCG@K
    for k in [10, 20, 50, 100]:
        if len(sorted_labels) >= k:
            dcg = np.sum(sorted_labels[:k] / np.log2(np.arange(2, k + 2)))
            ideal_labels = np.sort(labels)[::-1][:k]
            idcg = np.sum(ideal_labels / np.log2(np.arange(2, k + 2)))
            metrics[f'NDCG@{k}'] = dcg / idcg if idcg > 0 else 0
    
    return metrics

def calculate_fairness_metrics(predictions: np.ndarray, labels: np.ndarray, 
                              sensitive_groups: np.ndarray = None) -> Dict:
    """Calculate fairness metrics"""
    fairness = {}
    
    if sensitive_groups is not None and len(sensitive_groups) == len(predictions):
        # Group predictions by sensitive attribute
        unique_groups = np.unique(sensitive_groups)
        
        for group in unique_groups:
            mask = sensitive_groups == group
            group_preds = predictions[mask]
            group_labels = labels[mask]
            
            if len(group_preds) > 0:
                # Calculate metrics for this group
                group_metrics = calculate_metrics(group_preds, group_labels)
                
                # Store with group prefix
                for metric_name, value in group_metrics.items():
                    fairness[f'{metric_name}_group{group}'] = value
        
        # Calculate fairness disparities
        for metric in ['P@10', 'P@20', 'P@50', 'AP', 'NDCG@10']:
            group_values = []
            for group in unique_groups:
                key = f'{metric}_group{group}'
                if key in fairness:
                    group_values.append(fairness[key])
            
            if len(group_values) >= 2:
                # Statistical Parity Difference
                fairness[f'{metric}_max_diff'] = np.max(group_values) - np.min(group_values)
                fairness[f'{metric}_std'] = np.std(group_values)
    
    return fairness

def analyze_results(results: Dict) -> None:
    """Analyze and print results from all runs"""
    print("\n" + "="*80)
    print("MORAL EVALUATION RESULTS")
    print("="*80)
    
    for key, runs in results.items():
        print(f"\n{key} ({len(runs)} runs):")
        print("-" * 60)
        
        all_metrics = []
        all_fairness = []
        
        for i, (predictions, labels) in enumerate(runs):
            # Calculate metrics for this run
            metrics = calculate_metrics(predictions, labels)
            all_metrics.append(metrics)
            
            print(f"  Run {i+1}: ", end="")
            for metric_name in ['P@10', 'P@20', 'P@50', 'AP', 'NDCG@10']:
                if metric_name in metrics:
                    print(f"{metric_name}={metrics[metric_name]:.4f} ", end="")
            print()
        
        # Calculate average across runs
        if all_metrics:
            avg_metrics = {}
            for metric_name in all_metrics[0].keys():
                values = [m[metric_name] for m in all_metrics if metric_name in m]
                if values:
                    avg_metrics[metric_name] = np.mean(values)
                    avg_metrics[f'{metric_name}_std'] = np.std(values)
            
            print(f"\n  Average ({len(runs)} runs):")
            for metric_name in ['P@10', 'P@20', 'P@50', 'AP', 'NDCG@10']:
                if metric_name in avg_metrics:
                    mean = avg_metrics[metric_name]
                    std = avg_metrics.get(f'{metric_name}_std', 0)
                    print(f"    {metric_name}: {mean:.4f} ± {std:.4f}")

def plot_results(results: Dict, output_dir: str = "plots") -> None:
    """Create visualization plots"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # Prepare data for plotting
    plot_data = []
    
    for key, runs in results.items():
        parts = key.split('_')
        dataset = parts[0] if len(parts) > 0 else "unknown"
        method = parts[1] if len(parts) > 1 else "unknown"
        model = parts[2] if len(parts) > 2 else "unknown"
        
        for i, (predictions, labels) in enumerate(runs):
            metrics = calculate_metrics(predictions, labels)
            
            for metric_name, value in metrics.items():
                plot_data.append({
                    'dataset': dataset,
                    'method': method,
                    'model': model,
                    'run': i,
                    'metric': metric_name,
                    'value': value
                })
    
    if not plot_data:
        print("No data to plot")
        return
    
    import pandas as pd
    df = pd.DataFrame(plot_data)
    
    # Plot 1: Performance metrics by dataset/method
    plt.figure(figsize=(12, 6))
    
    # Filter to key metrics
    key_metrics = ['P@10', 'P@20', 'P@50', 'AP', 'NDCG@10']
    df_filtered = df[df['metric'].isin(key_metrics)]
    
    if not df_filtered.empty:
        # Group by dataset, method, model, metric
        df_avg = df_filtered.groupby(['dataset', 'method', 'model', 'metric'])['value'].mean().reset_index()
        
        # Pivot for easier plotting
        df_pivot = df_avg.pivot_table(index=['dataset', 'method', 'model'], 
                                      columns='metric', values='value').reset_index()
        
        # Create a combined label
        df_pivot['label'] = df_pivot['dataset'] + '_' + df_pivot['method'] + '_' + df_pivot['model']
        
        # Plot
        metrics_to_plot = [m for m in key_metrics if m in df_pivot.columns]
        x = np.arange(len(df_pivot))
        width = 0.8 / len(metrics_to_plot)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        for i, metric in enumerate(metrics_to_plot):
            offset = (i - len(metrics_to_plot)/2) * width + width/2
            ax.bar(x + offset, df_pivot[metric], width, label=metric)
        
        ax.set_xlabel('Dataset_Method_Model')
        ax.set_ylabel('Score')
        ax.set_title('MORAL Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(df_pivot['label'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_metrics.png", dpi=300)
        plt.savefig(f"{output_dir}/performance_metrics.pdf")
        print(f"✓ Saved plot to {output_dir}/performance_metrics.png")
    
    # Plot 2: Precision-Recall curves
    plt.figure(figsize=(10, 8))
    
    for key, runs in results.items():
        if len(runs) > 0:
            # Use first run for PR curve
            predictions, labels = runs[0]
            
            # Sort by predictions
            sorted_indices = np.argsort(-predictions)
            sorted_labels = labels[sorted_indices]
            
            # Calculate precision and recall
            precisions = []
            recalls = []
            num_positives = np.sum(labels)
            
            for i in range(1, len(sorted_labels) + 1):
                precisions.append(np.mean(sorted_labels[:i]))
                if num_positives > 0:
                    recalls.append(np.sum(sorted_labels[:i]) / num_positives)
                else:
                    recalls.append(0)
            
            # Plot
            plt.plot(recalls, precisions, label=key, linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pr_curves.png", dpi=300)
    plt.savefig(f"{output_dir}/pr_curves.pdf")
    print(f"✓ Saved plot to {output_dir}/pr_curves.png")
    
    # Plot 3: Distribution of predictions
    plt.figure(figsize=(12, 8))
    
    for idx, (key, runs) in enumerate(results.items()):
        if len(runs) > 0:
            # Combine predictions from all runs
            all_predictions = np.concatenate([p for p, _ in runs])
            
            plt.subplot(2, (len(results)+1)//2, idx+1)
            plt.hist(all_predictions, bins=50, alpha=0.7, edgecolor='black')
            plt.title(f'{key}\n(n={len(all_predictions)})')
            plt.xlabel('Prediction Score')
            plt.ylabel('Count')
            plt.grid(True, alpha=0.3)
    
    plt.suptitle('Distribution of Prediction Scores')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prediction_distributions.png", dpi=300)
    print(f"✓ Saved plot to {output_dir}/prediction_distributions.png")

def generate_report(results: Dict, output_file: str = "moral_evaluation_report.md") -> None:
    """Generate a Markdown report of results"""
    with open(output_file, 'w') as f:
        f.write("# MORAL Evaluation Report\n\n")
        f.write(f"Generated on: {np.datetime64('now', 's')}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"Total experiments evaluated: {len(results)}\n")
        f.write(f"Total runs: {sum(len(runs) for runs in results.values())}\n\n")
        
        f.write("## Results by Experiment\n\n")
        
        for key, runs in results.items():
            f.write(f"### {key}\n\n")
            f.write(f"Number of runs: {len(runs)}\n\n")
            
            if len(runs) > 0:
                # Calculate average metrics
                all_metrics = []
                for predictions, labels in runs:
                    metrics = calculate_metrics(predictions, labels)
                    all_metrics.append(metrics)
                
                # Create table
                f.write("| Metric | Mean | Std | Min | Max |\n")
                f.write("|--------|------|-----|-----|-----|\n")
                
                metric_names = set()
                for metrics in all_metrics:
                    metric_names.update(metrics.keys())
                
                for metric_name in sorted(metric_names):
                    if metric_name in ['P@10', 'P@20', 'P@50', 'AP', 'NDCG@10', 'NDCG@20']:
                        values = [m[metric_name] for m in all_metrics if metric_name in m]
                        if values:
                            f.write(f"| {metric_name} | {np.mean(values):.4f} | {np.std(values):.4f} | {np.min(values):.4f} | {np.max(values):.4f} |\n")
                
                f.write("\n")
        
        f.write("## Files Processed\n\n")
        f.write("```\n")
        for key in results.keys():
            f.write(f"{key}\n")
        f.write("```\n")

def main():
    parser = argparse.ArgumentParser(description="Evaluate MORAL fairness and performance metrics")
    parser.add_argument("--pattern", type=str, default="three_classifiers_*_final_ranking.pt",
                       help="Pattern to match ranking files")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory for output plots and reports")
    parser.add_argument("--no_plots", action="store_true",
                       help="Skip generating plots")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Load results
    print("Loading ranking files...")
    results = load_ranking_files(args.pattern)
    
    if not results:
        print("No ranking files found!")
        return
    
    # Analyze results
    analyze_results(results)
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        plot_results(results, args.output_dir)
    
    # Generate report
    print("\nGenerating report...")
    generate_report(results, f"{args.output_dir}/report.md")
    
    # Save raw results
    print("\nSaving raw results...")
    torch.save(results, f"{args.output_dir}/raw_results.pt")
    
    print(f"\n{'='*80}")
    print(f"Evaluation complete! Results saved to {args.output_dir}/")
    print(f"  - report.md: Summary report in Markdown format")
    print(f"  - raw_results.pt: All loaded results in PyTorch format")
    if not args.no_plots:
        print(f"  - Various .png/.pdf plots in {args.output_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Review the report at {args.output_dir}/report.md")
    print(f"  2. Compare with baseline methods if available")
    print(f"  3. Analyze fairness metrics across sensitive groups")

if __name__ == "__main__":
    main()