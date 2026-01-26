#!/usr/bin/env python3
"""
Script to analyze training logs and compute statistics across runs for each dataset.
Extracts: average values across runs, average epochs per run, epoch duration, and RAM usage.
"""

import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple
import statistics

def parse_log_file(log_file_path: str) -> Dict[str, List[Dict]]:
    """
    Parse the log file and organize data by dataset and run.
    
    Returns: dict with structure {dataset: [run1_data, run2_data, run3_data]}
    """
    datasets = defaultdict(list)
    current_dataset = None
    current_run = None
    run_data = {}
    
    # Compile regex patterns for efficiency
    dataset_pattern = re.compile(r"Processing dataset '([\w_]+)' with model")
    run_pattern = re.compile(r"Run (\d+)/(\d+)")
    epoch_pattern = re.compile(r"Epoch (\d+) \| train=([\d.]+) \| valid=([\d.]+)")
    early_stop_pattern = re.compile(r"Early stopping at epoch (\d+)")
    best_epoch_pattern = re.compile(r"Restoring best model from epoch (\d+)")
    # Updated pattern to capture milliseconds
    timestamp_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)")
    # RAM usage pattern for CPU and GPU
    ram_pattern = re.compile(r"CPU: [\d.]+% \| RAM: ([\d.]+)GB \| GPU: ([\d.]+)% \| GPU-MEM: ([\d.]+)/([\d.]+)GB")
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Skip warning lines
        if "Warning!" in line or "codecarbon" in line or "=================================================================" in line:
            continue
            
        # Check for dataset start
        dataset_match = dataset_pattern.search(line)
        if dataset_match:
            if current_dataset and run_data:
                datasets[current_dataset].append(run_data)
                run_data = {}
            current_dataset = dataset_match.group(1)
            current_run = None
            continue
        
        # Check for run start
        run_match = run_pattern.search(line)
        if run_match:
            if run_data:
                datasets[current_dataset].append(run_data)
            run_num = int(run_match.group(1))
            run_data = {
                'run_number': run_num,
                'epochs': [],
                'best_epoch': None,
                'early_stop_epoch': None,
                'start_time': None,
                'end_time': None,
                'epoch_timestamps': [],
                'ram_usage': [],  # List of RAM usage measurements
                'gpu_usage': [],  # List of GPU usage measurements
                'gpu_mem_usage': []  # List of GPU memory usage measurements
            }
            
            # Extract timestamp from this line
            time_match = timestamp_pattern.search(line)
            if time_match:
                try:
                    run_data['start_time'] = datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M:%S.%f")
                except:
                    pass
            continue
        
        # Check for epoch information
        epoch_match = epoch_pattern.search(line)
        if epoch_match and run_data:
            epoch_num = int(epoch_match.group(1))
            train_loss = float(epoch_match.group(2))
            valid_loss = float(epoch_match.group(3))
            
            run_data['epochs'].append({
                'epoch': epoch_num,
                'train_loss': train_loss,
                'valid_loss': valid_loss
            })
            
            # Extract timestamp
            time_match = timestamp_pattern.search(line)
            if time_match:
                try:
                    timestamp = datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M:%S.%f")
                    run_data['epoch_timestamps'].append((epoch_num, timestamp))
                except:
                    pass
        
        # Check for RAM usage information
        ram_match = ram_pattern.search(line)
        if ram_match and run_data:
            try:
                cpu_ram_gb = float(ram_match.group(1))  # CPU RAM in GB
                gpu_util = float(ram_match.group(2))    # GPU utilization %
                gpu_mem_used = float(ram_match.group(3))  # GPU memory used in GB
                gpu_mem_total = float(ram_match.group(4))  # GPU memory total in GB
                
                ram_measurement = {
                    'cpu_ram_gb': cpu_ram_gb,
                    'gpu_util_percent': gpu_util,
                    'gpu_mem_used_gb': gpu_mem_used,
                    'gpu_mem_total_gb': gpu_mem_total,
                    'gpu_mem_percent': (gpu_mem_used / gpu_mem_total) * 100 if gpu_mem_total > 0 else 0
                }
                run_data['ram_usage'].append(ram_measurement)
            except:
                pass
        
        # Check for early stopping
        early_stop_match = early_stop_pattern.search(line)
        if early_stop_match and run_data:
            run_data['early_stop_epoch'] = int(early_stop_match.group(1))
        
        # Check for best epoch
        best_epoch_match = best_epoch_pattern.search(line)
        if best_epoch_match and run_data:
            run_data['best_epoch'] = int(best_epoch_match.group(1))
        
        # Check for run completion
        if "Finished run" in line and run_data:
            time_match = timestamp_pattern.search(line)
            if time_match:
                try:
                    run_data['end_time'] = datetime.strptime(time_match.group(1), "%Y-%m-%d %H:%M:%S.%f")
                except:
                    pass
    
    # Don't forget to add the last run
    if current_dataset and run_data:
        datasets[current_dataset].append(run_data)
    
    return dict(datasets)

def compute_epoch_durations(run_data: Dict) -> List[float]:
    """
    Compute epoch durations from timestamps.
    Handles non-consecutive epochs by calculating average time between logged epochs.
    """
    durations = []
    
    if len(run_data['epoch_timestamps']) < 2:
        return durations
    
    # Sort by epoch number
    timestamps = sorted(run_data['epoch_timestamps'], key=lambda x: x[0])
    
    # Calculate time between consecutive logged epochs
    for i in range(1, len(timestamps)):
        epoch1, time1 = timestamps[i-1]
        epoch2, time2 = timestamps[i]
        
        if epoch2 > epoch1:  # Ensure epochs are in order
            time_diff = (time2 - time1).total_seconds()
            epoch_diff = epoch2 - epoch1
            
            # Average duration per epoch between these two logged points
            if epoch_diff > 0:
                avg_duration = time_diff / epoch_diff
                durations.append(avg_duration)
    
    return durations

def compute_statistics(datasets: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """
    Compute statistics for each dataset.
    """
    stats = {}
    
    for dataset, runs in datasets.items():
        if not runs:
            continue
            
        # Sort runs by run number
        runs = sorted(runs, key=lambda x: x['run_number'])
        
        # Calculate epochs per run
        total_epochs_per_run = []
        early_stop_epochs = []
        best_epochs = []
        
        # Calculate epoch durations
        all_epoch_durations = []
        run_durations = []
        
        # RAM usage statistics
        all_cpu_ram = []
        all_gpu_util = []
        all_gpu_mem_used = []
        all_gpu_mem_percent = []
        
        # Final validation loss from the last epoch of each run
        final_valid_losses = []
        
        # Total run times
        total_times = []
        
        for run in runs:
            # Total epochs trained (until early stopping)
            if run['early_stop_epoch']:
                total_epochs_per_run.append(run['early_stop_epoch'])
                early_stop_epochs.append(run['early_stop_epoch'])
            elif run['epochs']:
                total_epochs_per_run.append(len(run['epochs']))
            
            # Best epoch
            if run['best_epoch']:
                best_epochs.append(run['best_epoch'])
            
            # Calculate epoch durations for this run
            epoch_durations = compute_epoch_durations(run)
            if epoch_durations:
                all_epoch_durations.extend(epoch_durations)
                # Use median duration for this run
                run_durations.append(statistics.median(epoch_durations))
            
            # Collect RAM usage data
            if run['ram_usage']:
                cpu_ram_values = [m['cpu_ram_gb'] for m in run['ram_usage']]
                gpu_util_values = [m['gpu_util_percent'] for m in run['ram_usage']]
                gpu_mem_values = [m['gpu_mem_used_gb'] for m in run['ram_usage']]
                gpu_mem_percent_values = [m['gpu_mem_percent'] for m in run['ram_usage']]
                
                all_cpu_ram.extend(cpu_ram_values)
                all_gpu_util.extend(gpu_util_values)
                all_gpu_mem_used.extend(gpu_mem_values)
                all_gpu_mem_percent.extend(gpu_mem_percent_values)
            
            # Final validation loss from the last epoch of each run
            if run['epochs']:
                final_valid_losses.append(run['epochs'][-1]['valid_loss'])
            
            # Calculate total run time
            if run['start_time'] and run['end_time']:
                total_time = (run['end_time'] - run['start_time']).total_seconds()
                total_times.append(total_time)
        
        # Calculate RAM usage statistics
        cpu_ram_stats = {
            'avg': statistics.mean(all_cpu_ram) if all_cpu_ram else 0,
            'std': statistics.stdev(all_cpu_ram) if len(all_cpu_ram) > 1 else 0,
            'min': min(all_cpu_ram) if all_cpu_ram else 0,
            'max': max(all_cpu_ram) if all_cpu_ram else 0,
            'median': statistics.median(all_cpu_ram) if all_cpu_ram else 0
        }
        
        gpu_util_stats = {
            'avg': statistics.mean(all_gpu_util) if all_gpu_util else 0,
            'std': statistics.stdev(all_gpu_util) if len(all_gpu_util) > 1 else 0,
            'min': min(all_gpu_util) if all_gpu_util else 0,
            'max': max(all_gpu_util) if all_gpu_util else 0,
            'median': statistics.median(all_gpu_util) if all_gpu_util else 0
        }
        
        gpu_mem_stats = {
            'avg_used': statistics.mean(all_gpu_mem_used) if all_gpu_mem_used else 0,
            'std_used': statistics.stdev(all_gpu_mem_used) if len(all_gpu_mem_used) > 1 else 0,
            'min_used': min(all_gpu_mem_used) if all_gpu_mem_used else 0,
            'max_used': max(all_gpu_mem_used) if all_gpu_mem_used else 0,
            'avg_percent': statistics.mean(all_gpu_mem_percent) if all_gpu_mem_percent else 0,
            'median_percent': statistics.median(all_gpu_mem_percent) if all_gpu_mem_percent else 0
        }
        
        # Compile statistics
        stats[dataset] = {
            'num_runs': len(runs),
            'avg_epochs_per_run': statistics.mean(total_epochs_per_run) if total_epochs_per_run else 0,
            'std_epochs_per_run': statistics.stdev(total_epochs_per_run) if len(total_epochs_per_run) > 1 else 0,
            'avg_early_stop_epoch': statistics.mean(early_stop_epochs) if early_stop_epochs else 0,
            'avg_best_epoch': statistics.mean(best_epochs) if best_epochs else 0,
            'avg_final_valid_loss': statistics.mean(final_valid_losses) if final_valid_losses else 0,
            'std_final_valid_loss': statistics.stdev(final_valid_losses) if len(final_valid_losses) > 1 else 0,
            'avg_epoch_duration_seconds': statistics.mean(run_durations) if run_durations else 0,
            'std_epoch_duration_seconds': statistics.stdev(run_durations) if len(run_durations) > 1 else 0,
            'min_epoch_duration_seconds': min(run_durations) if run_durations else 0,
            'max_epoch_duration_seconds': max(run_durations) if run_durations else 0,
            'avg_total_run_time_seconds': statistics.mean(total_times) if total_times else 0,
            'cpu_ram_stats': cpu_ram_stats,
            'gpu_util_stats': gpu_util_stats,
            'gpu_mem_stats': gpu_mem_stats,
            'total_ram_measurements': len(all_cpu_ram)
        }
    
    return stats

def print_statistics(stats: Dict[str, Dict]):
    """
    Print the computed statistics in a readable format.
    """
    print("=" * 100)
    print("DATASET STATISTICS SUMMARY")
    print("=" * 100)
    print()
    
    for dataset, dataset_stats in stats.items():
        print(f"Dataset: {dataset.upper()}")
        print(f"  Number of runs: {dataset_stats['num_runs']}")
        print(f"  Average epochs per run: {dataset_stats['avg_epochs_per_run']:.2f} ± {dataset_stats['std_epochs_per_run']:.2f}")
        print(f"  Average early stopping epoch: {dataset_stats['avg_early_stop_epoch']:.2f}")
        print(f"  Average best epoch: {dataset_stats['avg_best_epoch']:.2f}")
        print(f"  Average final validation loss: {dataset_stats['avg_final_valid_loss']:.6f} ± {dataset_stats['std_final_valid_loss']:.6f}")
        
        if dataset_stats['avg_epoch_duration_seconds'] > 0:
            print(f"  Epoch duration statistics:")
            print(f"    Average: {dataset_stats['avg_epoch_duration_seconds']:.2f} seconds")
            print(f"    Std dev: {dataset_stats['std_epoch_duration_seconds']:.2f} seconds")
            print(f"    Min: {dataset_stats['min_epoch_duration_seconds']:.2f} seconds")
            print(f"    Max: {dataset_stats['max_epoch_duration_seconds']:.2f} seconds")
            print(f"    Average per hour: {3600/dataset_stats['avg_epoch_duration_seconds']:.2f} epochs/hour")
            print(f"    Total run time: {dataset_stats['avg_total_run_time_seconds']/60:.2f} minutes (average)")
        
        print(f"  RAM Usage Statistics ({dataset_stats['total_ram_measurements']} measurements):")
        print(f"    CPU RAM:")
        print(f"      Average: {dataset_stats['cpu_ram_stats']['avg']:.2f} GB")
        print(f"      Std dev: {dataset_stats['cpu_ram_stats']['std']:.2f} GB")
        print(f"      Range: {dataset_stats['cpu_ram_stats']['min']:.2f} - {dataset_stats['cpu_ram_stats']['max']:.2f} GB")
        print(f"      Median: {dataset_stats['cpu_ram_stats']['median']:.2f} GB")
        
        print(f"    GPU Utilization:")
        print(f"      Average: {dataset_stats['gpu_util_stats']['avg']:.1f}%")
        print(f"      Std dev: {dataset_stats['gpu_util_stats']['std']:.1f}%")
        print(f"      Range: {dataset_stats['gpu_util_stats']['min']:.1f} - {dataset_stats['gpu_util_stats']['max']:.1f}%")
        print(f"      Median: {dataset_stats['gpu_util_stats']['median']:.1f}%")
        
        print(f"    GPU Memory:")
        print(f"      Average used: {dataset_stats['gpu_mem_stats']['avg_used']:.2f} GB")
        print(f"      Range used: {dataset_stats['gpu_mem_stats']['min_used']:.2f} - {dataset_stats['gpu_mem_stats']['max_used']:.2f} GB")
        print(f"      Average utilization: {dataset_stats['gpu_mem_stats']['avg_percent']:.1f}%")
        print(f"      Median utilization: {dataset_stats['gpu_mem_stats']['median_percent']:.1f}%")
        
        print()

def create_csv_summary(stats: Dict[str, Dict], output_file: str = "dataset_summary.csv"):
    """
    Create a CSV file with the summary statistics.
    """
    import csv
    
    headers = [
        'Dataset', 'Num_Runs', 'Avg_Epochs_Per_Run', 'Std_Epochs_Per_Run',
        'Avg_Early_Stop_Epoch', 'Avg_Best_Epoch', 'Avg_Final_Valid_Loss',
        'Std_Final_Valid_Loss', 'Avg_Epoch_Duration_s', 'Std_Epoch_Duration_s',
        'Min_Epoch_Duration_s', 'Max_Epoch_Duration_s', 'Epochs_Per_Hour',
        'Avg_Total_Run_Time_min', 'CPU_RAM_Avg_GB', 'CPU_RAM_Std_GB',
        'CPU_RAM_Min_GB', 'CPU_RAM_Max_GB', 'CPU_RAM_Median_GB',
        'GPU_Util_Avg_%', 'GPU_Util_Std_%', 'GPU_Util_Min_%', 'GPU_Util_Max_%', 'GPU_Util_Median_%',
        'GPU_Mem_Avg_GB', 'GPU_Mem_Std_GB', 'GPU_Mem_Min_GB', 'GPU_Mem_Max_GB',
        'GPU_Mem_Avg_%', 'GPU_Mem_Median_%', 'Total_RAM_Measurements'
    ]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for dataset, dataset_stats in stats.items():
            epochs_per_hour = 0
            if dataset_stats['avg_epoch_duration_seconds'] > 0:
                epochs_per_hour = 3600 / dataset_stats['avg_epoch_duration_seconds']
            
            row = [
                dataset,
                dataset_stats['num_runs'],
                f"{dataset_stats['avg_epochs_per_run']:.2f}",
                f"{dataset_stats['std_epochs_per_run']:.2f}",
                f"{dataset_stats['avg_early_stop_epoch']:.2f}",
                f"{dataset_stats['avg_best_epoch']:.2f}",
                f"{dataset_stats['avg_final_valid_loss']:.6f}",
                f"{dataset_stats['std_final_valid_loss']:.6f}",
                f"{dataset_stats['avg_epoch_duration_seconds']:.2f}",
                f"{dataset_stats['std_epoch_duration_seconds']:.2f}",
                f"{dataset_stats['min_epoch_duration_seconds']:.2f}",
                f"{dataset_stats['max_epoch_duration_seconds']:.2f}",
                f"{epochs_per_hour:.2f}",
                f"{dataset_stats['avg_total_run_time_seconds']/60:.2f}",
                f"{dataset_stats['cpu_ram_stats']['avg']:.3f}",
                f"{dataset_stats['cpu_ram_stats']['std']:.3f}",
                f"{dataset_stats['cpu_ram_stats']['min']:.3f}",
                f"{dataset_stats['cpu_ram_stats']['max']:.3f}",
                f"{dataset_stats['cpu_ram_stats']['median']:.3f}",
                f"{dataset_stats['gpu_util_stats']['avg']:.2f}",
                f"{dataset_stats['gpu_util_stats']['std']:.2f}",
                f"{dataset_stats['gpu_util_stats']['min']:.2f}",
                f"{dataset_stats['gpu_util_stats']['max']:.2f}",
                f"{dataset_stats['gpu_util_stats']['median']:.2f}",
                f"{dataset_stats['gpu_mem_stats']['avg_used']:.3f}",
                f"{dataset_stats['gpu_mem_stats']['std_used']:.3f}",
                f"{dataset_stats['gpu_mem_stats']['min_used']:.3f}",
                f"{dataset_stats['gpu_mem_stats']['max_used']:.3f}",
                f"{dataset_stats['gpu_mem_stats']['avg_percent']:.2f}",
                f"{dataset_stats['gpu_mem_stats']['median_percent']:.2f}",
                dataset_stats['total_ram_measurements']
            ]
            writer.writerow(row)
    
    print(f"CSV summary saved to: {output_file}")

def print_detailed_ram_analysis(datasets: Dict[str, List[Dict]]):
    """
    Print detailed RAM usage analysis for each dataset.
    """
    print("\n" + "=" * 100)
    print("DETAILED RAM USAGE ANALYSIS")
    print("=" * 100)
    
    for dataset, runs in datasets.items():
        print(f"\nDataset: {dataset.upper()}")
        
        all_cpu_ram = []
        all_gpu_util = []
        all_gpu_mem = []
        
        for run in runs:
            if run['ram_usage']:
                cpu_ram_values = [m['cpu_ram_gb'] for m in run['ram_usage']]
                gpu_util_values = [m['gpu_util_percent'] for m in run['ram_usage']]
                gpu_mem_values = [m['gpu_mem_used_gb'] for m in run['ram_usage']]
                
                all_cpu_ram.extend(cpu_ram_values)
                all_gpu_util.extend(gpu_util_values)
                all_gpu_mem.extend(gpu_mem_values)
                
                print(f"  Run {run['run_number']}: {len(run['ram_usage'])} RAM measurements")
                if cpu_ram_values:
                    print(f"    CPU RAM: {statistics.mean(cpu_ram_values):.2f} GB avg, "
                          f"{min(cpu_ram_values):.2f}-{max(cpu_ram_values):.2f} GB range")
                if gpu_util_values:
                    print(f"    GPU Util: {statistics.mean(gpu_util_values):.1f}% avg, "
                          f"{min(gpu_util_values):.1f}-{max(gpu_util_values):.1f}% range")
                if gpu_mem_values:
                    print(f"    GPU Mem: {statistics.mean(gpu_mem_values):.2f} GB avg, "
                          f"{min(gpu_mem_values):.2f}-{max(gpu_mem_values):.2f} GB range")
        
        if all_cpu_ram:
            print(f"\n  Overall Dataset Statistics:")
            print(f"    Total RAM measurements: {len(all_cpu_ram)}")
            print(f"    CPU RAM - Overall: {statistics.mean(all_cpu_ram):.2f} GB avg, "
                  f"{statistics.stdev(all_cpu_ram):.2f} GB std")
            print(f"    GPU Util - Overall: {statistics.mean(all_gpu_util):.1f}% avg, "
                  f"{statistics.stdev(all_gpu_util):.1f}% std")
            print(f"    GPU Mem - Overall: {statistics.mean(all_gpu_mem):.2f} GB avg, "
                  f"{statistics.stdev(all_gpu_mem):.2f} GB std")

def print_detailed_epoch_times(datasets: Dict[str, List[Dict]]):
    """
    Print detailed epoch timing information for debugging.
    """
    print("\n" + "=" * 100)
    print("DETAILED EPOCH TIMING ANALYSIS")
    print("=" * 100)
    
    for dataset, runs in datasets.items():
        print(f"\nDataset: {dataset.upper()}")
        for run in runs:
            print(f"\n  Run {run['run_number']}:")
            if run['epoch_timestamps']:
                print(f"    Total logged epochs: {len(run['epoch_timestamps'])}")
                
                # Calculate time between first and last epoch
                if len(run['epoch_timestamps']) >= 2:
                    first_epoch, first_time = run['epoch_timestamps'][0]
                    last_epoch, last_time = run['epoch_timestamps'][-1]
                    
                    total_time = (last_time - first_time).total_seconds()
                    epoch_diff = last_epoch - first_epoch
                    
                    if epoch_diff > 0:
                        avg_time_per_epoch = total_time / epoch_diff
                        print(f"    First epoch: {first_epoch} at {first_time}")
                        print(f"    Last epoch: {last_epoch} at {last_time}")
                        print(f"    Time between: {total_time:.2f} seconds for {epoch_diff} epochs")
                        print(f"    Average per epoch: {avg_time_per_epoch:.2f} seconds")

def main():
    # Specify your log file path here
    log_file_path = "slurm/moral_early_stop_train_18649179.err"
    
    try:
        print("Parsing log file...")
        datasets = parse_log_file(log_file_path)
        
        print(f"Found {len(datasets)} datasets: {list(datasets.keys())}")
        for dataset, runs in datasets.items():
            total_ram_measurements = sum(len(run.get('ram_usage', [])) for run in runs)
            print(f"  {dataset}: {len(runs)} runs, {total_ram_measurements} RAM measurements")
        
        print("\nComputing statistics...")
        stats = compute_statistics(datasets)
        
        print_statistics(stats)
        
        # Print detailed analyses
        print_detailed_ram_analysis(datasets)
        print_detailed_epoch_times(datasets)
        
        # Optional: Save to CSV
        create_csv_summary(stats)
        
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        print("Please update the 'log_file_path' variable in the script.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()