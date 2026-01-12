import torch
import numpy as np
import glob
import pandas as pd
from pathlib import Path
from utils import get_dataset

# --- MORAL PAPER METRICS (Prec@1000 & NDKL) ---


def calculate_ndkl(sorted_groups, target_dist, k=1000):
    k = min(k, len(sorted_groups))
    actual_counts = np.zeros_like(target_dist)
    dkl_val = 0.0
    eps = 1e-10

    # 1. Calculate Actual DKL
    for i in range(1, k + 1):
        actual_counts[int(sorted_groups[i-1])] += 1
        p_i = np.clip(actual_counts / i, eps, 1.0)
        t_i = np.clip(target_dist, eps, 1.0)
        # Standard KL Divergence
        kl = np.sum(p_i * np.log(p_i / t_i))
        dkl_val += kl / np.log2(i + 1)

    # 2. Calculate Best possible DKL (Ideal prefix distribution)
    best_counts = np.zeros_like(target_dist)
    best_dkl = 0.0
    for i in range(1, k + 1):
        # Greedy choice to minimize KL at each step
        if i == 1:
            choice = np.argmax(target_dist)
        else:
            # We choose the group that brings the current distribution closest to target
            choice = np.argmax((target_dist * i) - best_counts)
        
        best_counts[choice] += 1
        p_b = np.clip(best_counts / i, eps, 1.0)
        kl_b = np.sum(p_b * np.log(p_b / t_i))
        best_dkl += kl_b / np.log2(i + 1)

    # 3. Normalization (Sum of discounts as used in MORAL repo)
    norm = np.sum(1.0 / np.log2(np.arange(1, k + 1) + 1))
    
    # Clip at 0 to avoid negative values caused by floating point/greedy approximation
    res = (dkl_val - best_dkl) / norm
    return res


# --- EXECUTION & CONSOLIDATION ---


def evaluate():
    results_dir, splits_dir = "./results", "./data/splits"
    datasets = ["facebook", "german", "nba", "pokec_n", "pokec_z", "credit"]
    summary_data = []

    for ds in datasets:
        try:
            # Reconstruct groups using get_dataset
            _, _, _, _, _, _, sens, _, _, splits = get_dataset(ds, splits_dir)
            test_edges = splits["test"]["edge"]
            sens_np = sens.cpu().numpy() if torch.is_tensor(sens) else np.array(sens)
            dyadic_groups = (
                sens_np[test_edges[:, 0]] != sens_np[test_edges[:, 1]]
            ).astype(int)
            target_dist = np.bincount(dyadic_groups) / len(dyadic_groups)

            # Group runs by method
            files = glob.glob(f"{results_dir}/*_{ds}_*_final_ranking.pt")
            method_files = {}
            for f in files:
                m_key = Path(f).stem.split("_")[
                    1
                ]  # Extracts "classifiers", "MORAL", etc.
                method_files.setdefault(m_key, []).append(f)

            for method, f_list in method_files.items():
                p_runs, n_runs = [], []
                for f in f_list:
                    scores, labels = torch.load(f)
                    idx = np.argsort(-scores.cpu().numpy())
                    p_runs.append(np.mean(labels.cpu().numpy()[idx][:1000]))
                    n_runs.append(calculate_ndkl(dyadic_groups[idx], target_dist))

                summary_data.append(
                    [
                        ds.upper(),
                        method,
                        f"{np.mean(p_runs):.4f} ± {np.std(p_runs):.4f}",
                        f"{np.mean(n_runs):.4f} ± {np.std(n_runs):.4f}",
                    ]
                )
        except:
            continue

    df = pd.DataFrame(summary_data, columns=["Dataset", "Method", "Prec@1000", "NDKL"])
    print("\n" + "=" * 70 + "\nMORAL REPRODUCTION SUMMARY\n" + "=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    evaluate()
