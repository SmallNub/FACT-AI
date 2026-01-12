import torch
import numpy as np
import glob
import pandas as pd
from pathlib import Path
from utils import get_dataset

def calculate_ndkl(sorted_groups, target_dist, k=1000):
    k = min(k, len(sorted_groups))
    actual_counts = np.zeros_like(target_dist, dtype=np.float32)
    dkl_val = 0.0
    eps = 1e-10
    
    target_dist = np.clip(target_dist, eps, 1.0)
    
    for i in range(1, k + 1):
        actual_counts[int(sorted_groups[i-1])] += 1
        p_i = actual_counts / i
        p_i = np.clip(p_i, eps, 1.0)
        kl = np.sum(p_i * np.log(p_i / target_dist))
        dkl_val += kl / np.log2(i + 1)
    
    norm = np.sum(1.0 / np.log2(np.arange(1, k + 1) + 1))
    return dkl_val / norm

def evaluate():
    results_dir, splits_dir = "./results", "./data/splits"
    datasets = ["facebook", "german", "nba", "pokec_n", "pokec_z", "credit"]
    summary_data = []

    for ds in datasets:
        try:
            _, _, _, _, _, _, sens, _, _, splits = get_dataset(ds, splits_dir)
            test_edges = splits["test"]["edge"]
            
            if torch.is_tensor(sens):
                sens_np = sens.cpu().numpy().astype(np.int64)
            else:
                sens_np = np.array(sens).astype(np.int64)
            
            test_edges_np = test_edges.cpu().numpy().astype(np.int64)
            edge_sens_groups = sens_np[test_edges_np[:, 0]] + sens_np[test_edges_np[:, 1]]
            
            counts = np.bincount(edge_sens_groups, minlength=3)
            target_dist = counts / len(edge_sens_groups)

            files = glob.glob(f"{results_dir}/*_{ds}_*_final_ranking.pt")
            method_files = {}
            for f in files:
                parts = Path(f).stem.split("_")
                if len(parts) >= 4:
                    m_key = parts[2]
                else:
                    m_key = "UNKNOWN"
                method_files.setdefault(m_key, []).append(f)

            for method, f_list in method_files.items():
                p_runs, n_runs = [], []
                for f in f_list:
                    try:
                        scores, labels = torch.load(f)
                        scores_np = scores.cpu().numpy()
                        idx = np.argsort(-scores_np)
                        ranked_groups = edge_sens_groups[idx]
                        labels_np = labels.cpu().numpy()
                        prec_at_k = np.mean(labels_np[idx][:1000]) if len(idx) >= 1000 else np.mean(labels_np[idx])
                        p_runs.append(prec_at_k)
                        ndkl_val = calculate_ndkl(ranked_groups, target_dist)
                        n_runs.append(ndkl_val)
                    except:
                        continue

                if p_runs:
                    summary_data.append([
                        ds.upper(),
                        method,
                        f"{np.mean(p_runs):.4f} ± {np.std(p_runs):.4f}",
                        f"{np.mean(n_runs):.4f} ± {np.std(n_runs):.4f}",
                    ])
                    
        except:
            continue

    if summary_data:
        df = pd.DataFrame(summary_data, columns=["Dataset", "Method", "Prec@1000", "NDKL"])
        print("\n" + "="*70)
        print("MORAL REPRODUCTION SUMMARY")
        print("="*70)
        print(df.to_string(index=False))
        print("="*70)

if __name__ == "__main__":
    evaluate()