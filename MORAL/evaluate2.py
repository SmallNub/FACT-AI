import torch
import numpy as np
import glob
import pandas as pd
from pathlib import Path
from utils import get_dataset

def calculate_ndkl(sorted_groups, target_dist, k=100):
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

def calculate_equal_opportunity(scores, labels, groups, k=100):
    if len(scores) == 0:
        return 0.0
    
    top_k = min(k, len(scores))
    top_indices = np.argsort(-scores)[:top_k]
    top_set = set(top_indices)
    
    tp_per_group = {0: 0, 1: 0, 2: 0}
    total_positives_per_group = {0: 0, 1: 0, 2: 0}
    
    for i in range(len(scores)):
        group = int(groups[i])
        if labels[i] == 1:
            total_positives_per_group[group] += 1
            if i in top_set:
                tp_per_group[group] += 1
    
    tpr_per_group = {}
    for g in [0, 1, 2]:
        if total_positives_per_group[g] > 0:
            tpr_per_group[g] = tp_per_group[g] / total_positives_per_group[g]
        else:
            tpr_per_group[g] = 0.0
    
    total_tp = sum(tp_per_group.values())
    total_positives = sum(total_positives_per_group.values())
    overall_tpr = total_tp / total_positives if total_positives > 0 else 0.0
    
    eo_gap = max([abs(tpr_per_group[g] - overall_tpr) for g in [0, 1, 2]])
    return eo_gap

def evaluate(k=100):
    results_dir, splits_dir = "./results", "./data/splits"
    datasets = ["facebook", "german", "nba", "pokec_n", "pokec_z", "credit"]
    # datasets = ["facebook"]
    summary_data = []

    for ds in datasets:
        try:
            adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx, data, splits = get_dataset(ds, splits_dir)
            
            if torch.is_tensor(sens):
                sens_np = sens.cpu().numpy().astype(np.int64)
            else:
                sens_np = np.array(sens).astype(np.int64)
            
            if adj.is_sparse:
                edge_coo = adj.coalesce()
                rows = edge_coo.indices()[0].cpu().numpy()
                cols = edge_coo.indices()[1].cpu().numpy()
            else:
                adj_np = adj.cpu().numpy()
                rows, cols = np.where(adj_np > 0)
            
            mask = rows != cols
            rows = rows[mask]
            cols = cols[mask]
            
            edges = []
            for u, v in zip(rows, cols):
                if u < v:
                    edges.append([u, v])
                else:
                    edges.append([v, u])
            
            edges = np.unique(np.array(edges), axis=0)
            
            edge_sens_groups = sens_np[edges[:, 0]] + sens_np[edges[:, 1]]
            counts = np.bincount(edge_sens_groups, minlength=3)
            target_dist = counts / len(edge_sens_groups)
            
            test_pos = splits["test"]["edge"]
            test_neg = splits["test"]["edge_neg"]
            test_edges_all = torch.cat([test_pos, test_neg], dim=0)
            test_labels_all = torch.cat([
                torch.ones(test_pos.size(0)),
                torch.zeros(test_neg.size(0))
            ], dim=0).numpy()
            
            test_edges_np = test_edges_all.cpu().numpy().astype(np.int64)
            test_edge_sens_groups = sens_np[test_edges_np[:, 0]] + sens_np[test_edges_np[:, 1]]
            
            test_pos_np = test_pos.cpu().numpy().astype(np.int64)
            test_pos_groups = sens_np[test_pos_np[:, 0]] + sens_np[test_pos_np[:, 1]]

            files = glob.glob(f"{results_dir}/three_classifiers_{ds}_*_final_ranking.pt")
            method_files = {}
            
            for f in files:
                parts = Path(f).stem.split("_")
                if len(parts) >= 4:
                    m_key = parts[2]
                else:
                    m_key = "UNKNOWN"
                method_files.setdefault(m_key, []).append(f)

            for method, f_list in method_files.items():
                p_runs, n_runs, eo_runs = [], [], []
                
                for f in f_list:
                    try:
                        scores_final, labels_final = torch.load(f)
                        scores_final_np = scores_final.cpu().numpy()
                        labels_final_np = labels_final.cpu().numpy()
                        
                        raw_file = f.replace("_final_ranking", "")
                        if Path(raw_file).exists():
                            scores_all = torch.load(raw_file).cpu().numpy()
                        else:
                            scores_all = scores_final_np
                        
                        if len(scores_all) != len(test_labels_all):
                            scores_all = scores_final_np
                        
                        eo_val = calculate_equal_opportunity(
                            scores_all, 
                            test_labels_all,
                            test_edge_sens_groups,
                            k=k
                        )
                        
                        idx = np.argsort(-scores_final_np)
                        ranked_groups = test_pos_groups[idx[:min(k, len(idx))]]
                        prec_at_k = np.mean(labels_final_np[idx][:k]) if len(idx) >= k else np.mean(labels_final_np[idx])
                        
                        p_runs.append(prec_at_k)
                        n_runs.append(calculate_ndkl(ranked_groups, target_dist, k=k))
                        eo_runs.append(eo_val)
                        
                    except Exception as e:
                        continue

                if p_runs:
                    summary_data.append([
                        ds.upper(),
                        method,
                        f"{np.mean(p_runs):.4f} ± {np.std(p_runs):.4f}",
                        f"{np.mean(n_runs):.4f} ± {np.std(n_runs):.4f}",
                        f"{np.mean(eo_runs):.4f} ± {np.std(eo_runs):.4f}"
                    ])
                    
        except Exception as e:
            continue

    if summary_data:
        df = pd.DataFrame(summary_data, columns=[
            "Dataset", "Method", f"Prec@{k}", f"NDKL@{k}", f"EO Gap@{k}"
        ])
        print(f"\n" + "="*80)
        print(f"EVALUATION SUMMARY @ top-{k}")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)

if __name__ == "__main__":
    evaluate(k=1000)