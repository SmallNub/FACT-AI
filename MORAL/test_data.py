import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(".")
from generate_data import load_raw_dataset, create_moral_splits

def analyze_distribution(dataset_name):
    print(f"\n{'='*60}")
    print(f"Analyzing distribution for: {dataset_name}")
    print('='*60)
    
    raw = load_raw_dataset(dataset_name)
    sens = raw["sens"]
    num_nodes = raw["features"].shape[0]
    all_edges = torch.sparse_coo_tensor(
        raw["adj"].coalesce().indices(),
        raw["adj"].coalesce().values(),
        raw["adj"].coalesce().size()
    )
    
    edges_coo = all_edges.coalesce().indices()
    mask = edges_coo[0] != edges_coo[1]
    all_edges = edges_coo[:, mask]
    
    splits, _ = create_moral_splits(all_edges, sens, num_nodes)
    
    def compute_edge_group_distribution(edges, sens):
        if len(edges) == 0:
            return np.array([0, 0, 0])
        
        if torch.is_tensor(edges):
            edges_np = edges.cpu().numpy()
        else:
            edges_np = edges
        
        if edges_np.ndim == 2 and edges_np.shape[0] == 2:
            edges_np = edges_np.T
        
        groups = []
        for u, v in edges_np:
            s_u = sens[u].item() if torch.is_tensor(sens) else sens[u]
            s_v = sens[v].item() if torch.is_tensor(sens) else sens[v]
            groups.append(s_u + s_v)
        
        counts = np.bincount(groups, minlength=3)
        return counts / len(edges_np)
    
    all_edges_np = all_edges.cpu().numpy().T if all_edges.shape[0] == 2 else all_edges.cpu().numpy()
    all_dist = compute_edge_group_distribution(all_edges_np, sens)
    
    train_pos_dist = compute_edge_group_distribution(splits["train"]["edge"], sens)
    train_neg_dist = compute_edge_group_distribution(splits["train"]["edge_neg"], sens)
    
    val_pos_dist = compute_edge_group_distribution(splits["valid"]["edge"], sens)
    val_neg_dist = compute_edge_group_distribution(splits["valid"]["edge_neg"], sens)
    
    test_pos_dist = compute_edge_group_distribution(splits["test"]["edge"], sens)
    test_neg_dist = compute_edge_group_distribution(splits["test"]["edge_neg"], sens)
    
    print("\nOriginal graph distribution:")
    print(f"  Group 0 (E_s'·s'): {all_dist[0]:.4f}")
    print(f"  Group 1 (E_s'·s):  {all_dist[1]:.4f}")
    print(f"  Group 2 (E_s·s):   {all_dist[2]:.4f}")
    
    print("\nTraining split:")
    print(f"  Positive edges: [{train_pos_dist[0]:.4f}, {train_pos_dist[1]:.4f}, {train_pos_dist[2]:.4f}]")
    print(f"  Negative edges: [{train_neg_dist[0]:.4f}, {train_neg_dist[1]:.4f}, {train_neg_dist[2]:.4f}]")
    
    print("\nValidation split:")
    print(f"  Positive edges: [{val_pos_dist[0]:.4f}, {val_pos_dist[1]:.4f}, {val_pos_dist[2]:.4f}]")
    print(f"  Negative edges: [{val_neg_dist[0]:.4f}, {val_neg_dist[1]:.4f}, {val_neg_dist[2]:.4f}]")
    
    print("\nTest split:")
    print(f"  Positive edges: [{test_pos_dist[0]:.4f}, {test_pos_dist[1]:.4f}, {test_pos_dist[2]:.4f}]")
    print(f"  Negative edges: [{test_neg_dist[0]:.4f}, {test_neg_dist[1]:.4f}, {test_neg_dist[2]:.4f}]")
    
    all_dists = [train_pos_dist, train_neg_dist, val_pos_dist, val_neg_dist, test_pos_dist, test_neg_dist]
    
    print("\nDistribution differences from original:")
    max_diff = 0
    for i, (name, dist) in enumerate([
        ("train_pos", train_pos_dist),
        ("train_neg", train_neg_dist),
        ("val_pos", val_pos_dist),
        ("val_neg", val_neg_dist),
        ("test_pos", test_pos_dist),
        ("test_neg", test_neg_dist)
    ]):
        diff = np.abs(dist - all_dist)
        max_diff_i = np.max(diff)
        max_diff = max(max_diff, max_diff_i)
        print(f"  {name:10s}: max diff = {max_diff_i:.4f} at groups {np.where(diff == max_diff_i)[0]}")
    
    print(f"\nMaximum difference from original: {max_diff:.4f}")
    
    if max_diff < 0.01:
        print("✓ Distribution preserved well (differences < 1%)")
    elif max_diff < 0.05:
        print("✓ Distribution reasonably preserved (differences < 5%)")
    else:
        print("✗ Distribution NOT well preserved")

def test_all_datasets():
    datasets = ["facebook", "german", "nba", "pokec_n", "pokec_z", "credit"]
    
    for dataset in datasets:
        try:
            analyze_distribution(dataset)
        except Exception as e:
            print(f"\nError analyzing {dataset}: {e}")
            continue

if __name__ == "__main__":
    test_all_datasets()