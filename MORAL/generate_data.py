import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Set
import numpy as np
import torch
from torch_geometric.data import Data

sys.path.append(".")

try:
    from datasets import Facebook, Google, German, Nba, Pokec_n, Pokec_z, Credit
except ImportError:
    print("Error: Could not import dataset classes.")
    sys.exit(1)

def load_raw_dataset(dataset_name: str, root: str = "./dataset") -> Dict:
    dataset_map = {
        "facebook": Facebook,
        "gplus": Google,
        "german": German,
        "nba": Nba,
        "pokec_n": Pokec_n,
        "pokec_z": Pokec_z,
        "credit": Credit,
    }

    if dataset_name not in dataset_map:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    dataset = dataset_map[dataset_name](root=root)
    sens = dataset.sens()
    
    if sens is None:
        print(f"Warning: Dataset {dataset_name} has no sensitive attributes.")

    return {
        "adj": dataset.adj(),
        "features": dataset.features(),
        "labels": dataset.labels(),
        "sens": sens,
        "idx_train": dataset.idx_train(),
        "idx_val": dataset.idx_val(),
        "idx_test": dataset.idx_test(),
        "sens_idx": dataset.sens_idx(),
    }

def get_all_edges(adj: torch.Tensor) -> torch.Tensor:
    if adj.is_sparse:
        edge_index = adj.coalesce().indices()
    else:
        rows, cols = np.where(adj.numpy() > 0)
        edge_index = torch.tensor([rows, cols], dtype=torch.long)

    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]

def ensure_n_by_2(edge_tensor: torch.Tensor) -> torch.Tensor:
    if edge_tensor.dim() != 2:
        raise ValueError(f"Edge tensor must be 2D, got {edge_tensor.shape}")

    if edge_tensor.shape[0] == 2 and edge_tensor.shape[1] != 2:
        return edge_tensor.t()
    
    if edge_tensor.shape[1] == 2:
        return edge_tensor

    if edge_tensor.shape[0] == 2 and edge_tensor.shape[1] == 2:
        return edge_tensor.t()

    raise ValueError(f"Invalid edge shape: {edge_tensor.shape}")

def generate_stratified_negatives(
    existing_edges: np.ndarray,
    sens: torch.Tensor,
    num_nodes: int,
    group_counts: dict,
    global_neg_set: Set[Tuple[int, int]],
    split_name: str,
) -> np.ndarray:
    pos_set = set()
    for u, v in existing_edges:
        if u > v:
            u, v = v, u
        pos_set.add((u, v))

    negatives_by_group = {0: [], 1: [], 2: []}
    total_needed = sum(group_counts.values())
    
    attempts = 0
    max_attempts = total_needed * 1000
    rng = np.random.default_rng()

    while any(len(negatives_by_group[g]) < group_counts[g] for g in [0, 1, 2]) and attempts < max_attempts:
        u = rng.integers(0, num_nodes)
        v = rng.integers(0, num_nodes)
        
        if u == v:
            continue
        
        if u > v:
            u, v = v, u
        
        pair = (u, v)
        if pair in pos_set or pair in global_neg_set:
            continue
        
        s_u = sens[u].item()
        s_v = sens[v].item()
        group = s_u + s_v
        
        if len(negatives_by_group[group]) < group_counts[group]:
            negatives_by_group[group].append([u, v])
            global_neg_set.add(pair)
        
        attempts += 1
    
    if attempts >= max_attempts:
        raise RuntimeError(f"Could not generate all stratified negatives for {split_name}")
    
    all_negatives = []
    for g in [0, 1, 2]:
        all_negatives.extend(negatives_by_group[g])
    
    return np.array(all_negatives, dtype=np.int64)

def create_moral_splits(
    all_edges: torch.Tensor,
    sens: torch.Tensor,
    num_nodes: int,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Dict:
    edges = all_edges.t().cpu().numpy()
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    np.random.seed(seed)
    
    train_edges_list = []
    val_edges_list = []
    test_edges_list = []
    
    train_counts = {0: 0, 1: 0, 2: 0}
    val_counts = {0: 0, 1: 0, 2: 0}
    test_counts = {0: 0, 1: 0, 2: 0}
    
    for group in [0, 1, 2]:
        group_edges = []
        for u, v in edges:
            s_u = sens[u].item()
            s_v = sens[v].item()
            if s_u + s_v == group:
                group_edges.append([u, v])
        
        if len(group_edges) == 0:
            continue
            
        group_edges = np.array(group_edges)
        perm = np.random.permutation(len(group_edges))
        group_edges = group_edges[perm]
        
        n_test = int(len(group_edges) * test_ratio)
        n_val = int(len(group_edges) * val_ratio)
        n_train = len(group_edges) - n_test - n_val
        
        train_edges_list.append(group_edges[:n_train])
        val_edges_list.append(group_edges[n_train:n_train + n_val])
        test_edges_list.append(group_edges[n_train + n_val:])
        
        train_counts[group] = n_train
        val_counts[group] = n_val
        test_counts[group] = n_test
    
    train_edges = np.vstack(train_edges_list) if train_edges_list else np.array([], dtype=np.int64).reshape(0, 2)
    val_edges = np.vstack(val_edges_list) if val_edges_list else np.array([], dtype=np.int64).reshape(0, 2)
    test_edges = np.vstack(test_edges_list) if test_edges_list else np.array([], dtype=np.int64).reshape(0, 2)
    
    np.random.shuffle(train_edges)
    np.random.shuffle(val_edges)
    np.random.shuffle(test_edges)

    global_neg_set = set()

    neg_train = generate_stratified_negatives(edges, sens, num_nodes, train_counts, global_neg_set, "train")
    neg_val = generate_stratified_negatives(edges, sens, num_nodes, val_counts, global_neg_set, "val")
    neg_test = generate_stratified_negatives(edges, sens, num_nodes, test_counts, global_neg_set, "test")

    splits = {
        "train": {
            "edge": torch.tensor(train_edges, dtype=torch.long),
            "edge_neg": torch.tensor(neg_train, dtype=torch.long),
        },
        "valid": {
            "edge": torch.tensor(val_edges, dtype=torch.long),
            "edge_neg": torch.tensor(neg_val, dtype=torch.long),
        },
        "test": {
            "edge": torch.tensor(test_edges, dtype=torch.long),
            "edge_neg": torch.tensor(neg_test, dtype=torch.long),
        },
    }

    for split in splits.values():
        split["edge"] = ensure_n_by_2(split["edge"])
        split["edge_neg"] = ensure_n_by_2(split["edge_neg"])

    train_edge_index = splits["train"]["edge"].t()

    return splits, train_edge_index

def verify_utils_compatibility(splits: Dict) -> bool:
    for name in ["train", "valid", "test"]:
        e = splits[name]["edge"]
        if e.dim() != 2 or e.shape[1] != 2:
            return False
        
        neg = splits[name]["edge_neg"]
        if neg.dim() != 2 or neg.shape[1] != 2:
            return False
            
    return True

def fix_existing_file(filepath: Path) -> bool:
    try:
        content = torch.load(filepath)
        if isinstance(content, tuple) and len(content) == 2:
            data, splits = content
        else:
            return False
    except Exception:
        return False

    changed = False
    
    for split in ["train", "valid", "test"]:
        for key in ["edge", "edge_neg"]:
            tensor = splits[split][key]
            if tensor.dim() == 2 and tensor.shape[0] == 2 and tensor.shape[1] != 2:
                splits[split][key] = tensor.t()
                changed = True

    if not hasattr(data, "sens") or data.sens is None:
        return False
        
    train_edges_count = splits["train"]["edge"].shape[0]
    if data.edge_index.shape[1] > train_edges_count:
        data.edge_index = splits["train"]["edge"].t()
        changed = True

    if changed:
        torch.save((data, splits), filepath)

    return verify_utils_compatibility(splits)

def generate_dataset(
    dataset_name: str,
    splits_dir: Path,
    force: bool,
    fix_existing: bool,
) -> Tuple[bool, str]:

    path = splits_dir / f"{dataset_name}.pt"

    if path.exists() and not force:
        if fix_existing:
            if fix_existing_file(path):
                return True, f"Fixed/Verified existing file: {path}"
            else:
                print("Regeneration required due to unfixable issues.")
        else:
            data, splits = torch.load(path)
            if verify_utils_compatibility(splits):
                return True, f"File already valid: {path}"

    raw = load_raw_dataset(dataset_name)
    num_nodes = raw["features"].shape[0]

    all_edges = get_all_edges(raw["adj"])
    
    splits, train_edge_index = create_moral_splits(all_edges, raw["sens"], num_nodes)

    data = Data(
        x=raw["features"],
        edge_index=train_edge_index,
        y=raw["labels"],
    )
    data.sens = raw["sens"]
    data.sens_idx = raw["sens_idx"]
    data.num_nodes = num_nodes

    splits_dir.mkdir(parents=True, exist_ok=True)
    torch.save((data, splits), path)

    verify_utils_compatibility(splits)
    return True, f"Saved {path}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--splits_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--force", action="store_true", help="Force regeneration")
    parser.add_argument("--no_fix", action="store_true", help="Skip fixing existing files")

    args = parser.parse_args()

    datasets = ["facebook", "gplus", "german", "nba", "pokec_n", "pokec_z", "credit"]

    if args.all:
        for d in datasets:
            try:
                ok, msg = generate_dataset(
                    d, args.splits_dir, args.force, not args.no_fix
                )
                print(msg)
            except Exception as e:
                print(f"Failed to generate {d}: {e}")
    elif args.dataset:
        ok, msg = generate_dataset(
            args.dataset, args.splits_dir, args.force, not args.no_fix
        )
        print(msg)
    else:
        print("Specify --dataset DATASET or --all")

if __name__ == "__main__":
    main()