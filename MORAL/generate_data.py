"""
Generate MORAL-compatible data files with correct format for utils.py
Based on: https://arxiv.org/pdf/2511.06568 (Breaking the Dyadic Barrier)
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Set
import numpy as np
import torch
from torch_geometric.data import Data

# Add the dataset module to path
sys.path.append(".")

try:
    from datasets import Facebook, Google, German, Nba, Pokec_n, Pokec_z, Credit
except ImportError:
    print("Error: Could not import dataset classes. Ensure 'datasets.py' is in the python path.")
    sys.exit(1)


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_raw_dataset(dataset_name: str, root: str = "./dataset") -> Dict:
    """Load raw dataset using MORAL dataset classes"""
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

    print(f"Loading {dataset_name} dataset...")
    dataset = dataset_map[dataset_name](root=root)

    # Ensure sensitivities are available
    sens = dataset.sens()
    if sens is None:
        print(f"Warning: Dataset {dataset_name} has no sensitive attributes.")

    return {
        "adj": dataset.adj(),
        "features": dataset.features(),
        "labels": dataset.labels(),
        "sens": sens,
        "idx_train": dataset.idx_train(),  # Note: These are node splits, usually ignored for link pred
        "idx_val": dataset.idx_val(),
        "idx_test": dataset.idx_test(),
        "sens_idx": dataset.sens_idx(),
    }


# ---------------------------------------------------------------------
# Edge utilities
# ---------------------------------------------------------------------

def get_all_edges(adj: torch.Tensor) -> torch.Tensor:
    """Return edge_index in shape [2, E] with self-loops removed"""
    if adj.is_sparse:
        edge_index = adj.coalesce().indices()
    else:
        rows, cols = np.where(adj.numpy() > 0)
        edge_index = torch.tensor([rows, cols], dtype=torch.long)

    # remove self-loops
    mask = edge_index[0] != edge_index[1]
    return edge_index[:, mask]


def ensure_n_by_2(edge_tensor: torch.Tensor) -> torch.Tensor:
    """
    Ensure edge tensor is [N, 2] as expected by MORAL splits.
    Standard PyG is [2, N], but MORAL utils often expect [N, 2] lists.
    """
    if edge_tensor.dim() != 2:
        raise ValueError(f"Edge tensor must be 2D, got {edge_tensor.shape}")

    # If shape is [2, N] (and N != 2), transpose it.
    # Ambiguity arises if N=2, but usually E >> 2.
    if edge_tensor.shape[0] == 2 and edge_tensor.shape[1] != 2:
        return edge_tensor.t()
    
    # If shape is [N, 2], keep it.
    if edge_tensor.shape[1] == 2:
        return edge_tensor

    # Fallback for empty or 2x2
    if edge_tensor.shape[0] == 2 and edge_tensor.shape[1] == 2:
        return edge_tensor.t() # Default to [N, 2] logic if ambiguous? Usually [2, N] is standard input

    raise ValueError(f"Invalid edge shape: {edge_tensor.shape}")


# ---------------------------------------------------------------------
# Negative edge generation
# ---------------------------------------------------------------------

def generate_negative_edges(
    existing_edges: np.ndarray,
    num_nodes: int,
    num_needed: int,
    global_neg_set: Set[Tuple[int, int]],
    split_name: str,
) -> np.ndarray:
    """
    Generate truly non-existent edges.
    - No overlap with positive edges
    - No overlap across splits (managed via global_neg_set)
    - Undirected graphs assumed
    """
    print(f"    Generating {num_needed} negative edges for {split_name}...")

    # Populate set of existing edges for fast O(1) lookup
    pos_set = set()
    for u, v in existing_edges:
        if u > v:
            u, v = v, u
        pos_set.add((u, v))

    negatives = []
    attempts = 0
    # Increase max attempts for dense graphs
    max_attempts = max(num_needed * 100, 1000000)

    rng = np.random.default_rng()

    while len(negatives) < num_needed and attempts < max_attempts:
        # Batch generation for speed
        batch_size = min(num_needed - len(negatives), 10000) * 2
        us = rng.integers(0, num_nodes, size=batch_size)
        vs = rng.integers(0, num_nodes, size=batch_size)
        
        for u, v in zip(us, vs):
            if len(negatives) >= num_needed:
                break
                
            if u == v:
                continue
            
            if u > v:
                u, v = v, u
            
            pair = (u, v)
            if pair not in pos_set and pair not in global_neg_set:
                negatives.append([u, v])
                global_neg_set.add(pair)
        
        attempts += batch_size

    if len(negatives) < num_needed:
        raise RuntimeError(
            f"Could not generate enough negative edges for {split_name} "
            f"({len(negatives)}/{num_needed}). Graph might be too dense."
        )

    return np.array(negatives, dtype=np.int64)


# ---------------------------------------------------------------------
# Split creation
# ---------------------------------------------------------------------

def create_moral_splits(
    all_edges: torch.Tensor,
    num_nodes: int,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Dict:
    """
    Create MORAL-compatible splits.
    Args:
        all_edges: Tensor of shape [2, E]
    Returns:
        splits: Dict containing train/valid/test splits (pos and neg)
        train_edge_index: Tensor of shape [2, E_train] for the Data object
    """
    print(f"  Creating splits from {all_edges.shape[1]} edges...")

    # Convert to numpy [E, 2] for processing
    edges = all_edges.t().cpu().numpy()

    # Canonicalize undirected edges (u < v) and remove duplicates
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    print(f"  Unique undirected edges: {edges.shape[0]}")

    np.random.seed(seed)
    perm = np.random.permutation(edges.shape[0])

    n_test = int(len(edges) * test_ratio)
    n_val = int(len(edges) * val_ratio)
    n_train = len(edges) - n_test - n_val

    train_edges = edges[perm[:n_train]]
    val_edges = edges[perm[n_train:n_train + n_val]]
    test_edges = edges[perm[n_train + n_val:]]

    print(f"  Split sizes: train={n_train}, val={n_val}, test={n_test}")

    # Track used negatives to ensure disjoint negative sets across splits
    global_neg_set = set()

    # Generate negatives (1:1 ratio used here; MORAL ranking may use more at eval time, 
    # but the stored file usually keeps 1:1 or specific ratio)
    neg_train = generate_negative_edges(edges, num_nodes, n_train, global_neg_set, "train")
    neg_val = generate_negative_edges(edges, num_nodes, n_val, global_neg_set, "val")
    neg_test = generate_negative_edges(edges, num_nodes, n_test, global_neg_set, "test")

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

    # Ensure [N, 2] shape for MORAL compatibility
    for split in splits.values():
        split["edge"] = ensure_n_by_2(split["edge"])
        split["edge_neg"] = ensure_n_by_2(split["edge_neg"])

    # Prepare PyG-compatible [2, E_train] edge_index for the Data object
    # Transpose back to [2, N]
    train_edge_index = splits["train"]["edge"].t()

    return splits, train_edge_index


# ---------------------------------------------------------------------
# File fixing / verification
# ---------------------------------------------------------------------

def verify_utils_compatibility(splits: Dict) -> bool:
    print("\n  Verifying utils.py compatibility...")
    for name in ["train", "valid", "test"]:
        e = splits[name]["edge"]
        # MORAL utils.py expects [N, 2]
        if e.dim() != 2 or e.shape[1] != 2:
            print(f"    ✗ {name} has invalid shape {e.shape} (Expected [N, 2])")
            return False
        
        # Check negatives
        neg = splits[name]["edge_neg"]
        if neg.dim() != 2 or neg.shape[1] != 2:
            print(f"    ✗ {name} negative edges invalid shape {neg.shape}")
            return False
            
        print(f"    ✓ {name}: {e.shape} (pos), {neg.shape} (neg)")
    return True


def fix_existing_file(filepath: Path) -> bool:
    print(f"\nChecking existing file: {filepath}")
    try:
        content = torch.load(filepath)
        if isinstance(content, tuple) and len(content) == 2:
            data, splits = content
        else:
            print("    ✗ File structure invalid (expected tuple(Data, splits))")
            return False
    except Exception as e:
        print(f"    ✗ Could not load file: {e}")
        return False

    changed = False
    
    # 1. Fix Shapes
    for split in ["train", "valid", "test"]:
        for key in ["edge", "edge_neg"]:
            tensor = splits[split][key]
            # If shape is [2, N], convert to [N, 2]
            if tensor.dim() == 2 and tensor.shape[0] == 2 and tensor.shape[1] != 2:
                splits[split][key] = tensor.t()
                changed = True

    # 2. Fix Missing Sensitive Attributes in Data
    if not hasattr(data, "sens") or data.sens is None:
        print("    ! Data object missing 'sens' attribute. Cannot fix without raw data reload.")
        # Note: We cannot easily fix missing raw data from the .pt file alone. 
        # The user should force regenerate if this is the case.
        return False
        
    # 3. Fix Leakage (Data.edge_index should only have train edges)
    # Check if edge_index size matches train size
    train_edges_count = splits["train"]["edge"].shape[0]
    if data.edge_index.shape[1] > train_edges_count:
        print("    ! Data leakage detected (edge_index > train_edges). Fixing...")
        data.edge_index = splits["train"]["edge"].t()
        changed = True

    if changed:
        torch.save((data, splits), filepath)
        print("  ✓ File fixed and saved")

    return verify_utils_compatibility(splits)


# ---------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------

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
                print("    -> Regeneration required due to unfixable issues (e.g., missing sens).")
        else:
            data, splits = torch.load(path)
            if verify_utils_compatibility(splits):
                return True, f"File already valid: {path}"

    print(f"\nGenerating dataset: {dataset_name}")

    raw = load_raw_dataset(dataset_name)
    num_nodes = raw["features"].shape[0]

    all_edges = get_all_edges(raw["adj"])
    
    # Create splits and get the training-only edge_index
    splits, train_edge_index = create_moral_splits(all_edges, num_nodes)

    # Construct Data object
    # CRITICAL: Include sens and sens_idx for fairness tasks
    # CRITICAL: Use only train_edge_index to prevent leakage
    data = Data(
        x=raw["features"], # Using 'x' is standard PyG, 'features' implies custom
        edge_index=train_edge_index,
        y=raw["labels"],
    )
    # Attach sensitivity info directly to Data object
    data.sens = raw["sens"]
    data.sens_idx = raw["sens_idx"]
    
    # Store other useful metadata if needed
    data.num_nodes = num_nodes

    splits_dir.mkdir(parents=True, exist_ok=True)
    torch.save((data, splits), path)

    verify_utils_compatibility(splits)
    return True, f"Saved {path}"


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

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