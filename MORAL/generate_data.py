"""
Generate MORAL-compatible data files with correct format for utils.py
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# Add the dataset module to path
sys.path.append('.')

try:
    from datasets import Facebook, Google, German, Nba, Pokec_n, Pokec_z, Credit
except ImportError:
    print("Error: Could not import dataset classes.")
    sys.exit(1)

def load_raw_dataset(dataset_name: str, root: str = "./dataset") -> Dict:
    """Load raw dataset using your dataset classes"""
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
        raise ValueError(f"Unknown dataset '{dataset_name}'.")
    
    print(f"Loading {dataset_name} dataset...")
    dataset = dataset_map[dataset_name](root=root)
    
    return {
        'adj': dataset.adj(),
        'features': dataset.features(),
        'labels': dataset.labels(),
        'sens': dataset.sens(),
        'idx_train': dataset.idx_train(),
        'idx_val': dataset.idx_val(),
        'idx_test': dataset.idx_test(),
        'sens_idx': dataset.sens_idx(),
    }

def get_all_edges(adj: torch.Tensor) -> torch.Tensor:
    """Get all edges from adjacency matrix"""
    if adj.is_sparse:
        edge_index = adj.coalesce().indices()
    else:
        adj_np = adj.numpy()
        rows, cols = np.where(adj_np > 0)
        edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    
    # Remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    
    return edge_index

def ensure_correct_shape(tensor: torch.Tensor, expected_dim: int = 2) -> torch.Tensor:
    """
    Ensure tensor has shape [N, 2] for edges or [2, N] for edge_index.
    This fixes the common issue where edges are stored in wrong orientation.
    """
    if tensor.dim() != 2:
        return tensor
    
    # If tensor is [2, N], transpose to [N, 2] for edge storage
    if tensor.size(0) == 2 and tensor.size(1) > 2:
        print(f"    Fixing shape: {tensor.shape} -> {tensor.t().shape}")
        return tensor.t()
    
    # If tensor is [N, 2] but N < 2 (edge case), keep as is
    if tensor.size(1) == 2:
        return tensor
    
    # Try to reshape if it's 1D
    if tensor.dim() == 1 and tensor.size(0) % 2 == 0:
        new_tensor = tensor.view(-1, 2)
        print(f"    Reshaping 1D tensor: {tensor.shape} -> {new_tensor.shape}")
        return new_tensor
    
    return tensor

def generate_negative_edges(
    existing_edges: np.ndarray,
    num_nodes: int,
    num_needed: int,
    split_name: str = "train"
) -> np.ndarray:
    """Generate non-existent edges"""
    print(f"    Generating {num_needed} negative edges for {split_name}...")
    
    # Create set of existing edges (both directions for undirected)
    edge_set = set()
    for u, v in existing_edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    negatives = []
    attempts = 0
    max_attempts = num_needed * 20
    
    while len(negatives) < num_needed and attempts < max_attempts:
        u = np.random.randint(0, num_nodes)
        v = np.random.randint(0, num_nodes)
        
        # For undirected graphs, we can store as (min, max)
        if u > v:
            u, v = v, u
        
        if u != v and (u, v) not in edge_set:
            negatives.append([u, v])
            edge_set.add((u, v))
            edge_set.add((v, u))
        
        attempts += 1
    
    if len(negatives) < num_needed:
        print(f"    Warning: Only generated {len(negatives)}/{num_needed} negative edges")
        # Pad with random edges
        while len(negatives) < num_needed:
            u = np.random.randint(0, num_nodes)
            v = np.random.randint(0, num_nodes)
            if u > v:
                u, v = v, u
            negatives.append([u, v])
    
    return np.array(negatives[:num_needed])

def create_moral_splits(
    all_edges: torch.Tensor,
    num_nodes: int,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1
) -> Dict:
    """
    Create splits in the exact format expected by utils.py
    Returns splits where each edge tensor has shape [N, 2]
    """
    print(f"  Creating splits from {all_edges.shape[1]} edges...")
    
    # Convert to numpy array [num_edges, 2]
    edges_np = all_edges.t().cpu().numpy()
    
    # Remove duplicates (keep unique edges)
    edges_np = np.unique(np.sort(edges_np, axis=1), axis=0)
    print(f"  Unique edges: {edges_np.shape[0]}")
    
    # Split positive edges
    total_edges = edges_np.shape[0]
    n_test = int(total_edges * test_ratio)
    n_val = int(total_edges * val_ratio)
    n_train = total_edges - n_test - n_val
    
    # Shuffle and split
    np.random.seed(42)
    shuffled_indices = np.random.permutation(total_edges)
    
    train_idx = shuffled_indices[:n_train]
    val_idx = shuffled_indices[n_train:n_train + n_val]
    test_idx = shuffled_indices[n_train + n_val:]
    
    pos_train = edges_np[train_idx]
    pos_val = edges_np[val_idx]
    pos_test = edges_np[test_idx]
    
    print(f"  Split sizes - Train: {n_train}, Val: {n_val}, Test: {n_test}")
    
    # Generate negative edges for each split
    # Use ALL existing edges when checking for negatives
    all_edges_set = set([(u, v) for u, v in edges_np])
    
    neg_train = generate_negative_edges(edges_np, num_nodes, n_train, "train")
    neg_val = generate_negative_edges(edges_np, num_nodes, n_val, "val")
    neg_test = generate_negative_edges(edges_np, num_nodes, n_test, "test")
    
    # Create splits with CORRECT shape: [N, 2]
    splits = {
        'train': {
            'edge': torch.tensor(pos_train, dtype=torch.long),  # Shape: [N, 2]
            'edge_neg': torch.tensor(neg_train, dtype=torch.long)  # Shape: [N, 2]
        },
        'valid': {
            'edge': torch.tensor(pos_val, dtype=torch.long),    # Shape: [N, 2]
            'edge_neg': torch.tensor(neg_val, dtype=torch.long)    # Shape: [N, 2]
        },
        'test': {
            'edge': torch.tensor(pos_test, dtype=torch.long),   # Shape: [N, 2]
            'edge_neg': torch.tensor(neg_test, dtype=torch.long)   # Shape: [N, 2]
        }
    }
    
    # Double-check and fix shapes if needed
    for split_name in ['train', 'valid', 'test']:
        splits[split_name]['edge'] = ensure_correct_shape(splits[split_name]['edge'])
        splits[split_name]['edge_neg'] = ensure_correct_shape(splits[split_name]['edge_neg'])
    
    print("  Final split shapes:")
    for split_name in ['train', 'valid', 'test']:
        edge_shape = splits[split_name]['edge'].shape
        neg_shape = splits[split_name]['edge_neg'].shape
        print(f"    {split_name}: pos{edge_shape}, neg{neg_shape}")
    
    return splits

def verify_utils_compatibility(splits: Dict) -> bool:
    """Verify that splits are compatible with utils.py"""
    print("\n  Verifying utils.py compatibility...")
    
    try:
        # Check train edges (the ones utils.py will transpose)
        train_edge = splits['train']['edge']
        
        # Should be [N, 2]
        if train_edge.dim() != 2 or train_edge.shape[1] != 2:
            print(f"    ✗ Wrong shape: {train_edge.shape}, expected [N, 2]")
            return False
        
        # After transpose, should be [2, N]
        train_edge_t = train_edge.t()
        if train_edge_t.shape[0] != 2:
            print(f"    ✗ After transpose: {train_edge_t.shape}, expected [2, N]")
            return False
        
        print(f"    ✓ Train edges: {train_edge.shape} -> after .t(): {train_edge_t.shape}")
        
        # Check all splits have same format
        for split_name in ['valid', 'test']:
            edge = splits[split_name]['edge']
            if edge.dim() != 2 or edge.shape[1] != 2:
                print(f"    ✗ {split_name} wrong shape: {edge.shape}")
                return False
            print(f"    ✓ {split_name} edges: {edge.shape}")
        
        return True
        
    except Exception as e:
        print(f"    ✗ Verification failed: {str(e)}")
        return False

def fix_existing_file(filepath: Path) -> bool:
    """Fix an existing .pt file if it has wrong shape"""
    print(f"\nChecking existing file: {filepath}")
    
    try:
        data, splits = torch.load(filepath)
        needs_fix = False
        
        # Check each split
        for split_name in ['train', 'valid', 'test']:
            if split_name in splits:
                edge = splits[split_name]['edge']
                edge_neg = splits[split_name]['edge_neg']
                
                # Fix if shape is [2, N] instead of [N, 2]
                if edge.dim() == 2 and edge.shape[0] == 2:
                    print(f"  Fixing {split_name} edges: {edge.shape} -> {edge.t().shape}")
                    splits[split_name]['edge'] = edge.t()
                    needs_fix = True
                
                if edge_neg.dim() == 2 and edge_neg.shape[0] == 2:
                    print(f"  Fixing {split_name} neg edges: {edge_neg.shape} -> {edge_neg.t().shape}")
                    splits[split_name]['edge_neg'] = edge_neg.t()
                    needs_fix = True
        
        if needs_fix:
            # Save fixed file
            torch.save((data, splits), filepath)
            print(f"  ✓ File fixed and saved")
            
            # Verify the fix
            return verify_utils_compatibility(splits)
        else:
            print(f"  ✓ File already has correct format")
            return verify_utils_compatibility(splits)
            
    except Exception as e:
        print(f"  ✗ Error checking file: {str(e)}")
        return False

def generate_dataset(
    dataset_name: str,
    splits_dir: Path = Path("data/splits"),
    force: bool = False,
    fix_existing: bool = True
) -> Tuple[bool, str]:
    """
    Generate or fix dataset for MORAL
    """
    splits_path = splits_dir / f"{dataset_name}.pt"
    
    # If file exists and we're not forcing regeneration
    if splits_path.exists() and not force:
        if fix_existing:
            # Try to fix the existing file
            success = fix_existing_file(splits_path)
            if success:
                return True, f"Fixed existing file: {splits_path}"
            else:
                print(f"  Will regenerate due to fix failure...")
        else:
            # Just check the file
            print(f"  File exists: {splits_path}")
            try:
                data, splits = torch.load(splits_path)
                if verify_utils_compatibility(splits):
                    return True, f"File exists and is valid: {splits_path}"
                else:
                    print(f"  File exists but has wrong format, regenerating...")
            except:
                print(f"  File exists but cannot be loaded, regenerating...")
    
    # Generate new file
    print(f"\n{'='*60}")
    print(f"Generating data for: {dataset_name}")
    print('='*60)
    
    try:
        # Load raw data
        data_dict = load_raw_dataset(dataset_name)
        
        # Get basic info
        features = data_dict['features']
        labels = data_dict['labels']
        num_nodes = features.shape[0]
        
        print(f"  Nodes: {num_nodes}")
        print(f"  Features: {features.shape[1]}")
        
        # Get all edges
        all_edges = get_all_edges(data_dict['adj'])
        print(f"  Total edges in graph: {all_edges.shape[1]}")
        
        # Create splits
        splits = create_moral_splits(all_edges, num_nodes)
        
        # Create Data object
        data = Data(
            num_nodes=num_nodes,
            edge_index=all_edges,  # All edges in shape [2, E]
            y=labels
        )
        
        # Verify before saving
        if not verify_utils_compatibility(splits):
            print(f"  ⚠ Warning: Splits may not be compatible with utils.py")
        
        # Ensure directory exists
        splits_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the file
        torch.save((data, splits), splits_path)
        print(f"\n✓ Saved: {splits_path}")
        
        return True, f"Successfully generated {splits_path}"
        
    except Exception as e:
        import traceback
        print(f"\n✗ Error generating {dataset_name}:")
        print(f"  {str(e)}")
        traceback.print_exc()
        return False, f"Failed: {str(e)}"

def test_with_real_utils(dataset_name: str, splits_dir: Path) -> bool:
    """Test with the actual utils.py get_dataset function"""
    print(f"\nTesting {dataset_name} with actual utils.py...")
    
    try:
        # Import the actual utils.py
        from utils import get_dataset
        
        # This should work if our format is correct
        adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx, data, splits = get_dataset(
            dataset=dataset_name,
            splits_dir=splits_dir
        )
        
        print(f"  ✓ Successfully loaded with get_dataset()")
        print(f"  ✓ Adjacency shape: {adj.shape}")
        print(f"  ✓ Features shape: {features.shape}")
        print(f"  ✓ Labels shape: {labels.shape}")
        print(f"  ✓ Splits loaded: train({splits['train']['edge'].shape}), "
              f"valid({splits['valid']['edge'].shape}), test({splits['test']['edge'].shape})")
        
        # Check that adj was created correctly
        if adj.is_sparse:
            print(f"  ✓ Adjacency is sparse tensor with {adj._nnz()} non-zero entries")
        else:
            print(f"  ✓ Adjacency is dense tensor")
        
        return True
        
    except Exception as e:
        print(f"  ✗ get_dataset() failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Generate/fix MORAL-compatible data files"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Specific dataset to generate (facebook, gplus, german, nba, pokec_n, pokec_z, credit)"
    )
    parser.add_argument(
        "--splits_dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory for split files"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if files exist"
    )
    parser.add_argument(
        "--no_fix",
        action="store_true",
        help="Don't try to fix existing files, regenerate instead"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test generated files with actual utils.py"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all datasets"
    )
    
    args = parser.parse_args()
    
    # Supported datasets
    datasets = ["facebook", "gplus", "german", "nba", "pokec_n", "pokec_z", "credit"]
    
    if args.test:
        # Test mode
        if args.dataset:
            test_with_real_utils(args.dataset, args.splits_dir)
        elif args.all:
            for dataset in datasets:
                test_with_real_utils(dataset, args.splits_dir)
        else:
            print("Please specify --dataset or --all with --test")
        return
    
    # Generate/fix mode
    if args.dataset:
        if args.dataset not in datasets:
            print(f"Error: Dataset '{args.dataset}' not supported.")
            print(f"Supported: {datasets}")
            return
        
        success, message = generate_dataset(
            args.dataset,
            args.splits_dir,
            args.force,
            not args.no_fix
        )
        
        print(f"\n{'='*60}")
        print(message)
        
        if success:
            # Test it
            test_with_real_utils(args.dataset, args.splits_dir)
    
    elif args.all:
        # Process all datasets
        print("Processing all MORAL datasets...")
        print("=" * 60)
        
        success_count = 0
        for dataset in datasets:
            success, message = generate_dataset(
                dataset,
                args.splits_dir,
                args.force,
                not args.no_fix
            )
            if success:
                success_count += 1
                # Test it
                test_with_real_utils(dataset, args.splits_dir)
            print("\n" + "=" * 60)
        
        print(f"\nSummary: {success_count}/{len(datasets)} datasets processed successfully")
        
        if success_count > 0:
            print(f"\nFiles saved to: {args.splits_dir}")
            print("\nRun MORAL with:")
            print(f"  python main.py --dataset facebook --splits_dir {args.splits_dir}")
    
    else:
        print("Please specify --dataset DATASET or --all")
        print(f"Available datasets: {datasets}")

if __name__ == "__main__":
    main()