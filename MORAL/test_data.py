"""
Test if generated data works with MORAL's get_dataset function
"""

import sys
from pathlib import Path

# Add utils to path
sys.path.append('.')

try:
    from utils import get_dataset
except ImportError:
    print("Error: Could not import get_dataset from utils.py")
    sys.exit(1)

def test_dataset(dataset_name: str, splits_dir: Path = Path("data/splits")):
    """Test if a dataset loads correctly"""
    print(f"\nTesting dataset: {dataset_name}")
    print("-" * 40)
    
    try:
        # This will use the exact same code path as MORAL
        adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx, data, splits = get_dataset(
            dataset=dataset_name,
            splits_dir=splits_dir
        )
        
        print(f"✓ Successfully loaded {dataset_name}")
        print(f"  Adjacency shape: {adj.shape}")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Sensitive attributes shape: {sens.shape}")
        print(f"  Train/Val/Test splits: {idx_train.shape[0]}/{idx_val.shape[0]}/{idx_test.shape[0]}")
        
        # Check splits
        print(f"  Edge splits - Train: {splits['train']['edge'].shape}")
        print(f"                 Valid: {splits['valid']['edge'].shape}")
        print(f"                 Test: {splits['test']['edge'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load {dataset_name}: {str(e)}")
        return False

def main():
    datasets = ["facebook", "gplus", "german", "nba", "pokec_n", "pokec_z", "credit"]
    splits_dir = Path("data/splits")
    
    if not splits_dir.exists():
        print(f"Error: Splits directory not found: {splits_dir}")
        print("Run generate_data.py first to create the data files.")
        return
    
    print("Testing MORAL data compatibility...")
    print("=" * 60)
    
    success_count = 0
    for dataset in datasets:
        success = test_dataset(dataset, splits_dir)
        if success:
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {success_count}/{len(datasets)} datasets loaded successfully")
    
    if success_count == len(datasets):
        print("\n✓ All datasets are ready for MORAL!")
        print("You can now run: python main.py --dataset [name]")
    else:
        print("\n⚠ Some datasets failed. Check the errors above.")

if __name__ == "__main__":
    main()