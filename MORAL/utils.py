from __future__ import annotations

import psutil
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.utils import coalesce

from datasets import Facebook, Google, German, Nba, Pokec_n, Pokec_z, Credit


try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


def to_torch_sparse_tensor(
    edge_index: Tensor,
    edge_attr: Optional[Tensor] = None,
    size: Optional[Union[int, Tuple[int, int]]] = None,
) -> Tensor:
    """Convert an edge index representation into a sparse COO tensor."""

    if size is None:
        num_nodes = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
        size = (num_nodes, num_nodes)
    elif isinstance(size, int):
        size = (size, size)

    num_rows, num_cols = size
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_rows, num_cols)
    if edge_attr is None:
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)

    sparse = torch.sparse_coo_tensor(edge_index, edge_attr, size=size, device=edge_index.device)
    return sparse.coalesce()


def get_dataset(dataset: str, splits_dir: Union[str, Path]) -> Tuple:
    """Load dataset tensors together with pre-computed edge splits."""

    splits_path = Path(splits_dir) / f"{dataset}.pt"
    if not splits_path.exists():
        raise FileNotFoundError(
            f"Could not find edge splits for '{dataset}'. Expected file at '{splits_path}'."
        )

    data, splits = torch.load(splits_path)

    dataset_map: Dict[str, object] = {
        "facebook": Facebook,
        "gplus": Google,
        "german": German,
        "nba": Nba,
        "pokec_n": Pokec_n,
        "pokec_z": Pokec_z,
        "credit": Credit,
    }
    try:
        dataset_obj = dataset_map[dataset]()
    except KeyError as exc:
        raise ValueError(f"Unknown dataset '{dataset}'.") from exc

    adj = to_torch_sparse_tensor(splits["train"]["edge"].t())
    features = dataset_obj.features()
    idx_train = dataset_obj.idx_train()
    idx_val = dataset_obj.idx_val()
    idx_test = dataset_obj.idx_test()
    labels = dataset_obj.labels()
    sens = dataset_obj.sens()
    sens_idx = dataset_obj.sens_idx()

    return adj, features, idx_train, idx_val, idx_test, labels, sens, sens_idx, data, splits


def get_cpu_stats():
    proc = psutil.Process()

    cpu = proc.cpu_percent(interval=0.1) / psutil.cpu_count()
    mem_bytes = proc.memory_info().rss

    return {
        "cpu_percent": cpu,
        "ram_used_gb": mem_bytes / 1024**3,
    }


def get_gpu_stats():
    if not GPU_AVAILABLE:
        return None

    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)

    return {
        "gpu_util_percent": util.gpu,
        "gpu_mem_used_gb": mem.used / 1024**3,
        "gpu_mem_total_gb": mem.total / 1024**3,
    }


def log_system_usage(logger):
    cpu_mem = get_cpu_stats()
    gpu = get_gpu_stats()

    msg = (
        f"CPU: {cpu_mem['cpu_percent']:.1f}% | "
        f"RAM: {cpu_mem['ram_used_gb']:.2f}"
    )

    if gpu:
        msg += (
            f" | GPU: {gpu['gpu_util_percent']}% | "
            f"GPU MEM: {gpu['gpu_mem_used_gb']:.2f}/"
            f"{gpu['gpu_mem_total_gb']:.2f} GB"
        )

    logger.info(msg)
