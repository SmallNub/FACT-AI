from typing import Dict
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.nn import GATConv
from torch_geometric.utils import dropout_node, dropout_edge
from torch.amp import autocast, GradScaler

from moral import model_fit

NUM_GROUPS = 3


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, GATConv):
        # Feature projection
        nn.init.xavier_uniform_(m.lin.weight)

        # Attention parameters
        nn.init.xavier_uniform_(m.att_src)
        nn.init.xavier_uniform_(m.att_dst)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def standardize(tensor: Tensor) -> Tensor:
    return (tensor - tensor.mean(dim=0, keepdim=True)) / (
        tensor.std(dim=0, keepdim=True) + 1e-6
    )


def normalize(tensor: Tensor) -> Tensor:
    return F.normalize(tensor, p=2, dim=1)


class EdgeGroupDataset(Dataset):
    def __init__(self, edges: Tensor, labels: Tensor, groups: Tensor):
        self.edges = edges.long()
        self.labels = labels.float()
        self.groups = groups.long()

    def __len__(self):
        return self.edges.size(0)

    def __getitem__(self, idx):
        return self.edges[idx, 0], self.edges[idx, 1], self.labels[idx], self.groups[idx]

    def __getitems__(self, indices):
        return (
            self.edges[indices, 0],
            self.edges[indices, 1],
            self.labels[indices],
            self.groups[indices],
        )


class BottleneckBlock(nn.Module):
    def __init__(self, hidden_channels: int):
        super().__init__()
        bottleneck = hidden_channels // 4

        self.down = nn.Sequential(
            nn.Linear(hidden_channels, bottleneck),
            nn.BatchNorm1d(bottleneck),
            nn.ELU(),
        )

        self.up = nn.Sequential(
            nn.Linear(bottleneck, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.down(x))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 8, dropout: float = 0.3):
        super().__init__()

        self.conv = GATConv(
            in_channels,
            out_channels // heads,
            heads=heads,
            dropout=dropout,
            residual=True,
        )
        self.norm = gnn.BatchNorm(out_channels)
        self.act = nn.ELU()
        self.bottleneck = BottleneckBlock(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.skip_proj = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x_in = x

        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = self.act(x)
        x = x + self.bottleneck(x)
        x = self.dropout(x)

        if self.skip_proj is not None:
            x = x + self.skip_proj(x_in)
        else:
            x = x + x_in

        return x


class SharedBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 3,
        heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        layer_sizes = [in_channels] + [hidden_channels] * (num_layers - 1)

        for layer_size in layer_sizes:
            self.convs.append(
                ConvBlock(layer_size, hidden_channels, heads, dropout)
            )

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv in self.convs:
            x = conv(x, edge_index)

        x = standardize(x)
        x = normalize(x)
        return x


class GroupHeads(nn.Module):
    def __init__(self, hidden_channels: int, num_groups: int = NUM_GROUPS, dropout: float = 0.3):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.group_weight = nn.Parameter(
            torch.empty(num_groups, hidden_channels // 2, 1)
        )
        self.group_bias = nn.Parameter(torch.zeros(num_groups, 1))
        nn.init.xavier_uniform_(self.group_weight, gain=0.5)

    def forward(self, node_emb: Tensor, edges: Tensor, groups: Tensor) -> Tensor:
        src, dst = edges.t()
        x = torch.cat([node_emb[src], node_emb[dst]], dim=-1)
        x = self.shared(x)

        groups = groups.long()
        w = self.group_weight[groups]
        b = self.group_bias[groups]
        scores = torch.bmm(x.unsqueeze(1), w).squeeze(1) + b
        return scores.view(-1)


class EfficientMORAL(nn.Module):
    def __init__(
        self,
        adj: Tensor,
        features: Tensor,
        sens: Tensor,
        edge_splits: Dict[str, Dict[str, Tensor]],
        num_hidden: int = 128,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        batch_size: int = 1024,
        device: str = "cpu",
        patience: int = 10,
    ):
        super().__init__()

        self.device = torch.device(device)
        self.batch_size = batch_size
        self.patience = patience
        self.accum_steps = 1

        features = standardize(features.float())
        self.features = features.to(self.device)

        self.edge_index = adj.coalesce().indices().to(self.device)
        self.sens = sens

        self.backbone = SharedBackbone(
            in_channels=features.size(1),
            hidden_channels=num_hidden,
        ).to(self.device)

        self.heads = GroupHeads(num_hidden).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.heads.parameters()),
            lr=lr,
            weight_decay=weight_decay,
            fused=True,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=patience,
            threshold_mode="rel",
            threshold=1e-3,
        )
        self.amp_enabled = torch.get_float32_matmul_precision() == "medium"
        self.scaler = GradScaler(enabled=self.amp_enabled)

        self.train_loader = self._build_loader(edge_splits.get("train"), True)
        self.valid_loader = self._build_loader(edge_splits.get("valid"), False)
        self.test_data = self._prepare_edges(edge_splits.get("test"))

        self.apply(init_weights)
        self.compile()

    def _prepare_edges(self, split):
        if split is None:
            return None

        pos = split["edge"]
        neg = split["edge_neg"]

        edges = torch.cat([pos, neg], dim=0)
        labels = torch.cat([torch.ones(pos.size(0)), torch.zeros(neg.size(0))], dim=0)
        groups = self.sens[edges].sum(dim=1)
        return edges, labels, groups

    def _build_loader(self, split, shuffle):
        if split is None:
            return None

        edges, labels, groups = self._prepare_edges(split)
        dataset = EdgeGroupDataset(edges, labels, groups)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
        )

    def _train_epoch(self):
        self.backbone.train()
        self.heads.train()

        total_loss = 0.0
        total_batches = 0
        loss_accum = 0.0
        step_count = 0

        for edges1, edges2, labels, groups in self.train_loader:
            edges = torch.stack((edges1, edges2), dim=1).long()
            edges = edges.to(self.device)
            labels = labels.to(self.device)
            groups = groups.to(self.device)

            if step_count == 0:
                edge_index, _, _ = dropout_node(self.edge_index, p=0.1, training=self.training)
                edge_index, _ = dropout_edge(edge_index, p=0.2, training=self.training)

                with autocast(device_type=self.device.type, enabled=self.amp_enabled, dtype=torch.bfloat16):
                    node_emb = self.backbone(self.features, edge_index)

            with autocast(device_type=self.device.type, enabled=self.amp_enabled, dtype=torch.bfloat16):
                scores = self.heads(node_emb, edges, groups)
                loss = self.criterion(scores, labels)

            loss_accum = loss_accum + loss
            total_loss += loss.item()
            total_batches += 1
            step_count += 1

            if step_count >= self.accum_steps:
                loss_accum = loss_accum / step_count
                self.optimizer.zero_grad()
                self.scaler.scale(loss_accum).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                step_count = 0
                loss_accum = 0.0

        if step_count > 0:
            loss_accum = loss_accum / step_count
            self.optimizer.zero_grad()
            self.scaler.scale(loss_accum).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        return total_loss / total_batches

    @torch.no_grad()
    def _evaluate(self):
        if self.valid_loader is None:
            return None

        self.backbone.eval()
        self.heads.eval()

        with autocast(device_type=self.device.type, enabled=self.amp_enabled, dtype=torch.bfloat16):
            node_emb = self.backbone(self.features, self.edge_index)

        total_loss = 0.0
        total_batches = 0

        for edges1, edges2, labels, groups in self.valid_loader:
            edges = torch.stack((edges1, edges2), dim=1).long()
            edges = edges.to(self.device)
            labels = labels.to(self.device)
            groups = groups.to(self.device)

            with autocast(device_type=self.device.type, enabled=self.amp_enabled, dtype=torch.bfloat16):
                scores = self.heads(node_emb, edges, groups)
                loss = self.criterion(scores, labels)

            total_loss += loss.item()
            total_batches += 1

        return total_loss / total_batches

    def fit(self, epochs=300):
        model_fit(self, epochs)

    @torch.no_grad()
    def predict(self):
        self.backbone.eval()
        self.heads.eval()

        edges, _, groups = self.test_data
        with autocast(device_type=self.device.type, enabled=self.amp_enabled, dtype=torch.bfloat16):
            node_emb = self.backbone(self.features, self.edge_index)

        scores = []
        for i in range(0, edges.size(0), self.batch_size):
            batch_edges = edges[i : i + self.batch_size].to(self.device)
            batch_groups = groups[i : i + self.batch_size].to(self.device)

            with autocast(device_type=self.device.type, enabled=self.amp_enabled, dtype=torch.bfloat16):
                score = self.heads(node_emb, batch_edges, batch_groups)

            scores.append(score)

        return torch.cat(scores, dim=0)

    @staticmethod
    def fair_metric(pred, labels, sens):
        idx_s0 = sens == 0
        idx_s1 = sens == 1
        idx_s0_y1 = idx_s0 & (labels == 1)
        idx_s1_y1 = idx_s1 & (labels == 1)

        parity = (
            (pred[idx_s0].mean() - pred[idx_s1].mean()).abs()
            if idx_s0.any() and idx_s1.any()
            else torch.tensor(0.0)
        )
        equality = (
            (pred[idx_s0_y1].mean() - pred[idx_s1_y1].mean()).abs()
            if idx_s0_y1.any() and idx_s1_y1.any()
            else torch.tensor(0.0)
        )

        return float(parity), float(equality)
