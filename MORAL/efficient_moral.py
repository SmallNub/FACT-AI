from typing import Dict, Optional
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm

from moral import model_fit

NUM_GROUPS = 3


class EdgeGroupDataset(Dataset):
    def __init__(self, edges: Tensor, labels: Tensor, groups: Tensor):
        self.edges = edges.long()
        self.labels = labels.float()
        self.groups = groups.long()

    def __len__(self):
        return self.edges.size(0)

    def __getitem__(self, idx):
        return self.edges[idx], self.labels[idx], self.groups[idx]


class SharedBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.residual = True

        # First layer
        self.convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads, dropout=dropout))
        self.norms.append(BatchNorm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, dropout=dropout))
            self.norms.append(BatchNorm(hidden_channels))

        # Final layer
        self.convs.append(GATConv(hidden_channels, hidden_channels, heads=1, concat=False, dropout=dropout))
        self.skip_proj = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else None

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x_initial = x
        for i, conv in enumerate(self.convs[:-1]):
            x_in = x
            x = conv(x, edge_index)
            if i < len(self.norms):
                x = self.norms[i](x)
            x = F.elu(x)
            x = self.dropout(x)
            if self.residual:
                if x.shape == x_in.shape:
                    x = x + x_in
                elif i == 0 and self.skip_proj is not None:
                    x = x + self.skip_proj(x_initial)
        x = self.convs[-1](x, edge_index)
        return x


class GroupHeads(nn.Module):
    def __init__(self, hidden_channels: int, num_groups: int = NUM_GROUPS):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
        )

        self.group_weight = nn.Parameter(
            torch.empty(num_groups, hidden_channels // 2, 1)
        )
        self.group_bias = nn.Parameter(torch.zeros(num_groups, 1))
        nn.init.xavier_uniform_(self.group_weight)

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

        self.features = features.to(self.device)
        self.edge_index = adj.coalesce().indices().to(self.device)
        self.sens = sens.to(self.device)

        self.backbone = SharedBackbone(
            in_channels=features.size(1),
            hidden_channels=num_hidden,
        ).to(self.device)

        self.heads = GroupHeads(num_hidden).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(
            list(self.backbone.parameters()) + list(self.heads.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=patience
        )

        self.train_loader = self._build_loader(edge_splits.get("train"), True)
        self.valid_loader = self._build_loader(edge_splits.get("valid"), False)
        self.test_data = self._prepare_edges(edge_splits.get("test"))

    def _prepare_edges(self, split):
        if split is None:
            return None

        pos = split["edge"]
        neg = split["edge_neg"]

        edges = torch.cat([pos, neg], dim=0)
        labels = torch.cat(
            [torch.ones(pos.size(0)), torch.zeros(neg.size(0))], dim=0
        )
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

        node_emb = self.backbone(self.features, self.edge_index)

        total_loss = 0.0
        total_batches = 0
        loss_accum = 0.0

        for edges, labels, groups in self.train_loader:
            edges = edges.to(self.device)
            labels = labels.to(self.device)
            groups = groups.to(self.device)

            scores = self.heads(node_emb, edges, groups)
            loss = self.criterion(scores, labels)

            loss_accum = loss_accum + loss
            total_loss += loss.item()
            total_batches += 1

        self.optimizer.zero_grad()
        loss_accum.backward()
        self.optimizer.step()

        return total_loss / total_batches


    @torch.no_grad()
    def _evaluate(self):
        if self.valid_loader is None:
            return None

        self.backbone.eval()
        self.heads.eval()

        node_emb = self.backbone(self.features, self.edge_index)

        total_loss = 0.0
        total_batches = 0

        for edges, labels, groups in self.valid_loader:
            edges = edges.to(self.device)
            labels = labels.to(self.device)
            groups = groups.to(self.device)

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
        node_emb = self.backbone(self.features, self.edge_index)

        scores = []
        for i in range(0, edges.size(0), self.batch_size):
            batch_edges = edges[i : i + self.batch_size].to(self.device)
            batch_groups = groups[i : i + self.batch_size].to(self.device)
            scores.append(self.heads(node_emb, batch_edges, batch_groups))

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
