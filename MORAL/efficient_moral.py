from typing import Dict, List, Optional, Sequence, Tuple
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm
from torch.utils.data import Dataset, DataLoader
from moral import model_fit
from loguru import logger
from utils import log_system_usage


class EdgeLabelDataset(Dataset):
    """PyTorch dataset returning edge indices and binary labels."""

    def __init__(self, edges: Tensor, labels: Tensor) -> None:
        self.edges = edges.long()
        self.labels = labels.float()

    def __len__(self) -> int:
        return int(self.edges.size(0))

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.edges[idx], self.labels[idx]


class SharedGNNBackbone(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers=2,
        heads=8,
        dropout=0.2,
        residual=True,
        norm="batch",
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if norm else None
        self.dropout = nn.Dropout(dropout)
        self.residual = residual

        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels // heads,
                heads=heads,
                dropout=dropout,
                add_self_loops=True,
                edge_dim=None,
            )
        )

        if norm == "batch":
            self.norms.append(BatchNorm(hidden_channels))
        elif norm == "layer":
            self.norms.append(nn.LayerNorm(hidden_channels))

        for i in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_channels,
                    hidden_channels // heads,
                    heads=heads,
                    dropout=dropout,
                    add_self_loops=True,
                    edge_dim=None,
                )
            )
            if self.norms is not None:
                if norm == "batch":
                    self.norms.append(BatchNorm(hidden_channels))
                elif norm == "layer":
                    self.norms.append(nn.LayerNorm(hidden_channels))

        self.convs.append(
            GATConv(
                hidden_channels,
                hidden_channels,
                heads=1,
                concat=False,
                dropout=dropout,
                add_self_loops=True,
                edge_dim=None,
            )
        )

        self.skip_proj = (
            nn.Linear(in_channels, hidden_channels)
            if in_channels != hidden_channels
            else None
        )

    def forward(self, x, edge_index, edge_weight=None):
        x_initial = x

        for i, conv in enumerate(self.convs[:-1]):
            x_in = x
            x = conv(x, edge_index, edge_weight)

            if self.norms and i < len(self.norms):
                x = self.norms[i](x)

            x = F.elu(x)
            x = self.dropout(x)

            if self.residual:
                if x.shape == x_in.shape:
                    x = x + x_in
                elif i == 0 and self.skip_proj is not None:
                    x = self.skip_proj(x_initial) + x

        x = self.convs[-1](x, edge_index, edge_weight)
        return x


class GroupSpecificHeads(nn.Module):
    def __init__(self, hidden_channels, num_heads=3, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList()

        for _ in range(num_heads):
            head = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.GELU(),
                nn.Linear(hidden_channels // 2, 1),
            )
            self.heads.append(head)

    def forward(self, node_embeddings, edges, groups):
        if len(edges) == 0:
            return torch.tensor([], device=node_embeddings.device)

        # edges should already be [batch_size, 2]
        u, v = edges[:, 0], edges[:, 1]
        edge_embs = torch.cat([node_embeddings[u], node_embeddings[v]], dim=-1)

        scores = []
        for group in range(3):
            mask = groups == group
            if mask.any():
                group_edges = edge_embs[mask]
                group_scores = self.heads[group](group_edges)
                scores.append(group_scores.squeeze(-1))  # squeeze last dimension

        return (
            torch.cat(scores, dim=0)
            if scores
            else torch.tensor([], device=node_embeddings.device)
        )


class EfficientMORAL(nn.Module):
    def __init__(
        self,
        adj: Tensor,
        features: Tensor,
        labels: Tensor,
        idx_train: Tensor,
        idx_val: Tensor,
        idx_test: Tensor,
        sens: Tensor,
        sens_idx: Tensor,
        edge_splits: Dict[str, Dict[str, Tensor]],
        dataset_name: str,
        num_hidden: int = 128,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        encoder: str = "gcn",
        decoder: str = "gae",
        batch_size: int = 1024,
        device: str = "cpu",
        patience: int = 10,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.criterion = nn.BCEWithLogitsLoss()

        self.features = features.to(self.device)
        self.adj = adj.to(self.device)
        self.edge_index = adj.coalesce().indices().to(self.device)
        self.labels = labels.to(self.device)
        self.sens = sens.to(self.device)
        self.sens_idx = sens_idx
        self.sens_cpu = sens.cpu()
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.edge_splits = edge_splits
        self.patience = patience

        self.backbone = SharedGNNBackbone(
            in_channels=features.shape[1], hidden_channels=num_hidden
        ).to(self.device)

        self.heads = GroupSpecificHeads(hidden_channels=num_hidden, num_heads=3).to(
            self.device
        )

        params = list(self.backbone.parameters()) + list(self.heads.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=patience,
        )

        self.train_loaders = self._build_group_loaders(edge_splits.get("train"), shuffle=True)
        self.valid_loaders = self._build_group_loaders(edge_splits.get("valid"), shuffle=False)

        test_edges, test_labels = self._prepare_edges(edge_splits.get("test"))
        self.test_edges = test_edges
        self.test_labels = test_labels

        self.best_state = None

    def _prepare_edges(self, split: Optional[Dict[str, Tensor]]) -> Tuple[Tensor, Tensor]:
        if split is None:
            return torch.empty(0, 2, dtype=torch.long), torch.empty(0)

        pos_edges = split["edge"].long()
        neg_edges = split["edge_neg"].long()
        edges = torch.cat([pos_edges, neg_edges], dim=0)
        labels = torch.cat(
            [torch.ones(pos_edges.size(0)), torch.zeros(neg_edges.size(0))], dim=0
        )
        return edges, labels

    def _build_group_loaders(self, split: Optional[Dict[str, Tensor]], shuffle: bool) -> List[Optional[DataLoader]]:
        if split is None:
            return [None] * 3

        edges, labels = self._prepare_edges(split)
        sens_groups = self.sens_cpu[edges].sum(dim=1)
        loaders: List[Optional[DataLoader]] = []
        
        for group in range(3):
            mask = sens_groups == group
            if mask.sum() == 0:
                loaders.append(None)
                continue

            dataset = EdgeLabelDataset(edges[mask], labels[mask])
            batch = len(dataset) if self.batch_size <= 0 else min(self.batch_size, len(dataset))

            loader = DataLoader(
                dataset,
                batch_size=batch,
                shuffle=shuffle,
                drop_last=False,
                pin_memory=True,
            )
            loaders.append(loader)
        
        return loaders

    def _train_epoch(self):
        self.backbone.train()
        self.heads.train()

        total_loss = 0.0
        total_batches = 0

        for group, loader in enumerate(self.train_loaders):
            if loader is None:
                continue

            for edges, labels in loader:
                self.optimizer.zero_grad()
                
                edges = edges.to(self.device)
                labels = labels.to(self.device)
                groups = torch.full((edges.size(0),), group, device=self.device, dtype=torch.long)

                node_embeddings = self.backbone(self.features, self.edge_index)
                scores = self.heads(node_embeddings, edges, groups)
                
                loss = self.criterion(scores, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += float(loss.item())
                total_batches += 1

        return total_loss / max(total_batches, 1)

    @torch.no_grad()
    def _evaluate(self, loaders: Sequence[Optional[DataLoader]]) -> Optional[float]:
        self.backbone.eval()
        self.heads.eval()

        total_loss = 0.0
        total_batches = 0
        
        # Compute node embeddings once for the entire evaluation
        node_embeddings = self.backbone(self.features, self.edge_index)

        for group, loader in enumerate(loaders):
            if loader is None:
                continue
            
            for edges, labels in loader:
                edges = edges.to(self.device)
                labels = labels.to(self.device)
                groups = torch.full((edges.size(0),), group, device=self.device, dtype=torch.long)

                scores = self.heads(node_embeddings, edges, groups)
                loss = self.criterion(scores, labels)
                
                total_loss += float(loss.item())
                total_batches += 1

        if total_batches == 0:
            return None
        return total_loss / total_batches

    def fit(self, epochs=300):
        """Train the model with early stopping and learning rate scheduling."""
        best_val = float("inf")
        best_state = None
        bad_epochs = 0

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch()
            val_loss = self._evaluate(self.valid_loaders)

            if epoch % 10 == 0 or epoch == 1:
                lr = self.optimizer.param_groups[0]["lr"]
                if val_loss is None:
                    logger.info(f"Epoch {epoch} | train={train_loss:.4f} | lr={lr:.6f}")
                else:
                    logger.info(f"Epoch {epoch} | train={train_loss:.4f} | valid={val_loss:.4f} | lr={lr:.6f}")
                log_system_usage(logger)

            if val_loss is not None:
                self.scheduler.step(val_loss)
                
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in self.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.patience * 3:
                        break

        if best_state is not None:
            self.load_state_dict(best_state)
            self.best_state = best_state

    @torch.no_grad()
    def predict(self, split="test"):
        self.backbone.eval()
        self.heads.eval()
        
        edge_index = self.adj.coalesce().indices()
        with torch.no_grad():
            node_emb = self.backbone(self.features, edge_index)
        
        pos_edges = self.edge_splits[split]["edge"]
        neg_edges = self.edge_splits[split]["edge_neg"]
        all_edges = torch.cat([pos_edges, neg_edges], dim=0).to(self.device)
        
        if len(all_edges) == 0:
            return torch.tensor([], device=self.device)
        
        edge_sens = self.sens[all_edges].sum(dim=1)
        
        scores = []
        batch_size = self.batch_size
        
        for i in range(0, len(all_edges), batch_size):
            batch_edges = all_edges[i:i+batch_size]
            batch_groups = edge_sens[i:i+batch_size]
            
            batch_scores = self.heads(node_emb, batch_edges, batch_groups)
            scores.append(batch_scores)
        
        if scores:
            return torch.cat(scores, dim=0)
        else:
            return torch.tensor([], device=self.device)

    def predict_by_group(self, split="test"):
        """Alternative: Predict using group-separated DataLoaders (like training)."""
        self.backbone.eval()
        self.heads.eval()

        # Create group loaders for prediction
        loaders = self._build_group_loaders(self.edge_splits.get(split), shuffle=False)
        
        all_scores = []
        all_edges = []
        group_indices = []
        
        node_embeddings = self.backbone(self.features, self.edge_index)

        for group, loader in enumerate(loaders):
            if loader is None:
                continue

            for edges, _ in loader:
                edges = edges.to(self.device)
                groups = torch.full((edges.size(0),), group, device=self.device, dtype=torch.long)
                
                batch_scores = self.heads(node_embeddings, edges, groups)
                
                all_scores.append(batch_scores)
                all_edges.append(edges)
                group_indices.append(torch.full((edges.size(0),), group, device=self.device))

        if all_scores:
            scores = torch.cat(all_scores, dim=0)
            edges = torch.cat(all_edges, dim=0)
            groups = torch.cat(group_indices, dim=0)
            
            return scores, edges, groups
        else:
            return torch.tensor([], device=self.device), torch.tensor([], device=self.device), torch.tensor([], device=self.device)

    @staticmethod
    def fair_metric(pred, labels, sens):
        idx_s0 = sens == 0
        idx_s1 = sens == 1
        idx_s0_y1 = torch.logical_and(idx_s0, labels == 1)
        idx_s1_y1 = torch.logical_and(idx_s1, labels == 1)

        parity = torch.tensor(0.0)
        equality = torch.tensor(0.0)
        if idx_s0.sum() > 0 and idx_s1.sum() > 0:
            parity = (pred[idx_s0].mean() - pred[idx_s1].mean()).abs()
        if idx_s0_y1.sum() > 0 and idx_s1_y1.sum() > 0:
            equality = (pred[idx_s0_y1].mean() - pred[idx_s1_y1].mean()).abs()

        return float(parity.item()), float(equality.item())