import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
from loguru import logger
from utils import log_system_usage

class SharedGNNBackbone(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels, hidden_channels))
        
    def forward(self, x, edge_index, edge_weight=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index, edge_weight).relu()
        x = self.convs[-1](x, edge_index, edge_weight)
        return x

class GroupSpecificHeads(nn.Module):
    def __init__(self, hidden_channels, num_heads=3):
        super().__init__()
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            head = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1)
            )
            self.heads.append(head)
    
    def forward(self, node_embeddings, edges, groups):
        if len(edges) == 0:
            return torch.tensor([], device=node_embeddings.device)
        
        u, v = edges[:, 0], edges[:, 1]
        edge_embs = torch.cat([node_embeddings[u], node_embeddings[v]], dim=-1)
        
        scores = []
        for group in range(3):
            mask = (groups == group)
            if mask.any():
                group_edges = edge_embs[mask]
                group_scores = self.heads[group](group_edges)
                scores.append(group_scores.squeeze())
        
        return torch.cat(scores, dim=0) if scores else torch.tensor([], device=node_embeddings.device)

class EfficientMORAL(nn.Module):
    def __init__(
        self,
        adj,
        features,
        labels,
        idx_train,
        idx_val,
        idx_test,
        sens,
        sens_idx,
        edge_splits,
        dataset_name,
        num_hidden=128,
        lr=0.001,
        weight_decay=0.0,
        encoder="gcn",
        decoder="gae",
        batch_size=1024,
        device="cpu"
    ):
        super().__init__()
        
        self.device = torch.device(device)
        self.sens = sens.to(self.device)
        self.features = features.to(self.device)
        self.adj = adj.to(self.device)
        self.edge_splits = edge_splits
        self.batch_size = batch_size
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.backbone = SharedGNNBackbone(
            in_channels=features.shape[1],
            hidden_channels=num_hidden
        ).to(self.device)
        
        self.heads = GroupSpecificHeads(
            hidden_channels=num_hidden,
            num_heads=3
        ).to(self.device)
        
        params = list(self.backbone.parameters()) + list(self.heads.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.labels = labels
        self.sens_idx = sens_idx
        self.dataset_name = dataset_name
        self.best_state = None
        
    def get_edges_for_group(self, group, split="train"):
        edges = self.edge_splits[split]["edge"]
        if len(edges) == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)
        
        edge_sens = self.sens[edges].sum(dim=1)
        group_mask = (edge_sens == group)
        return edges[group_mask]
    
    def get_negative_edges_for_group(self, group, num_needed, split="train"):
        neg_edges = self.edge_splits[split]["edge_neg"]
        if len(neg_edges) == 0 or num_needed == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)
        
        neg_edge_sens = self.sens[neg_edges].sum(dim=1)
        neg_group_mask = (neg_edge_sens == group)
        group_neg_edges = neg_edges[neg_group_mask]
        
        if len(group_neg_edges) > num_needed:
            indices = torch.randperm(len(group_neg_edges))[:num_needed]
            return group_neg_edges[indices]
        return group_neg_edges
    
    def _train_epoch(self):
        self.backbone.train()
        self.heads.train()
        
        total_loss = 0.0
        total_batches = 0
        
        for group in [0, 1, 2]:
            pos_edges = self.get_edges_for_group(group, "train")
            if len(pos_edges) == 0:
                continue
            
            neg_edges = self.get_negative_edges_for_group(group, len(pos_edges), "train")
            all_edges = torch.cat([pos_edges, neg_edges], dim=0)
            labels = torch.cat([
                torch.ones(len(pos_edges), device=self.device),
                torch.zeros(len(neg_edges), device=self.device)
            ])
            groups = torch.full((len(all_edges),), group, device=self.device, dtype=torch.long)
            
            edge_index = self.adj.coalesce().indices()
            node_embeddings = self.backbone(self.features, edge_index)
            scores = self.heads(node_embeddings, all_edges, groups)
            
            if len(scores) > 0:
                loss = self.criterion(scores, labels)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += float(loss.item())
                total_batches += 1
        
        return total_loss / max(total_batches, 1)
    
    def fit(self, epochs=300):
        best_val = float("inf")
        best_state = None
        
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch()
            val_loss = self._evaluate()
            
            if val_loss is not None and val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self.state_dict().items()}
            
            if epoch % 10 == 0 or epoch == 1:
                if val_loss is None:
                    logger.info(f"Epoch {epoch} | train={train_loss:.4f}")
                else:
                    logger.info(f"Epoch {epoch} | train={train_loss:.4f} | valid={val_loss:.4f}")
                
                log_system_usage(logger)
        
        if best_state is not None:
            self.load_state_dict(best_state)
            self.best_state = best_state
    
    @torch.no_grad()
    def _evaluate(self, split="valid"):
        self.backbone.eval()
        self.heads.eval()
        
        total_loss = 0.0
        total_batches = 0
        
        for group in [0, 1, 2]:
            pos_edges = self.get_edges_for_group(group, split)
            if len(pos_edges) == 0:
                continue
            
            neg_edges = self.get_negative_edges_for_group(group, len(pos_edges), split)
            all_edges = torch.cat([pos_edges, neg_edges], dim=0)
            labels = torch.cat([
                torch.ones(len(pos_edges), device=self.device),
                torch.zeros(len(neg_edges), device=self.device)
            ])
            groups = torch.full((len(all_edges),), group, device=self.device, dtype=torch.long)
            
            edge_index = self.adj.coalesce().indices()
            node_embeddings = self.backbone(self.features, edge_index)
            scores = self.heads(node_embeddings, all_edges, groups)
            
            if len(scores) > 0:
                loss = self.criterion(scores, labels)
                total_loss += float(loss.item())
                total_batches += 1
        
        if total_batches == 0:
            return None
        return total_loss / total_batches
    
    @torch.no_grad()
    def predict(self):
        return self._predict("test")
    
    def _predict(self, split="test"):
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