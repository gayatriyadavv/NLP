"""
PyTorch Geometric Graph Neural Network (GNN) Message-Passing Architecture.
Supports GATConv, GCNConv, and SAGEConv variants for heterogeneous scientific entity grounding.
Produces learned node embeddings and graph-level embeddings for retrieval & grounding tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool

class GNNGroundingModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 64,
        out_channels: int = 64,
        gnn_type: str = "GAT",
        num_layers: int = 2,
        dropout: float = 0.1,
        heads: int = 2
    ):
        super(GNNGroundingModel, self).__init__()
        self.gnn_type = gnn_type
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        # Input feature projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)

        # Message Passing Convolutional Layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_channels if i == 0 else hidden_channels
            
            if gnn_type == "GAT":
                # Multi-head Graph Attention Network
                out_dim = hidden_channels // heads if heads > 1 else hidden_channels
                conv = GATConv(in_dim, out_dim, heads=heads, concat=(heads > 1), dropout=dropout)
            elif gnn_type == "SAGE":
                # GraphSAGE Neighborhood Aggregation
                conv = SAGEConv(in_dim, hidden_channels)
            else:
                # Default GCN Conv
                conv = GCNConv(in_dim, hidden_channels)

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Output Node Embedding Projection
        self.out_proj = nn.Linear(hidden_channels, out_channels)

        # Link Prediction Bilinear Classifier
        self.link_score = nn.Bilinear(out_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass producing learned node embeddings Z.
        Input:
            x: Node feature matrix [N, in_channels]
            edge_index: Edge indices [2, E]
            batch: Batch index tensor [N]
        Returns:
            z: Learned node embeddings [N, out_channels]
        """
        # Project initial node features
        h = F.relu(self.input_proj(x))
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Message-passing layers
        for conv, norm in zip(self.convs, self.norms):
            if edge_index.numel() == 0:
                # Fallback for graphs with no edges
                h = norm(h)
            else:
                h_conv = conv(h, edge_index)
                h = norm(F.relu(h_conv) + h)  # Residual connection
            h = F.dropout(h, p=self.dropout, training=self.training)

        z = self.out_proj(h)
        # Normalize learned embeddings to unit sphere
        z = F.normalize(z, p=2, dim=-1)
        return z

    def get_graph_embedding(self, z: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Computes graph-level embedding via mean + max pooling over node embeddings.
        """
        if batch is None:
            batch = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        mean_pool = global_mean_pool(z, batch)
        max_pool = global_max_pool(z, batch)
        graph_emb = F.normalize(mean_pool + max_pool, p=2, dim=-1)
        return graph_emb

    def predict_link(self, z_src: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
        """
        Predicts edge connection probability between source and target node embeddings.
        """
        scores = torch.sigmoid(self.link_score(z_src, z_tgt)).squeeze(-1)
        return scores


class PairGroundingModel(nn.Module):
    """
    Supervised element <-> entity grounding scorer.

    Node encoder: ``GNNGroundingModel`` (message passing over the grounding graph).
    Pair head:    MLP over [z_u, z_v, z_u * z_v, pairwise_features] -> logit.

    With ``use_message_passing=False`` the encoder is run with an empty edge set, so each node
    embedding depends only on its own features. This is the no-message-passing diagnostic used
    to measure what the graph structure itself contributes.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 64,
        pair_feature_dim: int = 10,
        gnn_type: str = "GAT",
        num_layers: int = 2,
        dropout: float = 0.1,
        heads: int = 2,
        use_message_passing: bool = True,
    ):
        super(PairGroundingModel, self).__init__()
        self.use_message_passing = use_message_passing
        self.encoder = GNNGroundingModel(in_channels, hidden_channels, hidden_channels,
                                         gnn_type=gnn_type, num_layers=num_layers,
                                         dropout=dropout, heads=heads)
        self.head = nn.Sequential(
            nn.Linear(3 * hidden_channels + pair_feature_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, cand_index: torch.Tensor,
                cand_feat: torch.Tensor) -> torch.Tensor:
        if not self.use_message_passing:
            edge_index = edge_index[:, :0]
        z = self.encoder(x, edge_index)
        zu, zv = z[cand_index[0]], z[cand_index[1]]
        return self.head(torch.cat([zu, zv, zu * zv, cand_feat], dim=-1)).squeeze(-1)
