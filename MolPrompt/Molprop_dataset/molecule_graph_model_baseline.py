import torch
import torch.nn as nn

from model.graphormer_baseline import GraphEncoderBaseline


class Graph_pred_baseline(nn.Module):
    """Graphormer-base fine-tuning baseline.

    Uses clefourrier/graphormer-base-pcqm4mv1 (pre-trained on PCQM4Mv1)
    as the graph encoder with a linear prediction head.  No prompt encoder,
    no KV-PLM checkpoint, no contrastive pre-training required.
    """

    def __init__(
        self,
        graph_hidden_dim: int = 768,
        num_tasks: int = 1,
        drop_ratio: float = 0.1,
        device_id: int = 0,
    ):
        super().__init__()
        self.graph_hidden_dim = graph_hidden_dim
        self.device_str = f'cuda:{device_id}'

        self.graph_encoder = GraphEncoderBaseline(pretrained=True)
        self.dropout = nn.Dropout(drop_ratio)
        self.graph_pred_linear = nn.Linear(graph_hidden_dim, num_tasks)

    def forward(self, batch):
        device = torch.device(self.device_str)

        # Move graph tensors to device
        x             = batch.x.to(device)
        attn_bias     = batch.attn_bias.to(device)
        attn_edge_type = batch.attn_edge_type.to(device)
        spatial_pos   = batch.spatial_pos.to(device)
        in_degree     = batch.in_degree.to(device)
        out_degree    = batch.out_degree.to(device)
        edge_input    = batch.edge_input.to(device)

        # [B, hidden_dim]
        graph_feature = self.graph_encoder(
            x, attn_bias, attn_edge_type, spatial_pos,
            in_degree, out_degree, edge_input
        )
        graph_feature = self.dropout(graph_feature)
        output = self.graph_pred_linear(graph_feature)
        return graph_feature, output
