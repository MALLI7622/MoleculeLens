import torch
import torch.nn as nn
from transformers import GraphormerModel, GraphormerConfig


class GraphEncoderBaseline(nn.Module):
    """Graphormer encoder without prompt injection.

    Loads clefourrier/graphormer-base-pcqm4mv1 from HuggingFace and
    returns the virtual CLS graph token as the molecule representation.
    No prompt encoder required.

    Note on weight loading
    ----------------------
    The clefourrier checkpoint was saved from GraphormerForGraphClassification
    whose keys have an ``encoder.`` prefix (e.g. encoder.graph_encoder.layers.*).
    GraphormerModel itself has no such prefix.  We strip it manually so the
    pretrained backbone weights are actually used.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        config = GraphormerConfig.from_pretrained('clefourrier/graphormer-base-pcqm4mv1')
        self.graph_model = GraphormerModel(config)

        if pretrained:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id='clefourrier/graphormer-base-pcqm4mv1',
                filename='pytorch_model.bin'
            )
            raw_sd = torch.load(ckpt_path, map_location='cpu')
            # Strip 'encoder.' prefix; skip classifier / lm-head keys we don't need
            stripped = {}
            for k, v in raw_sd.items():
                if k.startswith('encoder.'):
                    stripped[k[len('encoder.'):]] = v
                # keys starting with 'classifier.' are intentionally dropped
            missing, unexpected = self.graph_model.load_state_dict(stripped, strict=False)
            if missing:
                print(f'[GraphEncoderBaseline] missing keys ({len(missing)}): '
                      f'{missing[:3]} …')
            if unexpected:
                print(f'[GraphEncoderBaseline] unexpected keys ({len(unexpected)}): '
                      f'{unexpected[:3]} …')

        self.graph_encoder = self.graph_model.graph_encoder

    def forward(self, x, attn_bias, attn_edge_type, spatial_pos,
                in_degree, out_degree, edge_input):
        """Return graph CLS token (virtual graph node at position 0).

        Args:
            x:              [B, N, node_feat_dim]
            attn_bias:      [B, N+1, N+1]
            attn_edge_type: [B, N, N, edge_feat_dim]
            spatial_pos:    [B, N, N]
            in_degree:      [B, N]
            out_degree:     [B, N]
            edge_input:     [B, N, N, max_dist, edge_feat_dim]

        Returns:
            graph_rep: [B, hidden_dim]
        """
        inner_states, graph_rep = self.graph_encoder(
            x, edge_input, attn_bias, in_degree, out_degree,
            spatial_pos, attn_edge_type, perturb=None
        )
        # graph_rep is already the virtual CLS node: shape [B, hidden_dim]
        return graph_rep
