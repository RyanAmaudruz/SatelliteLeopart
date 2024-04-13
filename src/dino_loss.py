import torch
from torch.nn.init import trunc_normal_


class DINOHead(torch.nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = torch.nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [torch.nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(torch.nn.BatchNorm1d(hidden_dim))
            layers.append(torch.nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(torch.nn.BatchNorm1d(hidden_dim))
                layers.append(torch.nn.GELU())
            layers.append(torch.nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = torch.nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = torch.nn.utils.weight_norm(torch.nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = torch.nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
