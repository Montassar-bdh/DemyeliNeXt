import monai
import torch
import torch.nn as nn
import torch.nn.functional as F
from easyfsl.methods import FewShotClassifier, PrototypicalNetworks


class FewShotDenseNet(monai.networks.nets.DenseNet121):
    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        dropout_prob=0.2,
        use_fc: bool = True,
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_prob=dropout_prob,
        )
        self.use_fc = use_fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.class_layers[:-1](x)
        if self.use_fc:
            x = self.class_layers[-1](x)
        return x

    def set_use_fc(self, use_fc: bool):
        self.use_fc = use_fc
