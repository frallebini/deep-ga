import torch
import torch.nn.functional as F
from torch import nn, Tensor

from uncompressed import UncompressedNN


class ConvNN(UncompressedNN):

    def __init__(self, seed: int = None) -> None:
        super().__init__(seed)
        self._define_layers()
        self.requires_grad_(False)
        self.apply(self._init_params)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.lin1(x))
        return self.lin2(x)

    def _define_layers(self) -> None:
        cfg = self.cfg
        self.conv1 = nn.Conv2d(
            cfg['stacked_frames'],
            cfg['conv1']['out_channels'],
            cfg['conv1']['kernel_size'],
            cfg['conv1']['stride'])
        self.conv2 = nn.Conv2d(
            cfg['conv1']['out_channels'],
            cfg['conv2']['out_channels'],
            cfg['conv2']['kernel_size'],
            cfg['conv2']['stride'])
        self.conv3 = nn.Conv2d(
            cfg['conv2']['out_channels'],
            cfg['conv3']['out_channels'],
            cfg['conv3']['kernel_size'],
            cfg['conv3']['stride'])
        self.lin1 = nn.Linear(
            cfg['lin1']['in_features'],  # assumes input to conv1 is 84x84
            cfg['lin1']['out_features'])
        self.lin2 = nn.Linear(
            cfg['lin1']['out_features'],
            cfg['actions'])

    def _init_params(self, module: nn.Module) -> None:
        torch.manual_seed(self.seeds[0])
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(module.bias)
