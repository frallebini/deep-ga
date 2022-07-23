import json
import random
import torch
from abc import ABC, abstractmethod
from torch import nn, Tensor


class UncompressedNN(ABC, nn.Module):

    def __init__(self, seed: int = None) -> None:
        super().__init__()
        with open('config.json') as f:
            self.cfg = json.load(f)
        self.seeds = [seed] if seed else [self._generate_seed()]

    def mutate(self, seed: int) -> None:
        torch.manual_seed(seed)
        self.seeds.append(seed)
        params = nn.utils.parameters_to_vector(self.parameters())
        params += self.cfg['mutation_power'] * torch.randn(params.size())
        nn.utils.vector_to_parameters(params, self.parameters())

    def _generate_seed(self) -> int:
        return random.randint(0, 2**self.cfg['seed_bits']-1)

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def _define_layers(self) -> None:
        pass

    @abstractmethod
    def _init_params(self, module: nn.Module) -> None:
        pass
