import json
import random
from typing import List


class CompressedNN:

    def __init__(self, seeds: List[int] = None) -> None:
        with open('config.json') as f:
            self.bits = json.load(f)['seed_bits']
        self.seeds = seeds if seeds else [self._generate_seed()]

    def __repr__(self) -> str:
        return f'CompressedNN({self.seeds})'

    def mutate(self, seed: int = None) -> None:
        self.seeds.append(seed if seed else self._generate_seed())

    def _generate_seed(self) -> int:
        return random.randint(0, 2**self.bits-1)
