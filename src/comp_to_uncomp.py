import json
import torch

from compressed import CompressedNN
from conv import ConvNN
from uncompressed import UncompressedNN


def compress(model: UncompressedNN) -> CompressedNN:
    return CompressedNN(model.seeds)


def uncompress(model: CompressedNN) -> UncompressedNN:
    with open('config.json') as f:
        device = torch.device(json.load(f)['device'])
    uncompressed = ConvNN(model.seeds[0]).to(device)
    for seed in model.seeds[1:]:
        uncompressed.mutate(seed)
    return uncompressed
