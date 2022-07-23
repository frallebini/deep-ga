import numpy as np
from typing import Dict, List, Tuple

from compressed import CompressedNN
from conv import ConvNN
from uncompressed import UncompressedNN


def compress(model: UncompressedNN) -> CompressedNN:
    return CompressedNN(model.seeds)


def uncompress(model: CompressedNN, type='conv') -> UncompressedNN:
    if type == 'conv':
        uncompressed = ConvNN(model.seeds[0])
    else:
        raise NotImplementedError
    for seed in model.seeds[1:]:
        uncompressed.mutate(seed)
    return uncompressed


def sort_by_score(models: List[CompressedNN], scores: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    models = np.array(models)
    scores = np.array(scores)
    by_decreasing_score = np.argsort(scores)[::-1]
    return models[by_decreasing_score], scores[by_decreasing_score]


def log_scores(scores: List[float], stats: Dict) -> Dict:
    best = scores[0]
    mean = np.mean(scores)
    std = np.std(scores)
    stats['best'].append(best)
    stats['mean'].append(mean)
    stats['std'].append(std)
    print(f'Best score:    {best:.1f}')
    print(f'Mean score:    {mean:.1f}')
    print(f'Std of scores: {std:.1f}')
    return stats
