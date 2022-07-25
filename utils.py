import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple

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


def sort_by_score(
        models: List[CompressedNN],
        scores: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    models = np.array(models)
    scores = np.array(scores)
    by_decreasing_score = np.argsort(scores)[::-1]
    return models[by_decreasing_score], scores[by_decreasing_score]


def log_stats(
        scores: List[float],
        frames: int,
        time: float,
        stats: Dict) -> Dict:
    best = scores[0]
    mean = np.mean(scores)
    std = np.std(scores)
    stats['best'].append(best)
    stats['mean'].append(mean)
    stats['std'].append(std)
    stats['tot_frames'].append(frames)
    stats['tot_time'].append(time)
    print(f'Best score:      {best:.1f}')
    print(f'Mean score:      {mean:.1f}')
    print(f'Std of scores:   {std:.1f}')
    print(f'Tot # of frames: {frames}')
    return stats


def save_checkpoint(
        model: CompressedNN,
        scores: List[float],
        gen: int,
        frames: int,
        time: float,
        stats: Dict,
        cfg: Dict) -> None:
    env_name = cfg['environment'].split('/')[-1].split('-')[0]
    stats = log_stats(scores, frames, time, stats)

    path = Path('stats')
    path.mkdir(exist_ok=True)
    with open(path/f'{env_name}_gen{gen}.json', 'w') as f:
        json.dump(stats, f, indent=4)
    if gen > 0:
        Path(path/f'{env_name}_gen{gen-1}.json').unlink()

    model = uncompress(model)
    path = Path('models')
    path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path/f'{env_name}_gen{gen}.pt')


def restore_checkpoint(env_name: str, gen: int) -> UncompressedNN:
    model = ConvNN()
    model.load_state_dict(torch.load(f'models/{env_name}_gen{gen}.pt'))
    return model
