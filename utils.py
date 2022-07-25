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
        scores: List[float]) -> Tuple[List[CompressedNN], List[float]]:
    models = np.array(models)
    scores = np.array(scores)
    by_decreasing_score = np.argsort(scores)[::-1]
    return list(models[by_decreasing_score]), list(scores[by_decreasing_score])


def log_stats(
        gen: int,
        scores: List[float],
        gen_frames: int,
        gen_time: int,
        tot_frames: int,
        tot_time: float,
        parents: List[CompressedNN],
        timestamp: str,
        stats: Dict) -> Dict:
    best = scores[0]
    mean = np.mean(scores)
    std = np.std(scores)
    stats['gen'] = gen
    stats['best'].append(best)
    stats['mean'].append(mean)
    stats['std'].append(std)
    stats['gen_frames'].append(gen_frames)
    stats['gen_time'].append(gen_time)
    stats['tot_frames'] = tot_frames
    stats['tot_time'] = tot_time
    if parents:
        stats['parents'] = [parent.seeds for parent in parents]
    stats['timestamp'] = timestamp
    print(f'Best score:      {best:.1f}')
    print(f'Mean score:      {mean:.1f}')
    print(f'Std of scores:   {std:.1f}')
    print(f'Tot # of frames: {tot_frames}')
    return stats


def save_checkpoint(
        model: CompressedNN,
        scores: List[float],
        gen: int,
        gen_frames: int,
        gen_time: float,
        tot_frames: int,
        tot_time: float,
        parents: List[CompressedNN],
        stats: Dict,
        timestamp: str,
        cfg: Dict) -> None:
    env_name = cfg['environment'].split('/')[-1].split('-')[0]
    stats = log_stats(
        gen, scores,
        gen_frames, gen_time, tot_frames, tot_time,
        parents, timestamp, stats)

    path = Path(f'stats/{timestamp}')
    path.mkdir(exist_ok=True)
    with open(path/f'{env_name}_gen{gen}.json', 'w') as f:
        json.dump(stats, f, indent=4)

    model = uncompress(model)
    path = Path(f'models/{timestamp}')
    path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path/f'{env_name}_gen{gen}.pt')


def restore_checkpoint(env_name: str, gen: int) -> UncompressedNN:
    model = ConvNN()
    model.load_state_dict(torch.load(f'models/{env_name}_gen{gen}.pt'))
    return model
