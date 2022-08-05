import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple

from comp_to_uncomp import uncompress
from compressed import CompressedNN
from conv import ConvNN
from uncompressed import UncompressedNN


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
        tstamp: str,
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
    stats['parents'] = [parent.seeds for parent in parents]
    stats['timestamp'] = tstamp
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
        tstamp: str,
        cfg: Dict) -> None:
    env_name = get_env_name(cfg)
    stats = log_stats(
        gen, scores,
        gen_frames, gen_time, tot_frames, tot_time,
        parents, tstamp, stats)

    path = Path('stats')/tstamp
    path.mkdir(exist_ok=True)
    with open(path/f'{env_name}_gen{gen}.json', 'w') as f:
        json.dump(stats, f, indent=4)

    model = uncompress(model)
    path = Path('models')/tstamp
    path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), path/f'{env_name}_gen{gen}.pt')


def load_model(cfg: Dict) -> UncompressedNN:
    tstamp = cfg['timestamp']
    gen = cfg['generation']
    env_name = get_env_name(cfg)

    print(f'Loading model w/ timestamp {tstamp} @ gen {gen} trained on {env_name}')

    tstamp_path = sorted(Path('models').iterdir())[-1] if tstamp == 'latest' else Path('models')/tstamp
    model_path = get_latest(tstamp_path) if gen == 'latest' else tstamp_path/f'{env_name}_gen{gen}.pt'

    model = ConvNN()
    device = torch.device(cfg['device'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def get_env_name(cfg: Dict) -> str:
    return cfg['environment'].split('/')[-1].split('-')[0]


def get_latest(path: Path) -> Path:
    return sorted(path.iterdir(), key=lambda p: int(p.stem.split('_')[-1].replace('gen', '')))[-1]


def delete_meta_files(path: Path) -> None:
    for f in path.glob('*.meta.json'):
        f.unlink()
