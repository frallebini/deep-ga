import copy
import gym
import json
import random
from datetime import datetime
from pathlib import Path
from time import time
from tqdm import tqdm
from typing import Dict

from compressed import CompressedNN
from evaluate import play_episode
from preprocess import DownSampler, FrameStacker
from utils import get_env_name, get_latest, save_checkpoint, sort_by_score, uncompress

gym.logger.set_level(40)


def train(env: gym.Env, cfg: Dict, stats: Dict = None) -> None:
    if stats:
        restart = True
        gen = stats['gen']
        gen_frames = stats['gen_frames'][-1]
        tot_frames = stats['tot_frames']
        tot_time = stats['tot_time']
        parents = [CompressedNN(seeds) for seeds in stats['parents']]
        scores = [stats['best'][-1]]
        tstamp = stats['timestamp']
    else:
        restart = False
        gen = 0
        gen_frames = 0
        tot_frames = 0
        tot_time = 0
        parents = []
        scores = []
        stats = {'best': [], 'mean': [], 'std': [], 'gen_frames': [], 'gen_time': [], 'parents': []}
        tstamp = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')

    while tot_frames < cfg['max_train_frames']:
        start = time()
        if gen == 0:
            models = [CompressedNN() for _ in range(cfg['population_size'])]
        else:
            if not restart:
                parents = models[:cfg['truncation_size']]
                scores = [scores[0]]
            models = [parents[0]]
        for i in tqdm(range(cfg['population_size']), desc=f'Gen {gen}'):
            if gen == 0:
                score, ep_frames = play_episode(uncompress(models[i]), env, cfg)
            else:
                model = copy.deepcopy((random.choice(parents)))
                model.mutate()
                models.append(model)
                score, ep_frames = play_episode(uncompress(model), env, cfg)
            scores.append(score)
            gen_frames += ep_frames
        models, scores = sort_by_score(models, scores)
        end = time()

        gen_time = end - start
        tot_time += gen_time
        tot_frames += gen_frames
        save_checkpoint(
            models[0], scores, gen,
            gen_frames, gen_time, tot_frames, tot_time, parents,
            stats, tstamp, cfg)
        restart = False
        gen += 1


def restart_training(env: gym.Env, cfg: Dict) -> None:
    tstamp = cfg["timestamp"]
    tstamp_path = sorted(Path('stats').iterdir())[-1] if tstamp == 'latest' else Path('stats')/tstamp
    stats_path = get_latest(tstamp_path)
    with open(stats_path) as f:
        stats = json.load(f)
    stats['gen'] += 1
    train(env, cfg, stats)


if __name__ == '__main__':
    random.seed(42)  # there will still be the stochasticity of the environment

    with open('config.json') as f:
        cfg = json.load(f)

    env = FrameStacker(DownSampler(gym.make(
        cfg['environment'],
        obs_type='grayscale',
        full_action_space=True)))
    env_name = get_env_name(cfg)

    restart = cfg['restart'] == 'True'
    if restart:
        print(f'Restarting training on {env_name}')
        restart_training(env, cfg)
    else:
        print(f'Training from scratch on {env_name}')
        train(env, cfg)
