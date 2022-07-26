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
from utils import save_checkpoint, sort_by_score

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
        timestamp = stats['timestamp']
    else:
        restart = False
        gen = 0
        gen_frames = 0
        tot_frames = 0
        tot_time = 0
        parents = []
        scores = []
        stats = {'best': [], 'mean': [], 'std': [], 'gen_frames': [], 'gen_time': [], 'parents': []}
        timestamp = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')

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
                score, ep_frames = play_episode(models[i], env, cfg)
            else:
                model = copy.deepcopy((random.choice(parents)))
                model.mutate()
                models.append(model)
                score, ep_frames = play_episode(model, env, cfg)
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
            stats, timestamp, cfg)
        restart = False
        gen += 1


def restart_training(env: gym.Env, cfg: Dict) -> None:
    timestamp_path = Path('stats') / f'{cfg["timestamp"]}'
    stats_path = sorted(timestamp_path.iterdir(), key=lambda path: path.name)[-1]
    with open(stats_path) as f:
        stats = json.load(f)
    stats['gen'] += 1
    train(env, cfg, stats)


if __name__ == '__main__':
    random.seed(42)  # there will still be the stochasticity of the environment
    restart = True

    with open('config.json') as f:
        cfg = json.load(f)

    env = FrameStacker(DownSampler(gym.make(
        cfg['environment'],
        obs_type='grayscale',
        # render_mode='human',
        full_action_space=True)))

    if restart:
        restart_training(env, cfg)
    else:
        train(env, cfg)
