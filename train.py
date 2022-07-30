import copy
import gym
import json
import numpy as np
import random
import torch
from datetime import datetime
from pathlib import Path
from time import time
from tqdm import tqdm
from typing import Dict, List, Tuple

from compressed import CompressedNN
from preprocess import DownSampler, FrameStacker
from utils import get_env_name, get_latest, save_checkpoint, sort_by_score, uncompress

gym.logger.set_level(40)


def play_one_episode(
        compressed: CompressedNN,
        env: gym.Env,
        cfg: Dict) -> Tuple[float, int]:
    device = torch.device(cfg['device'])
    model = uncompress(compressed).to(device)
    obs = env.reset().to(device)
    score = 0
    while True:
        action = torch.argmax(model(obs)).item()
        obs, reward, done, info = env.step(action)
        obs = obs.to(device)
        score += reward
        if done or info['episode_frame_number'] > cfg['max_episode_frames']:
            break
    return score, info['episode_frame_number']


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
        scores = []
        stats = {'best': [], 'mean': [], 'std': [], 'gen_frames': [], 'gen_time': []}
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
                score, ep_frames = play_one_episode(models[i], env, cfg)
            else:
                model = copy.deepcopy((random.choice(parents)))
                model.mutate()
                models.append(model)
                score, ep_frames = play_one_episode(model, env, cfg)
            scores.append(score)
            gen_frames += ep_frames
        models, scores = sort_by_score(models, scores)
        elite, refined_scores, add_frames = select_elite(models, env, cfg)
        scores[:cfg['elite_candidates']] = refined_scores
        scores[0], scores[elite] = scores[elite], scores[0]
        models[0], models[elite] = models[elite], models[0]
        gen_frames += add_frames
        end = time()

        gen_time = end - start
        tot_time += gen_time
        tot_frames += gen_frames
        if gen == 0:
            parents = models[:cfg['truncation_size']]
        save_checkpoint(
            models[0], scores, gen,
            gen_frames, gen_time, tot_frames, tot_time, parents,
            stats, tstamp, cfg)
        restart = False
        gen += 1


def select_elite(
        models: List[CompressedNN],
        env: gym.Env,
        cfg: Dict) -> Tuple[int, List[float], int]:
    candidates = models[:cfg['elite_candidates']]
    scores = np.zeros((len(candidates), cfg['additional_episodes']))
    add_frames = 0
    for (i, model) in enumerate(candidates):
        for j in tqdm(range(cfg['additional_episodes']), desc=f'\tAdditional episodes {i}'):
            score, ep_frames = play_one_episode(model, env, cfg)
            scores[i, j] = score
            add_frames += ep_frames
    scores = np.mean(scores, axis=1)
    elite = np.argmax(scores)
    return elite, list(scores), add_frames


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

    if cfg['restart'] == 'True':
        print(f'Restarting training on {env_name}')
        restart_training(env, cfg)
    else:
        print(f'Training from scratch on {env_name}')
        train(env, cfg)
