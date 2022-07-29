import gym
import json
import numpy as np
import torch
from gym.wrappers import RecordVideo
from pathlib import Path
from typing import Dict, List, Tuple

from preprocess import DownSampler, FrameStacker
from uncompressed import UncompressedNN
from utils import get_env_name, load_model

gym.logger.set_level(40)


def play_n_episodes(
        model: UncompressedNN,
        env: gym.Env,
        cfg: Dict) -> Tuple[List[float], float]:
    device = torch.device(cfg['device'])
    model = model.to(device)
    obs = env.reset().to(device)
    n = cfg['eval_episodes']
    ep_count = 0
    scores = np.zeros(n)
    while ep_count < n:
        action = torch.argmax(model(obs)).item()
        obs, reward, done, info = env.step(action)
        obs = obs.to(device)
        scores[ep_count] += reward
        if done:
            print(f'Done episode {ep_count}')
            ep_count += 1
            env.reset()
    return list(scores), np.mean(scores)


def evaluate(env: gym.Env, cfg: Dict) -> float:
    model = load_model(cfg)
    scores, mean = play_n_episodes(model, env, cfg)
    return {'scores': scores, 'mean': mean}


if __name__ == '__main__':
    with open('config.json') as f:
        cfg = json.load(f)
    tstamp = cfg['timestamp']
    env_name = get_env_name(cfg)
    gen = cfg['generation']
    render = cfg['render'] == 'True'
    res_path = Path('results')/tstamp/env_name/f'gen{gen}'

    env = RecordVideo(FrameStacker(DownSampler(gym.make(
        cfg['environment'],
        obs_type='grayscale',
        render_mode='human' if render else 'rgb_array',
        full_action_space=True))),
        video_folder=res_path,
        episode_trigger=lambda i: i < cfg['eval_episodes'],
        name_prefix='test')

    res = evaluate(env, cfg)
    with open(res_path/'scores.json', 'w') as f:
        json.dump(res, f, indent=4)
