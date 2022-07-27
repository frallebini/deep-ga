import json
import gym
import torch
from gym.wrappers import RecordVideo
from typing import Dict, Tuple

from preprocess import FrameStacker, DownSampler
from uncompressed import UncompressedNN
from utils import get_env_name, load_model

gym.logger.set_level(40)


def play_episode(
        model: UncompressedNN,
        env: gym.Env,
        cfg: Dict) -> Tuple[float, int]:
    device = torch.device(cfg['device'])
    model = model.to(device)
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


def evaluate(env: gym.Env, cfg: Dict) -> float:
    model = load_model(cfg)
    score, _ = play_episode(model, env, cfg)
    return score


if __name__ == '__main__':
    with open('config.json') as f:
        cfg = json.load(f)
    tstamp = cfg['timestamp']
    env_name = get_env_name(cfg)
    gen = cfg['generation']

    env = RecordVideo(FrameStacker(DownSampler(gym.make(
        cfg['environment'],
        obs_type='grayscale',
        render_mode='human',
        full_action_space=True))),
        video_folder=f'videos/{tstamp}/{env_name}_gen{gen}',
        name_prefix='')

    score = evaluate(env, cfg)
    print(score)
