import gym
import torch
from typing import Dict, Tuple

from compressed import CompressedNN
from utils import uncompress


def play_episode(
        compressed_model: CompressedNN,
        env: gym.Env,
        cfg: Dict) -> Tuple[float, int]:
    model = uncompress(compressed_model)
    device = torch.device(cfg['device'])
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
