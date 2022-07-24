import copy
import gym
import json
import random
import torch
from time import time
from tqdm import tqdm
from typing import Dict, Tuple

from compressed import CompressedNN
from preprocess import DownSampler, FrameStacker
from utils import save_checkpoint, sort_by_score, uncompress

gym.logger.set_level(40)


def play_episode(
        compressed_model: CompressedNN,
        env: gym.Env,
        max_frames: int) -> Tuple[float, int]:
    model = uncompress(compressed_model)
    obs = env.reset()
    score = 0
    while True:
        action = torch.argmax(model(obs)).item()
        obs, reward, done, info = env.step(action)
        score += reward
        if done or info['episode_frame_number'] > max_frames:
            break
    return score, info['episode_frame_number']


def train(env: gym.Env, cfg: Dict) -> None:
    gen_count = 0
    frame_count = 0
    time_count = 0
    stats = {'best': [], 'mean': [], 'std': [], 'tot_time': [], 'tot_frames': []}
    while frame_count < cfg['max_train_frames']:
        start = time()
        if gen_count == 0:
            models = [CompressedNN() for _ in range(cfg['population_size'])]
            scores = []
        else:
            parents = models[:cfg['truncation_size']]
            models = [parents[0]]
            scores = [scores[0]]
        for i in tqdm(range(cfg['population_size']), desc=f'Gen {gen_count}'):
            if gen_count == 0:
                score, n_frames = play_episode(models[i], env, cfg['max_episode_frames'])
            else:
                model = copy.deepcopy((random.choice(parents)))
                model.mutate()
                models.append(model)
                score, n_frames = play_episode(model, env, cfg['max_episode_frames'])
            scores.append(score)
            frame_count += n_frames
        models, scores = sort_by_score(models, scores)
        end = time()
        time_count += end - start
        save_checkpoint(models[0], scores, gen_count, frame_count, time_count, stats, cfg)
        gen_count += 1


if __name__ == '__main__':
    random.seed(42)

    with open('config.json') as f:
        cfg = json.load(f)

    env = FrameStacker(DownSampler(gym.make(
        cfg['environment'],
        obs_type='grayscale',
        # render_mode='human',
        full_action_space=True)))

    train(env, cfg)
