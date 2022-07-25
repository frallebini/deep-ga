import copy
import gym
import json
import random
import torch
from datetime import datetime
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


def train(env: gym.Env, cfg: Dict) -> None:
    gen = 0
    gen_frames = 0
    tot_frames = 0
    tot_time = 0
    timestamp = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    stats = {'best': [], 'mean': [], 'std': [], 'gen_frames': [], 'gen_time': [], 'parents': []}

    while tot_frames < cfg['max_train_frames']:
        start = time()
        if gen == 0:
            parents = []
            models = [CompressedNN() for _ in range(cfg['population_size'])]
            scores = []
        else:
            parents = models[:cfg['truncation_size']]
            models = [parents[0]]
            scores = [scores[0]]
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
        gen += 1


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
