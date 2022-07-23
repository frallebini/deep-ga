import copy
import gym
import json
import random
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Dict

from compressed import CompressedNN
from preprocess import FrameStacker, DownSampler
from utils import uncompress, sort_by_score, log_scores

gym.logger.set_level(40)


def play_episode(compressed_model: CompressedNN, env: gym.Env, max_frames: int) -> Tuple[float, int]:
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


def train(env: gym.Env, cfg: Dict) -> Tuple[CompressedNN, Dict]:
    gen_count = 0
    frame_count = 0
    stats = {'best': [], 'mean': [], 'std': []}
    while frame_count < 1:  # cfg['max_train_frames']:
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
        stats = log_scores(scores, stats)
        gen_count += 1
    stats['gen_count'] = gen_count
    return models[0], stats


if __name__ == '__main__':
    random.seed(42)

    with open('config.json') as f:
        cfg = json.load(f)
    fname = cfg['environment'].split('/')[-1].split('-')[0]

    env = FrameStacker(DownSampler(gym.make(
        cfg['environment'],
        obs_type='grayscale',
        # render_mode='human',
        full_action_space=True)))

    model, stats = train(env, cfg)

    path = Path('stats')
    path.mkdir(exist_ok=True)
    with open(path/f'{fname}.json', 'w') as f:
        json.dump(stats, f, indent=4)

    path = Path('models')
    path.mkdir(exist_ok=True)
    model = uncompress(model)
    torch.save(model.state_dict(), path/f'{fname}.pt')
