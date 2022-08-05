import gym
import json
import numpy as np
import torch
from collections import deque
from skimage.transform import resize


class DownSampler(gym.ObservationWrapper):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return resize(obs, (84, 84))


class FrameStacker(gym.ObservationWrapper):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        with open('config.json') as f:
            self.n_frames = json.load(f)['stacked_frames']
        self.frames = deque(maxlen=self.n_frames)

    def observation(self, obs: np.ndarray) -> torch.Tensor:
        self.frames.append(obs)
        if len(self.frames) == 1:
            self.frames.extend([obs for _ in range(self.n_frames-1)])
        return torch.unsqueeze(torch.from_numpy(np.array(self.frames)).float(), dim=0)
