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

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self.frames.append(obs)
        if len(self.frames) == 1:
            self.frames.extend([obs for _ in range(self.n_frames-1)])
        return torch.unsqueeze(torch.from_numpy(np.array(self.frames)).float(), dim=0)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def show_frames(frames: np.ndarray, title: str) -> None:
        n_frames = frames.shape[0]
        for i in range(n_frames):
            plt.subplot(n_frames//2, 2, i+1)
            plt.imshow(frames[i], cmap='gray')
            plt.title(f'Frame {i+1}')
            plt.xticks([])
            plt.yticks([])
        plt.suptitle(title)
        plt.show()

    env = FrameStacker(DownSampler(gym.make('ALE/Breakout-v5', obs_type='grayscale')))
    obs = env.reset().squeeze().numpy()
    show_frames(obs, 'Start')
    for i in range(env.n_frames):
        obs, _, _, _ = env.step(env.get_action_meanings().index('LEFT'))
        obs = obs.squeeze().numpy()
        show_frames(obs, f'Step {i+1}')
