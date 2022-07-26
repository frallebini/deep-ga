import json
import gym
import torch

from conv import ConvNN
from preprocess import FrameStacker, DownSampler
from utils import compress, uncompress, restore_checkpoint

with open('config.json') as f:
    cfg = json.load(f)
env = FrameStacker(DownSampler(gym.make(cfg['environment'], obs_type='grayscale', full_action_space=True)))
# model = ConvNN(seed=0)
obs = env.reset()
# pred = model(obs)
# model.mutate(seed=1)
# pred = model(obs)
# comp = compress(model)
# uncomp = uncompress(comp)
# pred = uncomp(obs)

model = ConvNN()
pred = model(obs)

model = restore_checkpoint('SpaceInvaders', 0)

pred = model(obs)
var = 1