import gym

from preprocess import FrameStacker, DownSampler


env = FrameStacker(DownSampler(gym.make(
    'ALE/SpaceInvaders-v5',
    obs_type='grayscale',
    # render_mode='human',
    full_action_space=True)))

obs, info = env.reset(return_info=True)
done = False
for i in range(1000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()
env.close()
