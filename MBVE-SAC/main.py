
from agents.sac_agent import SAC
from common.replay_buffer import ReplayBuffer
from agents.rand_agent import RandomAgent
from common.env_sampler import EnvSampler
import gym
import pybulletgym
import torch.nn as nn
import numpy as np

env_fn = lambda : gym.make('HumanoidFlagrunHarderPyBulletEnv-v0')

EPOCHS = 400
STEPS_PER_EPOCH = 4000
START_STEPS = 10000
UPDATE_AFTER = 1000
UPDATE_EVERY = 50

env = env_fn()
env.spec

env_pool = ReplayBuffer(
    env_fn(),
    int(1E6)
)
sac_agent = SAC(
    env = env_fn(),
    hidden_sizes = [256, 256, 256],
    activation = nn.ReLU,
    lr = 5e-4
)
rand_agent = RandomAgent(
    env = env_fn()
)
env_sampler = EnvSampler(
    env = env_fn(),
    max_ep_len = 1000
)



for t in range(EPOCHS * STEPS_PER_EPOCH):

    if t > START_STEPS:
        env_sampler.sample_and_push(
            env_pool,
            sac_agent,
            rescale_factor=1
        )
    else:
        env_sampler.sample_and_push(
            env_pool,
            rand_agent,
            rescale_factor=1
        )
        
    if t >= UPDATE_AFTER and t % UPDATE_EVERY == 0:
        for j in range(UPDATE_EVERY):
            batch = env_pool.sample(batch_size = 256)
            sac_agent.update(batch)
            
    if (t + 1) % STEPS_PER_EPOCH == 0:
        epoch = (t + 1) // STEPS_PER_EPOCH
        sac_agent.test()
        
        print(f'Epoch: {epoch}\tTotal steps: {t + 1}')
        sac_agent.print_log()
        env_sampler.mean_ep_len = env_sampler.mean_ep_len*0.5 + sac_agent.mean_ep_len*0.5
        
        sac_agent.save('/home/grandpadzb/MathscriptsLib/RL/MBVE-SAC/save/HumanoidFlagrunHarderPyBulletEnv1')
    
