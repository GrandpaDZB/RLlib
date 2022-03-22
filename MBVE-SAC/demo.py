import imp
from agents.sac_agent import SAC
from common.env_predictor import EnvPredictor
from common.env_sampler import EnvSampler
from agents.rand_agent import RandomAgent
from common.replay_buffer import ReplayBuffer
from common import core
import torch
import torch.nn as nn
import numpy as np
import gym
import sys

env_fn = lambda : gym.make('HalfCheetah-v3')

sac_agent = SAC(
    env_fn(),
    hidden_sizes=(256, 256)
)
rand_agent = RandomAgent(
    env_fn(),
)

env_predictor = EnvPredictor(
    ensemble_nums=7,
    hidden_sizes=(200, 200),
    lr = 1E-3
)
env_sampler = EnvSampler(
    env_fn(),
)

env_pool = ReplayBuffer(
    env_fn(),
    int(1e6)
)
pred_pool = ReplayBuffer(
    env_fn(),
    int(1e6)
)

EPOCHS = 200
STEPS_PER_EPOCH = 1000

UPDATE_START_STEPS = 3000
RANDOM_STEPS = 1000

ENV_UPDATE_INTERVAL = 25
AC_UPDATE_INTERVAL = 50

ENV_UPDATE_START_STEPS = 5*STEPS_PER_EPOCH
FIRST_ENV_UPDATE = False

ENV_TRAIN_ITERATIONS = 500
ENV_STEPS = 800
MODEL_ROLLOUTS = 50
GRAD_UPDATES = 50



# for _ in range(40000):
#     env_sampler.sample_and_push(
#         env_pool,
#         sac_agent,
#     )
# for _ in range(ENV_TRAIN_ITERATIONS):
#     batchs = [env_pool.sample(256) for _ in range(7)]
#     env_predictor.train(batchs)
#     print(env_predictor.loss)


for _ in range(10000):
    env_sampler.sample_and_push(
        env_pool,
        sac_agent,
    )
for _ in range(5000):
    batchs = [env_pool.sample(256) for _ in range(7)]
    env_predictor.train(batchs)
    print(env_predictor.loss)

for step in range(STEPS_PER_EPOCH * EPOCHS):

    env_sampler.sample_and_push(
        env_pool,
        sac_agent,
    )

    
    if step % 50 == 0:
        if not FIRST_ENV_UPDATE:
            FIRST_ENV_UPDATE = True
            for _ in range(ENV_TRAIN_ITERATIONS):
                batchs = [env_pool.sample(256) for _ in range(7)]
                env_predictor.train(batchs)
                print(env_predictor.loss)
            
        for _ in range(50):
            batchs = [env_pool.sample(256) for _ in range(7)]
            env_predictor.train(batchs)
        for _ in range(MODEL_ROLLOUTS):
            env_predictor.rollout_and_push(
                env_pool,
                pred_pool,
                sac_agent,
                1,
                256
            )
   
    if step % 50 == 0 and step > UPDATE_START_STEPS:
        for _ in range(GRAD_UPDATES):
            batch = pred_pool.sample(256)
            sac_agent.update(batch)
      
                
    if step % STEPS_PER_EPOCH == 0:
        sac_agent.test()
        print(f'Epoch: {int(step/STEPS_PER_EPOCH)}\tTotal steps: {step}')
        sac_agent.print_log()
        print(env_predictor.loss)
    







if True:
    sys.exit(0)
    
    
sac_agent = SAC(
    env_fn(),
    hidden_sizes=(256, 256)
)

for _ in range(10000):
    env_sampler.sample_and_push(
        env_pool,
        sac_agent,
    )
for _ in range(5000):
    batchs = [env_pool.sample(256) for _ in range(7)]
    env_predictor.train(batchs)
    print(env_predictor.loss)


for _ in range(400):
    env_predictor.rollout_and_push(
        env_pool,
        pred_pool,
        sac_agent,
        1,
        256
    )
    
    
for _ in range(500):
    batch = pred_pool.sample(256)
    sac_agent.update(batch)
sac_agent.test()
sac_agent.print_log()
print(env_predictor.loss)