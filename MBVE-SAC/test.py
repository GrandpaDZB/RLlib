import numpy as np
import torch
import torch.nn as nn
import gym
import time
import common.core as core
import psutil
import os
import pybulletgym

def sac(
    env_fn = lambda : gym.make("HumanoidFlagrunHarderPyBulletEnv-v0"),
    actor_critic = core.MLPActorCritic,
    ac_kwargs = dict(
        hidden_sizes=(255,255),
        activation=nn.ReLU
    ),
    seed = 0,
    max_ep_len = 1000,
    load_path = ''
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = env_fn()
    act_dim = env.action_space.shape[0]
    
    act_limit = env.action_space.high[0]
    
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac.load_state_dict(torch.load(load_path))
    
    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)
    
    env.render()
    o, d, ep_ret, ep_len = env.reset(), False, 0, 0
    while not(d or (ep_len == max_ep_len)):
        o, r, d, _ = env.step(get_action(o, True))
        ep_ret += r
        ep_len += 1
    print(f'EpRet: {ep_ret}')
    return

if __name__ == "__main__":
    sac(
        env_fn = lambda : gym.make("HumanoidFlagrunHarderPyBulletEnv-v0"),
        actor_critic = core.MLPActorCritic,
        ac_kwargs = dict(
            hidden_sizes=(256, 256, 256),
            activation=nn.ReLU
        ),
        seed = 0,
        max_ep_len = 1000,
        load_path='./save/HumanoidFlagrunHarderPyBulletEnv1.pt'
    )
