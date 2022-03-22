import numpy as np
import torch
import torch.nn as nn
import gym
import time
import core
import psutil
import os


def ddpg(
    env_fn = lambda : gym.make("LunarLanderContinuous-v2"),
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
    
    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)
    
    o, d, ep_ret, ep_len = env.reset(), False, 0, 0
    while not(d or (ep_len == max_ep_len)):
        env.render()
        o, r, d, _ = env.step(get_action(o, 0))
        ep_ret += r
        ep_len += 1
    print(f'EpRet: {ep_ret}')
    return

if __name__ == "__main__":
    ddpg(
        env_fn = lambda : gym.make("BipedalWalkerHardcore-v2"),
        actor_critic = core.MLPActorCritic,
        ac_kwargs = dict(
            hidden_sizes=(128, 128, 128),
            activation=nn.ReLU
        ),
        seed = 17,
        max_ep_len = 1000,
        load_path='./save/BipedalWalkerHardcore1.pt'
    )
