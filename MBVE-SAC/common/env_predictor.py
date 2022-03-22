import gym
import torch
import torch.nn as nn
import numpy as np
from . import core
import random
import torch.distributions as td


class EnvPredictor:
    def __init__(
        self, 
        ensemble_nums = 5,
        hidden_sizes = (256, 256),
        env = gym.make("HalfCheetah-v3"),
        lr = 3E-4,

    ):
        self.ensemble_nums = ensemble_nums
        self.act_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]
        
        self.input_dim = self.obs_dim + self.act_dim
        self.output_dim = self.obs_dim + 1
        
        self.ensemble_envs = [
            core.mlp([self.input_dim] + list(hidden_sizes), activation=nn.ReLU) for _ in range(ensemble_nums)
        ]
        self.ensemble_mean_layers = [
            nn.Linear(hidden_sizes[-1], self.output_dim) for _ in range(ensemble_nums)
        ]
        self.ensemble_logvar_layers = [
            nn.Linear(hidden_sizes[-1], self.output_dim) for _ in range(ensemble_nums)
        ]
        

        self.env_optimizers = [
            torch.optim.Adam(self.ensemble_envs[i].parameters(), lr=lr) for i in range(ensemble_nums) 
        ]
        self.loss = [0 for _ in range(ensemble_nums)]
        
        self.sfp = nn.Softplus()

    def step(self, s, a):
        a = torch.as_tensor(a, dtype=torch.float32)
        
        #kmin = np.argmax(self.loss)
        k = random.randint(0, self.ensemble_nums-1)
        #k = kmin
        with torch.no_grad():
            feats = self.ensemble_envs[k](torch.cat([s, a], dim=-1))
            
            s2_r_mean = self.ensemble_mean_layers[k](feats)
            s2_r_logvar = self.ensemble_logvar_layers[k](feats)
            
            max_logvar = torch.max(s2_r_logvar, 0).values
            min_logvar = torch.min(s2_r_logvar, 0).values
            
            s2_r_logvar = max_logvar - self.sfp(max_logvar - s2_r_logvar)
            s2_r_logvar = min_logvar + self.sfp(s2_r_logvar - min_logvar)
            
            s2_r_var = torch.exp(s2_r_logvar)
        
            dist = td.Normal(s2_r_mean, s2_r_var)
        
            s2_r = dist.sample()
        
        s2 = s2_r[:,:-1]
        r = s2_r[:,-1]
        return s2, r, torch.zeros_like(r)
    
    def train(self, batchs):
        for n in range(self.ensemble_nums):
            s, a, r, s2 = batchs[n]['s'], batchs[n]['a'], batchs[n]["r"], batchs[n]['s2']
            r = torch.unsqueeze(r, dim=1)
            
            feats = self.ensemble_envs[n](torch.cat([s, a], dim=-1))
            
            s2_r_mean = self.ensemble_mean_layers[n](feats)
            s2_r_logvar = self.ensemble_logvar_layers[n](feats)
            
            max_logvar = torch.max(s2_r_logvar, 0).values
            min_logvar = torch.min(s2_r_logvar, 0).values
            
            s2_r_logvar = max_logvar - self.sfp(max_logvar - s2_r_logvar)
            s2_r_logvar = min_logvar + self.sfp(s2_r_logvar - min_logvar)
            
            s2_r_var = torch.exp(s2_r_logvar)
            
            s2_r_true = torch.cat([s2, r], dim=-1)
            
            loss = (((s2_r_true-s2_r_mean)**2)/s2_r_var + s2_r_logvar).sum()
            
            self.env_optimizers[n].zero_grad()
            loss.backward()
            self.env_optimizers[n].step()
            with torch.no_grad():
                self.loss[n] = loss.detach().cpu().numpy()
            
    def rollout_and_push(
        self,
        env_pool, 
        pred_pool, 
        agent, 
        k = 5, 
        batch_size = 256
        ):
        for _ in range(k):
            s = torch.squeeze(env_pool.sample_s(batch_size))
            a = torch.as_tensor(agent.act(s))
            with torch.no_grad():
                s2, r, d = self.step(s, a)
            pred_pool.push_batch(s, a, r, s2, d)
            s = s2
            
            
    
    