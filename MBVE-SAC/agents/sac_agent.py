import numpy as np
from agents.rand_agent import RLAgent
import torch
import torch.nn as nn
from common import core
import gym

device = "cuda"

class SAC(RLAgent):
    def __init__(
        self, 
        env = gym.make('HalfCheetah-v2'), 
        hidden_sizes = [256, 256],
        activation = nn.ReLU,
        seed = 0,
        gamma = 0.99,
        polyak = 0.995,
        lr = 3e-4,
        ):
        self.env = env
        self.gamma = gamma
        self.polyak = polyak
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.ac = core.MLPActorCritic(
            env.observation_space,
            env.action_space,
            hidden_sizes,
            activation
        ).to(device)
        
        self.ac_targ = core.MLPActorCritic(
            env.observation_space,
            env.action_space,
            hidden_sizes,
            activation
        ).to(device)
        
        obs_dim = env.observation_space.shape
        self.act_dim = env.action_space.shape[0]
        
        self.entropy = 1.0
        self.loss_pi = None
        self.loss_q1 = None
        self.loss_q2 = None
        self.mean_ep_ret = None
        self.max_ep_ret = None
        self.min_ep_ret = None
        self.mean_ep_len = None
        self.history = []
        
        
        self.pi_optimizer = torch.optim.Adam(self.ac.pi.parameters(), lr = lr)
        self.q1_optimizer = torch.optim.Adam(self.ac.q1.parameters(), lr = lr)
        self.q2_optimizer = torch.optim.Adam(self.ac.q2.parameters(), lr = lr)
        self.alpha_optimizer = torch.optim.Adam([self.ac.alpha], lr = lr)
        
    
    def act(self, s, deterministic=False, std_scale = 1.0):
        s = torch.as_tensor(s, dtype=torch.float32).to(device)
        return self.ac.act(s, deterministic=deterministic, std_scale=std_scale)
    
    def _compute_loss_q(self, batch):
        s, a, r, s2, d = batch['s'], batch['a'], batch["r"], batch['s2'], batch['d']

        q1 = self.ac.q1(s, a)
        q2 = self.ac.q2(s, a)
        
        with torch.no_grad():
            a2, logp_a2 = self.ac.pi(s2)
            
            q1_targ = self.ac_targ.q1(s2, a2)
            q2_targ = self.ac_targ.q2(s2, a2)
            q_targ = torch.min(q1_targ, q2_targ)
            
            q_togo = r + self.gamma * (1 - d) * (q_targ - self.ac.alpha * logp_a2)
        
        loss_q1 = ((q1 - q_togo)**2).mean()
        loss_q2 = ((q2 - q_togo)**2).mean()
        return loss_q1, loss_q2
    
    def _compute_loss_pi(self, batch):
        s = batch['s']
        pi, logp_pi = self.ac.pi(s)
        q1 = self.ac.q1(s, pi)
        q2 = self.ac.q2(s, pi)
        q = torch.min(q1, q2)
        
        loss_pi = (self.ac.alpha * logp_pi - q).mean()
        
        with torch.no_grad():
            self.entropy = (-logp_pi.mean()) #.clone().detach()
        
        loss_alpha = self.ac.alpha * self.entropy + self.ac.alpha * self.act_dim
        
        return loss_pi, loss_alpha
        
    
    
    
    def update(self, batch):
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        loss_q1, loss_q2 = self._compute_loss_q(batch)
        loss_q1.backward()
        loss_q2.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        
        
        for p in self.ac.q1.parameters():
            p.requires_grad = False
        for p in self.ac.q2.parameters():
            p.requires_grad = False
            
        self.pi_optimizer.zero_grad()
        loss_pi, loss_alpha = self._compute_loss_pi(batch)
        loss_pi.backward(retain_graph=True)
        self.pi_optimizer.step()
        
        self.alpha_optimizer.zero_grad()
        loss_alpha.backward()
        self.alpha_optimizer.step()
        
        for p in self.ac.q1.parameters():
            p.requires_grad = True
        for p in self.ac.q2.parameters():
            p.requires_grad = True
            
        
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(), self.ac_targ.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        
            self.loss_pi = loss_pi
            self.loss_q1 = loss_q1
            self.loss_q2 = loss_q2

    def save(self, path):
        torch.save(self.ac.state_dict(), path + '.pt')
        torch.save(self.ac_targ.state_dict(), path + '_targ.pt')
        
    def load(self, path):
        self.ac.load_state_dict(torch.load(path + '.pt'))
        self.ac_targ.load_state_dict(torch.load(path + '_targ.pt'))
        
    def test(self, num_test_episodes = 10, max_ep_len = 1000):
        mean_ep_ret = 0
        mean_ep_len = 0
        max_ep_ret = 0
        min_ep_ret = 0
        
        for j in range(num_test_episodes):
            s, d, ep_ret, ep_len = self.env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                s, r, d, _ = self.env.step(self.act(s, True))
                ep_ret += r
                ep_len += 1
            mean_ep_ret = (j/(j+1))*mean_ep_ret + (1/(j+1))*ep_ret
            mean_ep_len = (j/(j+1))*mean_ep_len + (1/(j+1))*ep_len
            
            if ep_ret > max_ep_ret or j == 0:
                max_ep_ret = ep_ret
            if ep_ret < min_ep_ret or j == 0:
                min_ep_ret = ep_ret
        self.mean_ep_ret = mean_ep_ret
        self.mean_ep_len = mean_ep_len
        self.max_ep_ret = max_ep_ret
        self.min_ep_ret = min_ep_ret
        self.history.append(self.mean_ep_ret)
        
    def print_log(self):
        print('------------------------------------')
        print(f'EpRet: {self.mean_ep_ret}')
        print(f'EpLen: {self.mean_ep_len}')
        print(f'MaxRet: {self.max_ep_ret}')
        print(f'MinRet: {self.min_ep_ret}')
        print(f'Entropy: {self.entropy}')
        print(f'Temperature: {self.ac.alpha.data}')
        print(f'LossQ1: {self.loss_q1}')
        print(f'LossQ2: {self.loss_q2}')
        print('------------------------------------\n')