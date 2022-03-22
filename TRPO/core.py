
import torch
import torch.nn as nn
import numpy as np
import scipy
import gym
import scipy.signal as signal
import time

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(* layers)

def img_preprocess(images):
    images = images.transpose(-1,1)
    images = images/255.0
    return images

class cnn(torch.nn.Module):
    def __init__(self, out_dim):
        super(cnn, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3 , 32, 8, 4),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )
    def forward(self, x):
        x = img_preprocess(x)
        x = x.to('cuda')
        x = self.network(x)
        return x

def discount_cumsum(x, discount):
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError()
    
    def _log_prob_from_distributions(self, pi, act):
        raise NotImplementedError()
    
    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distributions(pi, act)
        return pi, logp_a
    
class MLPCategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim]+list(hidden_sizes)+[act_dim], activation)
    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return torch.distributions.Categorical(logits=logits)
    def _log_prob_from_distributions(self, pi, act):
        return pi.log_prob(act)

class CNNCategoricalActor(Actor):
    '''
    CNN Actor for discrete action space env
    '''
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.logits_net = cnn(act_dim).to('cuda')
        
    def _distribution(self, obs):
        logits = self.logits_net(obs)
        return torch.distributions.Categorical(logits=logits)
    
    def _log_prob_from_distributions(self, pi, act):
        return pi.log_prob(act)


class MLPGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.log_std = nn.Parameter(torch.ones(act_dim, dtype=torch.float32))
        self.mu_net = mlp([obs_dim]+list(hidden_sizes)+[act_dim], activation)
        
    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)
    
    def _log_prob_from_distributions(self, pi, act):
        '''
        for a given action, each dim of pi gives the probability of the same dim of action\n
        the possibility of action is the multiply of all dims\n
        so the log_prob should be the sum of it
        '''
        return pi.log_prob(act).sum(axis=-1)
    
class CNNGaussianActor(Actor):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.log_std = nn.Parameter(torch.ones(act_dim, dtype=torch.float32))
        self.mu_net = cnn(act_dim).to('cuda')
        
    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mu, std)
    
    def _log_prob_from_distributions(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)
    
class MLPCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim]+list(hidden_sizes)+[1], activation)
    
    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)
    
class CNNCritic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.v_net = cnn(1).to('cuda')
    
    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)
    
class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()
        
        obs_dim = observation_space.shape[0]

        if isinstance(action_space, gym.spaces.Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, gym.spaces.Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        self.v = MLPCritic(obs_dim, hidden_sizes, activation)
        
    def step(self, obs):
        '''
        for a given obs, sample the action from the distribution pi\n
        return sampled action, value of obs, logp_a
        '''
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distributions(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()
  
    
    def act(self, obs):
        '''
        same as step, only return the action
        '''
        return self.step(obs)[0]
 
class CNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        
        obs_dim = observation_space.shape[0]

        if isinstance(action_space, gym.spaces.Box):
            self.pi = CNNGaussianActor(obs_dim, action_space.shape[0]).to('cuda')
        elif isinstance(action_space, gym.spaces.Discrete):
            self.pi = CNNCategoricalActor(obs_dim, action_space.n).to('cuda')

        self.v = CNNCritic(obs_dim).to('cuda')
        
    def step(self, obs):
        '''
        for a given obs, sample the action from the distribution pi\n
        return sampled action, value of obs, logp_a
        '''
        obs = torch.unsqueeze(obs, 0)
        obs = obs.to('cuda')
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distributions(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()
  
    
    def act(self, obs):
        '''
        same as step, only return the action
        '''
        return self.step(obs)[0]   


def flat_concat(xs):
    return torch.cat([torch.reshape(x, (-1,)) for x in xs])

def flat_grad(f, params):
    return flat_concat(torch.autograd.grad(
        f, 
        params, 
        create_graph=True, 
        allow_unused=True, 
        retain_graph=True
        )
    )
def hessian_vector_product(f, v, model):
    '''
    H = grad**2 f, compute Hv
    '''
    g = torch.autograd.grad(f, model.parameters(), create_graph=True)
    # prod = sum([(g*v).sum() for g,v in zip(g, v)])
    g = flat_concat(g)
    prod = (g*v).sum()
    h = torch.autograd.grad(prod, model.parameters(), create_graph=True, allow_unused=True)
    return h
def concat_flat(x, shape_list):
    numel_list = [x.numel() for x in shape_list]
    pre = torch.split(x, numel_list)
    concat_x = [torch.reshape(x, s) for x, s in zip(pre, shape_list)]
    return concat_x
def concat_flat_for_nn(x, shape_list, key_list):
    concat_x = concat_flat(x, shape_list)
    return dict([(k,v) for k,v in zip(key_list, concat_x)])
