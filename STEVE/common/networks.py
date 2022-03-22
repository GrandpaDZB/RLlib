import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal
import torch.nn.functional as F
import gym
import time

device = 'cuda'


'''
All models were feedforward neural networks with ReLU nonlinearities. The policy network, reward
model, and termination model each had 4 layers of size 128, while the transition model had 8 layers
of size 512. All environments were reset after 1000 timesteps. Parameters were trained with the
Adam optimizer Kingma and Ba [18] with a learning rate of 3e-4.
Policies were trained using minibatches of size 512 sampled uniformly at random from a replay buffer
of size 1e6. The first 1e5 frames were sampled via random interaction with the environment; after
that, 4 policy updates were performed for every frame sampled from the environment. (In Section
4.4, the policy updates and frames were instead de-synced.) Policy checkpoints were saved every 500
updates; these checkpoints were also frozen and used as θ. For model-based algorithms, the most
recent checkpoint of the model was loaded every 500 updates as well.
Each policy training had 8 agents interacting with the environment to send frames back to the replay
buffer. These agents typically took the greedy action predicted by the policy, but with probability
 = 0.05, instead took an action sampled from a normal distribution surrounding the pre-tanh
logit predicted by the policy. In addition, each policy had two greedy agents interacting with the
environment for evaluation.
Dynamics models were trained using minibatches of size 1024 sampled uniformly at random from
a replay buffer of size 1e6. The first 1e5 frames were sampled via random interaction with the
environment; the dynamics model was then pre-trained for 1e5 updates. After that, 4 model updates
were performed for every frame sampled from the environment. (In Section 4.4, the model updates
and frames were instead de-synced.) Model checkpoints were saved every 500 updates.
All ensembles were of size 4. During training, each ensemble member was trained on an
independently-sampled minibatch; all minibatches were drawn from the same buffer. Additionally, M, N, L = 4 for all experiments.

'''


class TransferNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(TransferNet, self).__init__()
        self.T = nn.Sequential(
            nn.Linear(s_dim+a_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, 512),
            # nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            # nn.Linear(512, 512),
            # nn.ReLU(),
            # nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(256, s_dim+1)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, s, a):
        s2_d = self.T(torch.cat([s, a], dim=1))
        return s2_d[:,:-1], self.sigmoid(s2_d[:,-1])
    
class RewardNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(RewardNet, self).__init__()
        self.R = nn.Sequential(
            nn.Linear(s_dim * 2 + a_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            # nn.Linear(128, 128),
            # nn.ReLU(),
            nn.Linear(128, 1),
        )
    def forward(self, s, a, s2):
        return torch.squeeze(self.R(torch.cat([s, a, s2], dim=1)))
        
class World(nn.Module):
    def __init__(self, s_dim, a_dim, ensem_n = 4):
        super(World, self).__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.ensem_n = ensem_n
        
        self.Ts = [TransferNet(s_dim, a_dim).to('cuda') for _ in range(ensem_n)]
        self.Rs = [RewardNet(s_dim, a_dim).to('cuda') for _ in range(ensem_n)]
        
        self.T_optims = [torch.optim.Adam(self.Ts[i].parameters(), lr=3E-4) for i in range(ensem_n)]
        self.R_optims = [torch.optim.Adam(self.Rs[i].parameters(), lr=3E-4) for i in range(ensem_n)]
        
        self.MSE_loss = torch.nn.MSELoss()
        self.CE_loss = torch.nn.CrossEntropyLoss()
        
        self.loss = [None for _ in range(self.ensem_n)]
        
    def compute_loss_T(self, s, a, s2, d, index=0):
        pred_s2, pred_d = self.Ts[index](s, a)
        return (self.MSE_loss(pred_s2, s2) + -(d*torch.log(pred_d+1E-8) + (1-d)*torch.log(1-pred_d+1E-8))).mean()
    
    def compute_loss_R(self, s, a, r, s2, index = 0):
        pred_r = self.Rs[index](s, a, s2)
        return (self.MSE_loss(pred_r, r)).mean()
    
    def train(self, batches):
        for i in range(self.ensem_n):
            
            s = batches[i]['s']
            a = batches[i]['a']
            r = batches[i]['r']
            s2 = batches[i]['s2']
            d = batches[i]['d']
            
            
            
            loss1 = self.compute_loss_T(s, a, s2, d, i)
            self.T_optims[i].zero_grad()
            loss1.backward()
            self.T_optims[i].step()
            
            
            loss2 = self.compute_loss_R(s, a, r, s2, i)
            self.R_optims[i].zero_grad()
            loss2.backward()
            self.R_optims[i].step()
            
            self.loss[i] = (loss1+loss2).detach().cpu()
        



def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianMLPActor(nn.Module):
    '''
    a(s,x) = tanh*( mu(s) + std(s) * x ), x ~ N(0,1)
    Network should output a mean array and a std array 
    '''
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super(SquashedGaussianMLPActor, self).__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit
        
    def forward(self, obs, deterministic=False, with_logprob= True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi
    
class MLPQFunction(nn.Module):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)

class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
                
        self.alpha = torch.nn.Parameter(torch.Tensor([0.2]), requires_grad=True)

    def act(self, obs, deterministic=False):
        with torch.no_grad():
            a, _ = self.pi(obs, deterministic, False)
            return a.cpu().numpy()
        
class RandomAgent:
    def __init__(self, 
                 env = gym.make('HalfCheetah-v2'), 
                ):
        self.action_space = env.action_space
    
    def act(self, s, deterministic=False):
        return self.action_space.sample()

        
class SAC():
    def __init__(
        self, 
        env,
        ensem_n = 4,
        H = 3,
        hidden_sizes=(256,256),
        gamma = 0.99,
        polyak = 0.995
    ):
        self.env = env
        self.H = H
        self.ensem_n = ensem_n
        self.gamma = gamma
        self.polyak = polyak
        
        self.act_dim = env.action_space.shape[0]
        self.history = []
        
        self.ACs = [MLPActorCritic(
            env.observation_space, 
            env.action_space,
            hidden_sizes
        ).to(device) for _ in range(ensem_n)]
        
        self.ACs_targ = [MLPActorCritic(
            env.observation_space, 
            env.action_space,
            hidden_sizes
        ).to(device)  for _ in range(ensem_n)]
        for each in self.ACs_targ:
            for p in each.parameters():
                p.requires_grad = False
        
        self.world = World(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            ensem_n
        ).to(device)
        
        self.pi_opts = [
            torch.optim.Adam(self.ACs[i].pi.parameters(), lr=3E-4) for i in range(self.ensem_n)
        ]
        self.q_opts = [
            torch.optim.Adam(self.ACs[i].q.parameters(), lr=3E-4) for i in range(self.ensem_n)
        ]
        self.alpha_opts = [
            torch.optim.Adam([self.ACs[i].alpha], lr=3E-4) for i in range(self.ensem_n)
        ]
        
        self.loss_q = None
        self.loss_pi = None
        self.ent = None
        
    def TH_MVE(self, r, s2, d, m, n, k):
        '''
        m - AC idx
        n - T idx
        k - R idx
        '''
        THs = torch.zeros((self.H+1, int(r.shape[0])), dtype=torch.float32).to(device)
        iter_v = r.clone()
        
        s_i = s2
        D_i = 1 - d
        for i in range(self.H + 1):
            a_i, logpi_i = self.ACs[m].pi(s_i, False, True)
            Q = self.ACs_targ[m].q(s_i, a_i)
            
            s_i2, d_i2 = self.world.Ts[n](s_i, a_i)
            D_i2 = D_i*(1-d_i2)
            
            if i != 0:
                iter_v += D_i*(self.gamma ** i)*self.world.Rs[k](s_i0, a_i0, s_i)
            THs[i,:] = (iter_v + D_i2*(self.gamma ** (i+1))*(Q - self.ACs_targ[m].alpha.detach()*logpi_i))
            
            s_i0 = s_i
            a_i0 = a_i
            s_i = s_i2
            D_i = D_i2
        return THs            
            
            
    def T_STEVE(self, r, s2, d):
        N = self.ensem_n ** 3
        THs = torch.zeros((N, self.H+1, int(r.shape[0])), dtype=torch.float32).to(device)        
        
        i = 0
        with torch.no_grad():
            for m in range(self.ensem_n):
                for n in range(self.ensem_n):
                    for k in range(self.ensem_n):
                        THs[i,:,:] = self.TH_MVE(r, s2, d, m, n, k)
                        i += 1
            T_mean = torch.mean(THs, dim=0)
            T_var = torch.var(THs, dim=0)
            w = 1/(T_var + 1E-8)
            T_steve = (1/w.sum(0))*(w*T_mean).sum(0)
        return T_steve
        
    def compute_loss_q(self, batches, use_model = True):
        if use_model:
            loss_q = []
            for i in range(self.ensem_n):
                s = batches[i]['s']
                a = batches[i]['a']
                r = batches[i]['r']
                s2 = batches[i]['s2']
                d = batches[i]['d']
                
                T_steve = self.T_STEVE(r, s2, d)
                q = self.ACs[i].q(s, a)
                loss_q.append(((q - T_steve)**2).mean())
        else:
            loss_q = []
            for i in range(self.ensem_n):
                s = batches[i]['s']
                a = batches[i]['a']
                r = batches[i]['r']
                s2 = batches[i]['s2']
                d = batches[i]['d']
                
                q = self.ACs[i].q(s, a)
                
                with torch.no_grad():
                    a2, logp_a2 = self.ACs[i].pi(s2)
                    for j in range(self.ensem_n):
                        if j == 0:
                            q_targ = self.ACs_targ[j].q(s2, a2)
                        else:
                            q_targ = torch.min(q_targ, self.ACs_targ[j].q(s2, a2))
                    
                    q_togo = r + self.gamma * (1 - d) * (q_targ - self.ACs[i].alpha * logp_a2)
                loss_q.append(((q - q_togo)**2).mean())
            
        return loss_q
    
    def compute_loss_pi(self, batches):
        loss_pi = []
        loss_alpha = []
        ent = []
        
        pi = [None for _ in range(self.ensem_n)]
        logpi = [None for _ in range(self.ensem_n)]
        
        for i in range(self.ensem_n):
            s = batches[i]['s']
            q_pi = torch.zeros((self.ensem_n, s.shape[0])).to(device)
            pi[i], logpi[i] = self.ACs[i].pi(s)
            for j in range(self.ensem_n):
                q_pi[j,:] = self.ACs[j].q(s, pi[i])
            q_pi = torch.min(q_pi, dim=0).values
        
            loss_pi.append((self.ACs[i].alpha.detach() * logpi[i] - q_pi).mean())
            with torch.no_grad():
                ent.append(-logpi[i].mean())
            loss_alpha.append(self.ACs[i].alpha * (ent[-1] + self.act_dim))
            
        return loss_pi, loss_alpha, ent
            

    def update(self, batches, use_model = True):
        s = []
        a = []
        r = []
        s2 = []
        d = []
        
        for i in range(self.ensem_n):
            s.append(batches[i]['s'])
            a.append(batches[i]['a'])
            r.append(batches[i]['r'])
            s2.append(batches[i]['s2'])
            d.append(batches[i]['d'])
            
        loss_q = self.compute_loss_q(batches, use_model)
        self.loss_q = [each.detach().cpu() for each in loss_q]
        for i in range(self.ensem_n):
            self.q_opts[i].zero_grad()
            loss_q[i].backward()
            self.q_opts[i].step()
                
        for ac in self.ACs:
            for p in ac.q.parameters():
                p.requires_grad = False
            
        loss_pi, loss_alpha, ent = self.compute_loss_pi(batches)
        self.loss_pi = [each.detach().cpu()for each in loss_pi]
        self.ent = [each.detach().cpu() for each in ent]
        for i in range(self.ensem_n):
            self.pi_opts[i].zero_grad()
            loss_pi[i].backward(retain_graph=True)
            self.pi_opts[i].step()
            
            # loss_alpha[i].backward()
            # self.alpha_opts[i].step()
            
        for ac in self.ACs:
            for p in ac.q.parameters():
                p.requires_grad = True
                
        with torch.no_grad():
            for ac, ac_targ in zip(self.ACs, self.ACs_targ):
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    p_targ.data.mul_(self.polyak)
                    p_targ.data.add_((1 - self.polyak) * p.data)
                    
        
    def act(self, s, deterministic=False):
        s = torch.as_tensor(s, dtype = torch.float32).to(device)
        return self.ACs[0].act(s, deterministic)
    
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
        print(f'LossQ: {self.loss_q}')
        print(f'LossPi: {self.loss_pi}')
        print(f'Ent: {self.ent}')
        print('------------------------------------\n')
        
        
# env = gym.make("HalfCheetah-v2")
# sac = SAC(
#     env.observation_space,
#     env.action_space
# )    

# batches = []
# for i in range(4):
#     s = torch.randn((32, 17))
#     a = torch.randn((32, 6))
#     s2 = torch.randn((32, 17))
#     r = torch.randn((32,))
#     d = torch.ones((32,), dtype=torch.long)

#     batch = {
#         's': s,
#         'a': a,
#         'r': r,
#         's2': s2,
#         'd': d
#     }
#     batches.append(batch)