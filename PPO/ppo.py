import torch
import torch.nn as nn
import numpy as np
import scipy
import gym
import scipy.signal as signal
import time
import core
import psutil
import os
import mpi
import sys

class GAEBuffer:
    """
    A buffer for storing trajectories experienced by a VPG agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        
    def store(self, obs, act, rew, val, logp):
        '''
        Append one timestep of agent-environment interaction to the buffer
        '''
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1
        
    def finish_path(self, last_val = 0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """
        # from path_start_idx to ptr, excluding ptr
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # compute the GAE-Lambda advantage function
        deltas = rews[:-1] + self.gamma*vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma*self.lam)
        
        # compute the reward-to-go
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr
        
    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        
        # normalize adv buffer
        adv_mean = np.mean(self.adv_buf)
        adv_std = np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf-adv_mean)/adv_std
        
        if core.use_cuda:
            data = dict(
                obs = torch.as_tensor(self.obs_buf, dtype=torch.float32).to('cuda'),
                act = torch.as_tensor(self.act_buf, dtype=torch.float32).to('cuda'),
                ret = torch.as_tensor(self.ret_buf, dtype=torch.float32).to('cuda'),
                adv = torch.as_tensor(self.adv_buf, dtype=torch.float32).to('cuda'),
                logp = torch.as_tensor(self.logp_buf, dtype=torch.float32).to('cuda')
            )
        else:
            data = dict(
                obs = torch.as_tensor(self.obs_buf, dtype=torch.float32),
                act = torch.as_tensor(self.act_buf, dtype=torch.float32),
                ret = torch.as_tensor(self.ret_buf, dtype=torch.float32),
                adv = torch.as_tensor(self.adv_buf, dtype=torch.float32),
                logp = torch.as_tensor(self.logp_buf, dtype=torch.float32)
            )
        return data
    
    
    
def ppo(
    env_fn = lambda : gym.make('CartPole-v1'), 
    actor_critic=core.MLPActorCritic, 
    ac_kwargs=dict(
        hidden_sizes=(128,128,128),
        activation=nn.ReLU    
    ), 
    seed=0, 
    steps_per_epoch=4000, 
    epochs=50, 
    gamma=0.99, 
    clip_ratio=0.2, 
    pi_lr=3e-4,
    vf_lr=1e-3, 
    train_pi_iters=80, 
    train_v_iters=80, 
    lam=0.97, 
    max_ep_len=1000,
    target_kl=0.01, 
    save_freq=10,
    save_path=''
    ):
    
    mpi.setup_pytorch_for_mpi()
    seed += 10000 * mpi.proc_id()
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = env_fn()
    obs_shape = env.observation_space.shape
    act_dim = env.action_space.shape
    
    if core.use_cuda:
        ac = actor_critic(
            env.observation_space,
            env.action_space,
            **ac_kwargs
        ).to('cuda')
    else:
        ac = actor_critic(
            env.observation_space,
            env.action_space,
            **ac_kwargs
        )
    mpi.sync_params(ac)
    
    buf = GAEBuffer(
        obs_shape,
        act_dim,
        steps_per_epoch,
        gamma,
        lam
    )
    
    history = {}
    history['reward'] = []
    
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio*adv, clip_adv)).mean()
        
        KL_div = (logp_old - logp).mean()
        Entropy = pi.entropy().mean()
        
        return loss_pi, KL_div, Entropy
    
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()
    
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=vf_lr)
    
    def update():
        data = buf.get()
        
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, KL_div_policy, Entropy = compute_loss_pi(data)
            
            mpi_kl = mpi.mpi_avg(KL_div_policy.detach().numpy())
            
            if mpi_kl > 1.5 * target_kl:
                # print(f'Early stopping at step {i} due to reaching max kl.')
                break
            loss_pi.backward()
            
            mpi.mpi_avg_grads(ac.pi)
            
            pi_optimizer.step()
        
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            
            mpi.mpi_avg_grads(ac.v)
            
            vf_optimizer.step()
            
        return loss_pi, loss_v, mpi_kl, Entropy
            
            
    o, ep_ret, ep_len = env.reset(), 0, 0
    for epoch in range(epochs):
        mean_ep_ret = 0
        ep_ret_idx = 0
        mean_ep_len = 0
        ep_len_idx = 0
        
        max_ret = -99999
        min_ret = 999999
        for t in range(steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))
            
            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1
            
            buf.store(o, a, r, v, logp)
            
            o = next_o
            
            timeout = (ep_len == max_ep_len)
            terminal = (d or timeout)
            epoch_ended = (t == steps_per_epoch-1)
            
            if terminal or epoch_ended:
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                buf.finish_path(v)
                if terminal:
                    mean_ep_ret = (ep_ret_idx/(ep_ret_idx+1))*mean_ep_ret + (1/(ep_ret_idx+1))*ep_ret
                    mean_ep_len = (ep_len_idx/(ep_len_idx+1))*mean_ep_len + (1/(ep_len_idx+1))*ep_len
                    ep_ret_idx += 1
                    ep_len_idx += 1
                    if ep_ret > max_ret:
                        max_ret = ep_ret
                    if ep_ret < min_ret:
                        min_ret = ep_ret
                o, ep_ret, ep_len = env.reset(), 0, 0
                

        loss_pi, loss_v, kl_div, entropy = update()
        mpi.sync_params(ac)
        
    
        if mpi.proc_id() == 0:
            print(f'Epoch: {epoch+1}')
            print('------------------------------------')
            print(f'EpRet: {mean_ep_ret}')
            print(f'EpLen: {mean_ep_len}')
            print(f'KL: {kl_div}')
            print(f'Entropy: {entropy}')
            print(f'LossPi: {loss_pi}')
            print(f'LossV: {loss_v}')
            print(f'MaxRet: {max_ret}')
            print(f'MinRet: {min_ret}')
            print('------------------------------------\n')
            print('当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
            history['reward'].append(mean_ep_ret)
            sys.stdout.flush()
            
            if epoch%save_freq == 0 and save_path != "":
                torch.save(ac.state_dict(), save_path)

        
    return history
        
        
    
if __name__ == "__main__":
    
    
    history = ppo(
        env_fn = lambda : gym.make('BipedalWalkerHardcore-v2'), 
        actor_critic=core.MLPActorCritic, 
        ac_kwargs=dict(
            hidden_sizes = (128, 128, 128),
            activation = nn.ReLU
        ), 
        seed=0, 
        steps_per_epoch=4000, 
        max_ep_len=500,
        epochs=250, 
        gamma=0.995, 
        clip_ratio=0.2, 
        pi_lr=3e-4,
        vf_lr=8e-4, 
        train_pi_iters=80, 
        train_v_iters=80, 
        lam=0.97, 
        target_kl=0.01, 
        save_freq=10,
        save_path='./save/BipedalWalkerHardcore1.pt'
    )
    
    # history = ppo(
    #     env_fn = lambda : gym.make('Qbert-v4'), 
    #     actor_critic=core.CNNActorCritic, 
    #     ac_kwargs=dict(), 
    #     seed=0, 
    #     steps_per_epoch=1000, 
    #     max_ep_len=2000,
    #     epochs=150, 
    #     gamma=0.99, 
    #     clip_ratio=0.2, 
    #     pi_lr=3e-4,
    #     vf_lr=5e-4, 
    #     train_pi_iters=80, 
    #     train_v_iters=100, 
    #     lam=0.97, 
    #     target_kl=0.01, 
    #     save_freq=10,
    #     save_path='./save/Qbert1.pt'
    # )
