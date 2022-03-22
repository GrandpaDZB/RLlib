import numpy as np
import torch
import torch.nn as nn
import gym
import time
import core
import psutil
import os

class ReplayBuffer:
    '''
    replay buffer storing (s, a, r, s', d)
    '''
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        
    def store(self, obs, act, rew, next_obs, done):
        '''
        If number of stored data excess the buffer size,
        the new stored date would replace the oldest stored one.
        '''
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    act=self.act_buf[idxs],
                    rew=self.rew_buf[idxs],
                    done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
    
    
    
    
def ddpg(
    env_fn = lambda : gym.make("HalfCheetah-v2"),
    actor_critic = core.MLPActorCritic,
    ac_kwargs = dict(
        hidden_sizes=(128,128,128),
        activation=nn.ReLU
    ),
    seed = 0,
    steps_per_epoch = 4000,
    epochs = 100,
    replay_size = int(1e6),
    gamma = 0.99,
    polyak = 0.995,
    pi_lr = 1e-3,
    q_lr = 1e-3,
    batch_size = 100,
    start_steps = 10000,
    update_after = 1000,
    update_every = 50,
    act_noise = 0.1,
    target_noise = 0.2,
    noise_clip = 0.5,
    policy_delay =2,
    num_test_episodes = 10,
    max_ep_len = 1000,
    save_freq = 10,
    save_path = ''
):
    '''
    polyak: param_targ = polyak * param_targ + (1 - polyak) * param 
    start_eteps: before the start_steps, actions would be sampled in action_space randomly by uniform
    update_after: update starts after it
    update_every: update runs every this num, and runs for this num
    
    
    '''

    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    
    act_limit = env.action_space.high[0]
    
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    
    history = dict(
        reward = []
    )
    
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)
        
        # Bellman backup for Q function
        with torch.no_grad():
            pi_targ = ac_targ.pi(o2)
            
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)
            
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            
            backup = r + gamma * (1 - d) * q_pi_targ
        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        return loss_q1, loss_q2
    
    def compute_loss_pi(data):
        o = data['obs']
        q1_pi = ac.q1(o, ac.pi(o))
        return -q1_pi.mean()
    
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    q1_optimizer = torch.optim.Adam(ac.q1.parameters(), lr=q_lr)
    q2_optimizer = torch.optim.Adam(ac.q2.parameters(), lr=q_lr)
    
    
    def update(data, timer):
        q1_optimizer.zero_grad()
        q2_optimizer.zero_grad()
        loss_q1, loss_q2 = compute_loss_q(data)
        loss_q1.backward()
        loss_q2.backward()
        q1_optimizer.step()
        q2_optimizer.step()
        
        # Freeze Q-network to not waste conputational effort 
        # on policy learning steps
        if timer % policy_delay == 0:
            
            for p in ac.q1.parameters():
                p.requires_grad = False
            for p in ac.q2.parameters():
                p.requires_grad = False
                
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()
            
            for p in ac.q1.parameters():
                p.requires_grad = True
            for p in ac.q2.parameters():
                p.requires_grad = True
            
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)
        return loss_q1, loss_q2
                
            
    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)
    
    def test_agent():
        mean_ep_ret = 0
        mean_ep_len = 0
        max_ep_ret = -99999
        min_ep_ret = 99999
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            mean_ep_len = (j/(j+1))*mean_ep_len + (1/(j+1))*ep_len
            mean_ep_ret = (j/(j+1))*mean_ep_ret + (1/(j+1))*ep_ret
            if ep_ret > max_ep_ret:
                max_ep_ret = ep_ret
            if ep_ret < min_ep_ret:
                min_ep_ret = ep_ret
        return mean_ep_len, mean_ep_ret, max_ep_ret, min_ep_ret
    
    total_steps = steps_per_epoch * epochs
    
    o, ep_ret, ep_len = env.reset(), 0, 0
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        d = False if ep_len==max_ep_len else d
        replay_buffer.store(o, a, r, o2, d)

        o = o2

        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0

        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                loss_q1, loss_q2 = update(data=batch, timer=j)

        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            if (epoch % save_freq == 0) or (epoch == epochs):
                torch.save(ac.state_dict(), save_path)

            # Test the performance of the deterministic version of the agent.
            mean_ep_len, mean_ep_ret, max_ret, min_ret = test_agent()   
            history['reward'].append(mean_ep_ret)
            print(f'Epoch: {epoch}')
            print('------------------------------------')
            print(f'EpRet: {mean_ep_ret}')
            print(f'EpLen: {mean_ep_len}')
            print(f'MaxRet: {max_ret}')
            print(f'MinRet: {min_ret}')
            print(f'LossQ1: {loss_q1}')
            print(f'LossQ2: {loss_q2}')
            print('------------------------------------\n')
            print('当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
    
    return history

if __name__ == "__main__":
    history = ddpg(
    env_fn = lambda : gym.make("BipedalWalkerHardcore-v3"),
    # env_fn = lambda : gym.make("HalfCheetah-v2"),
    actor_critic = core.MLPActorCritic,
    ac_kwargs = dict(
        hidden_sizes=(128,128,128),
        activation=nn.ReLU
    ),
    seed = 0,
    steps_per_epoch = 4000,
    epochs = 200,
    replay_size = int(1e6),
    gamma = 0.99,
    polyak = 0.995,
    pi_lr = 1e-3,
    q_lr = 1e-3,
    batch_size = 128,
    start_steps = 10000,
    update_after = 1000,
    update_every = 50,
    act_noise = 0.06,
    target_noise = 0.12,
    noise_clip = 0.3,
    policy_delay =2,
    num_test_episodes = 10,
    max_ep_len = 1000,
    save_freq = 10,
    save_path = './save/BipedalWalkerHardcore1.pt'
)
