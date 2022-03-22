
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
    
    
    
def sac(
    env_fn = lambda : gym.make("HalfCheetah-v2"),
    actor_critic=core.MLPActorCritic, 
    ac_kwargs = dict(
        hidden_sizes=(128,128,128),
        activation=nn.ReLU
    ),
    seed=0,     
    steps_per_epoch=4000, 
    epochs=100, 
    replay_size=int(1e6), 
    gamma=0.99, 
    polyak=0.995, 
    lr=1e-3, 
    batch_size=100, 
    start_steps=10000, 
    update_after=1000, 
    update_every=50, 
    num_test_episodes=10, 
    max_ep_len=1000, 
    save_freq = 10,
    save_path = ''
    ):
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    history = dict(
        reward = []
    )

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    
    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
    
    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        '''
        Q* = r + gamma * (1 - d) * (minQ_targ(s', a') - alpha * logpi(a'|s'))
        a' here is sampled from pi instead of the one from replay buffer
        '''
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - ac.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        return loss_q1, loss_q2

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        '''
        maximize E[ minQ(s,a') - alpha * log pi(a'|s) ]
        '''
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (ac.alpha * logp_pi - q_pi).mean()
        ent = -logp_pi.mean()
        
        loss_alpha = ac.alpha*ent.clone().detach() + ac.alpha*act_dim

        return loss_pi, loss_alpha, ent
    
    def compute_loss_alpha(ent):
        loss_alpha = -ac.alpha*ent - ac.alpha*(0.5)
        return loss_alpha

    # Set up optimizers for policy and q-function
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=lr)
    q1_optimizer = torch.optim.Adam(ac.q1.parameters(), lr=lr)
    q2_optimizer = torch.optim.Adam(ac.q2.parameters(), lr=lr)
    alpha_optimizer = torch.optim.Adam([ac.alpha], lr=lr)


    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q1_optimizer.zero_grad()
        q2_optimizer.zero_grad()
        loss_q1, loss_q2 = compute_loss_q(data)
        loss_q1.backward()
        loss_q2.backward()
        q1_optimizer.step()
        q2_optimizer.step()


        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in ac.q1.parameters():
            p.requires_grad = False
        for p in ac.q2.parameters():
            p.requires_grad = False
            

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, loss_alpha, ent = compute_loss_pi(data)
        loss_pi.backward(retain_graph=True)
        pi_optimizer.step()
        
        alpha_optimizer.zero_grad()
        loss_alpha.backward()
        alpha_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in ac.q1.parameters():
            p.requires_grad = True
        for p in ac.q2.parameters():
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)
        return loss_pi, loss_q1, loss_q2, ent 

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)

    def test_agent():
        mean_ep_ret = 0
        mean_ep_len = 0
        max_ep_ret = -99999
        min_ep_ret = 99999
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            mean_ep_len = (j/(j+1))*mean_ep_len + (1/(j+1))*ep_len
            mean_ep_ret = (j/(j+1))*mean_ep_ret + (1/(j+1))*ep_ret
            if ep_ret > max_ep_ret:
                max_ep_ret = ep_ret
            if ep_ret < min_ep_ret:
                min_ep_ret = ep_ret
        return mean_ep_len, mean_ep_ret, max_ep_ret, min_ep_ret
    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs

    o, ep_ret, ep_len = env.reset(), 0, 0
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)
        k = 1 + replay_buffer.size/replay_buffer.max_size

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(int(update_every * k)):
                batch = replay_buffer.sample_batch(
                    int(batch_size * k)
                )
                loss_pi, loss_q1, loss_q2, entropy = update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch
            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                torch.save(ac.state_dict(), save_path)

            # Test the performance of the deterministic version of the agent.
            mean_ep_len, mean_ep_ret, max_ret, min_ret = test_agent()

            # Log info about epoch
            history['reward'].append(mean_ep_ret)
            print(f'Epoch: {epoch}')
            print('------------------------------------')
            print(f'Total step: {t+1}')
            print(f'EpRet: {mean_ep_ret}')
            print(f'EpLen: {mean_ep_len}')
            print(f'MaxRet: {max_ret}')
            print(f'MinRet: {min_ret}')
            print(f'Entropy: {entropy}')
            print(f'Temperature: {ac.alpha.data}')
            print(f'LossQ1: {loss_q1}')
            print(f'LossQ2: {loss_q2}')
            print('------------------------------------\n')
            print('当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
            
if __name__ == "__main__":
    history = sac(
        env_fn = lambda : gym.make("Humanoid-v2"),
        actor_critic=core.MLPActorCritic, 
        ac_kwargs = dict(
            hidden_sizes=(256,256),
            activation=nn.ReLU
        ),
        seed=0,     
        steps_per_epoch=4000, 
        epochs=250, 
        replay_size=int(1e6), 
        gamma=0.99, 
        polyak=0.995, 
        lr=8e-4, 
        batch_size=256, 
        start_steps=10000, 
        update_after=1000, 
        update_every=50, 
        num_test_episodes=10, 
        max_ep_len=1000, 
        save_freq = 10,
        save_path = ''
    )
