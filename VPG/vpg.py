
import torch
import torch.nn as nn
import numpy as np
import gym
import core



class VPGBuffer:
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
        
        data = dict(
            obs = torch.as_tensor(self.obs_buf, dtype=torch.float32),
            act = torch.as_tensor(self.act_buf, dtype=torch.float32),
            ret = torch.as_tensor(self.ret_buf, dtype=torch.float32),
            adv = torch.as_tensor(self.adv_buf, dtype=torch.float32),
            logp = torch.as_tensor(self.logp_buf, dtype=torch.float32)
        )
        return data
    
def vpg(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=50,
    gamma=0.99,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_v_iters=80,
    lam=0.97,
    max_ep_len=1000,
    save_freq=10,
    save_path=''
):
    """
    Vanilla Policy Gradient 

    (with GAE-Lambda for advantage estimation)

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to VPG.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    torch.manual_seed(seed)
    np.random.seed(seed)
        
    env = env_fn()

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    
    history = {}
    history["reward"] = []
    
    buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)
    
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        pi, logp = ac.pi(obs, act)
        loss_pi = -(logp*adv).mean()
        
        # useful extra info
        approx_kl = (logp_old-logp).mean().item()
        ent = pi.entropy().mean().item()
        pi_info = dict(
            kl=approx_kl,
            ent=ent
        )
        
        return loss_pi, pi_info
    
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()
    
    pi_optimizer = torch.optim.Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=vf_lr)
    
    def update():
        data = buf.get()

        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()
        
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()
        kl, ent = pi_info['kl'], pi_info['ent']
        return kl, ent, loss_pi, loss_v

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
            
            buf.store(o,a,r,v,logp)
            
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
                
        kl, ent, loss_pi, loss_v = update()
        
        print(f'Epoch: {epoch+1}')
        print('------------------------------------')
        print(f'EpRet: {mean_ep_ret}')
        print(f'EpLen: {mean_ep_len}')
        print(f'KL: {kl}')
        print(f'Ent: {ent}')
        print(f'LossPi: {loss_pi}')
        print(f'LossV: {loss_v}')
        print(f'MaxRet: {max_ret}')
        print(f'MinRet: {min_ret}')
        print('------------------------------------\n')
        history['reward'].append(mean_ep_ret)
        
        if epoch%save_freq == 0 and save_path != "":
            torch.save(ac.state_dict(), save_path)
        
    return history
        
        
if __name__ == "__main__":
    historty = vpg(
        lambda : gym.make("LunarLander-v2"),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(
            hidden_sizes=(64, 64, 64),
            activation=nn.Tanh
        ),
        gamma=0.99,
        seed=0,
        steps_per_epoch=6000,
        epochs=300,
        lam=0.97,
        train_v_iters=80,
    )