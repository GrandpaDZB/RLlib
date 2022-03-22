
import torch
import torch.nn as nn
import numpy as np
import gym
import core
import psutil
import os


EPS = 1E-8
use_cuda = False

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
        
        if use_cuda:
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

    
def trpo(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=50,
    gamma=0.99,
    vf_lr=1e-3,
    train_pi_iters=80,
    train_v_iters=80,
    lam=0.97,
    max_ep_len=1000,
    delta = 0.01,
    damping_coeff=0.1,
    cg_iter = 20,
    line_search_iter = 10,
    line_search_coeff = 0.8,
    save_path = "",
    save_freq = 10
):
    """
    Trust Region Policy Optimization 

    (with support for Natural Policy Gradient)

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ============  ================  ========================================
            Symbol        Shape             Description
            ============  ================  ========================================
            ``pi``        (batch, act_dim)  | Samples actions from policy given 
                                            | states.
            ``logp``      (batch,)          | Gives log probability, according to
                                            | the policy, of taking actions ``a_ph``
                                            | in states ``x_ph``.
            ``logp_pi``   (batch,)          | Gives log probability, according to
                                            | the policy, of the action sampled by
                                            | ``pi``.
            ``info``      N/A               | A dict of any intermediate quantities
                                            | (from calculating the policy or log 
                                            | probabilities) which are needed for
                                            | analytically computing KL divergence.
                                            | (eg sufficient statistics of the
                                            | distributions)
            ``info_phs``  N/A               | A dict of placeholders for old values
                                            | of the entries in ``info``.
            ``d_kl``      ()                | A symbol for computing the mean KL
                                            | divergence between the current policy
                                            | (``pi``) and the old policy (as 
                                            | specified by the inputs to 
                                            | ``info_phs``) over the batch of 
                                            | states given in ``x_ph``.
            ``v``         (batch,)          | Gives the value estimate for states
                                            | in ``x_ph``. (Critical: make sure 
                                            | to flatten this!)
            ============  ================  ========================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to TRPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        delta (float): KL-divergence limit for TRPO / NPG update. 
            (Should be small for stability. Values like 0.01, 0.05.)

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        damping_coeff (float): Artifact for numerical stability, should be 
            smallish. Adjusts Hessian-vector product calculation:
            
            .. math:: Hv \\rightarrow (\\alpha I + H)v

            where :math:`\\alpha` is the damping coefficient. 
            Probably don't play with this hyperparameter.

        cg_iters (int): Number of iterations of conjugate gradient to perform. 
            Increasing this will lead to a more accurate approximation
            to :math:`H^{-1} g`, and possibly slightly-improved performance,
            but at the cost of slowing things down. 

            Also probably don't play with this hyperparameter.

        backtrack_iters (int): Maximum number of steps allowed in the 
            backtracking line search. Since the line search usually doesn't 
            backtrack, and usually only steps back once when it does, this
            hyperparameter doesn't often matter.

        backtrack_coeff (float): How far back to step during backtracking line
            search. (Always between 0 and 1, usually above 0.5.)

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

    """
    torch.manual_seed(seed)
    np.random.seed(seed)
        
    env = env_fn()

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    test_ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    shape_list = [x.shape for x in ac.pi.state_dict().values()]
    key_list = ac.pi.state_dict().keys()
    
    history = {}
    history["reward"] = []
    
    buf = GAEBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)
    
    def cg(Ax, b, cg_iter):
        '''
        Conjugate gradient algorithm\n
        for a given Ax=b, approximate x
        '''
        if use_cuda:
            b = b.detach().cpu().numpy()
        else:
            b = b.detach().numpy()
        x = np.zeros_like(b)
        r = b.copy()
        p = r.copy()
        r_dot_old = np.dot(r,r)
        for _ in range(cg_iter):
            z = Ax(torch.as_tensor(p))
            alpha = r_dot_old / (np.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r, r)
            p = r + (r_dot_new / (r_dot_old+EPS)) * p
            r_dot_old = r_dot_new
        return x

    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        pi, logp = ac.pi(obs, act)
        loss_pi = (logp*adv).mean()
        
        # useful extra info
        kl_div = (logp_old-logp).mean()
        #ent = pi.entropy().mean().item()
        
        return loss_pi, kl_div, pi.entropy()
    
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()
    
    def compute_new_pi_loss(deltaTheta, data, flatten_old_theta):
        if use_cuda:
            new_theta = flatten_old_theta + torch.as_tensor(deltaTheta).to('cuda')
        else:
            new_theta = flatten_old_theta + torch.as_tensor(deltaTheta)
        new_theta = core.concat_flat_for_nn(new_theta, shape_list, key_list)
        test_ac.pi.load_state_dict(new_theta)
        
        obs, act, adv = data['obs'], data['act'], data['adv']
        _, logp = test_ac.pi(obs, act)
        loss_pi_new = (logp*adv).mean() 

        return loss_pi_new, new_theta
    
    vf_optimizer = torch.optim.Adam(ac.v.parameters(), lr=vf_lr)
    
    def update():
        data = buf.get()
        
        loss_pi, kl_div, ent = compute_loss_pi(data)
        
        # compute policy gradient g
        g_concat = torch.autograd.grad(loss_pi, ac.pi.parameters(), create_graph=True)
        g = core.flat_concat(g_concat)
        
        # compute Hessian vector product Hx
        if use_cuda:
            Hx = lambda x : (core.flat_concat(core.hessian_vector_product(kl_div, x.to('cuda'), ac.pi))+damping_coeff*x.to('cuda')).detach().cpu().numpy()
        else:
            Hx = lambda x : (core.flat_concat(core.hessian_vector_product(kl_div, x, ac.pi))+damping_coeff*x).detach().numpy()
        # compute x = H-1 * g with cg iteration
        x = cg(Hx, g, cg_iter)
        deltaTheta = np.sqrt(2*delta/np.abs(np.dot(x, Hx(torch.as_tensor(x)))+EPS))*x
        
        # Do line search and decide whether update or not
        

        do_update = False
        flatten_old_theta = core.flat_concat(ac.pi.parameters())
        for j in range(line_search_iter):
            tmp_dTheta = deltaTheta*(line_search_coeff**j)
            loss_pi_new, new_state_dict = compute_new_pi_loss(tmp_dTheta, data, flatten_old_theta)
            Dkl_mean = 0.5*np.dot(tmp_dTheta, Hx(torch.as_tensor(tmp_dTheta)))
            if loss_pi_new > loss_pi and abs(Dkl_mean) <= delta and Dkl_mean is not None:
                do_update = True
                deltaTheta = tmp_dTheta
                print(f'Accepting new params at step {j+1} of line search.')
                break
        else:
            print('Line search failed! Keeping old params.')
            
        if do_update:
            ac.pi.load_state_dict(new_state_dict)


        for _ in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            vf_optimizer.step()

        return kl_div, loss_v, Dkl_mean, ent

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
                if epoch_ended and not terminal:
                    pass
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
                
        kl, loss_v, KL_theta, ent = update()
        
        print(f'Epoch: {epoch+1}')
        print('------------------------------------')
        print(f'EpRet: {mean_ep_ret}')
        print(f'EpLen: {mean_ep_len}')
        print(f'KL: {kl}')
        print(f'Entropy: {ent.mean()}')
        print(f'KLtheta: {KL_theta}')
        print(f'LossV: {loss_v}')
        print(f'MaxRet: {max_ret}')
        print(f'MinRet: {min_ret}')
        print('------------------------------------\n')
        print('当前进程的内存使用：%.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
        history['reward'].append(mean_ep_ret)
        
        if epoch%save_freq == 0 and save_path != "":
            torch.save(ac.state_dict(), save_path)
        
    return history
        
        
        
if __name__ == "__main__":
    historty = trpo(
        lambda : gym.make("LunarLander-v2"),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(
            hidden_sizes=(128, 128, 128),
            activation=nn.ReLU
        ),
        seed=0,
        steps_per_epoch=6000,
        epochs=200,
        gamma=0.99,
        vf_lr=1e-3,
        train_v_iters=80,
        lam=0.96,
        max_ep_len=200,
        delta = 0.01,
        damping_coeff=0.1,
        cg_iter = 20,
        line_search_iter = 1,
        line_search_coeff = 0.8,
        save_path="./save/LunarLander1.pt",
        save_freq=10
    )
    
    # historty = trpo(
    #     lambda : gym.make("Qbert-v0"),
    #     actor_critic=core.CNNActorCritic,
    #     ac_kwargs=dict(),
    #     seed=0,
    #     steps_per_epoch=1000,
    #     epochs=100,
    #     gamma=0.99,
    #     vf_lr=5e-4,
    #     train_v_iters=60,
    #     lam=0.96,
    #     max_ep_len=200,
    #     delta = 0.01,
    #     damping_coeff=0.1,
    #     cg_iter = 20,
    #     line_search_iter = 1,
    #     line_search_coeff = 0.8,
    #     save_path="./save/Qbert1.pt",
    #     save_freq=10
    # )