import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import scipy
import gym
import tensorflow_probability.python as tfp
import tensorflow_probability.python.distributions as tfd
import scipy.signal as signal
import buffers
import core

from time import *

EPS = 1E-8

    
def trpo(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=50,
    gamma=0.99,
    vf_lr=1e-3,
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
    tf.random.set_seed(seed)
    np.random.seed(seed)
        
    env = env_fn()
    # env = gym.make('CartPole-v1')

    obs_shape = env.observation_space.shape
    act_dim = env.action_space.shape
    
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    # ac = core.MLPActorCritic(
    #     env.observation_space,
    #     env.action_space,
    #     hidden_sizes=(128,128,128),
    #     activation=keras.activations.relu
    # )


    shape_list = [x.shape for x in ac.pi.trainable_variables]
    
    history = {}
    history["reward"] = []
    
    buf = buffers.GAEBuffer(obs_shape, act_dim, steps_per_epoch, gamma, lam)

    
    vf_optimizer = keras.optimizers.Adam(vf_lr)
    
    def update():
        data = buf.get()
        
        # compute policy gradient g
        with tf.GradientTape() as tape:
            loss_pi, kl_div = core.compute_loss_pi(data, ac)
        g = tape.gradient(loss_pi, ac.pi.trainable_variables)
        g = core.flat_concat(g)
        
        # compute Hessian vector product Hx
        Hx = lambda x : (core.flat_concat(core.hessian_vector_product(data, x, ac.pi))+damping_coeff*x).numpy()
        
        # compute x = H-1 * g with cg iteration
        x = core.cg(Hx, g, cg_iter)
        deltaTheta = np.sqrt(2*delta/np.abs(np.dot(x, Hx(tf.convert_to_tensor(x)))+EPS))*x
        
        # Do line search and decide whether update or not
        do_update = False
        numpy_g = g.numpy()
        for j in range(line_search_iter):
            tmp_dTheta = deltaTheta*(line_search_coeff**j)
            L = np.dot(numpy_g, tmp_dTheta)
            Dkl_mean = 0.5*np.dot(tmp_dTheta, Hx(tf.convert_to_tensor(tmp_dTheta)))
            if L >= 0 and abs(Dkl_mean) <= delta and Dkl_mean is not None:
                do_update = True
                deltaTheta = tmp_dTheta
                print(f'Accepting new params at step {j+1} of line search.')
                break
        else:
            print('Line search failed! Keeping old params.')
            
        if do_update:
            new_theta = core.flat_concat(ac.pi.trainable_variables) + tf.convert_to_tensor(deltaTheta)
            new_theta = core.concat_flat(new_theta, shape_list)
            ac.pi.set_weights(new_theta)
            

        for _ in range(train_v_iters):
            with tf.GradientTape() as tape:
                loss_v = core.compute_loss_v(data, ac)
            g = tape.gradient(loss_v, ac.v.trainable_variables)    
            vf_optimizer.apply_gradients(zip(g, ac.v.trainable_variables))

        return kl_div, loss_v, Dkl_mean

    o, ep_ret, ep_len = env.reset(), 0, 0
    o = core.obs_preprocess(o, obs_shape)
    for epoch in range(epochs):
        mean_ep_ret = 0
        ep_ret_idx = 0
        mean_ep_len = 0
        ep_len_idx = 0
        
        max_ret = -99999
        min_ret = 999999
        
        for t in range(steps_per_epoch):
            # ts = time()
            a, v, logp = ac.step(o)
            
            # print(1)
            # print(time()-ts)
            # ts = time()
            
            next_o, r, d, _ = env.step(a)
            next_o = core.obs_preprocess(next_o, obs_shape)
            ep_ret += r
            ep_len += 1
            
            # print(2)
            # print(time()-ts)
            # ts = time()
            
            buf.store(o,a,r,v,logp)
            
            # print(3)
            # print(time()-ts)
            # ts = time()
            
            o = next_o
            
            timeout = (ep_len == max_ep_len)
            terminal = (d or timeout)
            epoch_ended = (t == steps_per_epoch-1)
            
            # print(4)
            # print(time()-ts)
            # ts = time()
            
            if terminal or epoch_ended:
                if epoch_ended and not terminal:
                    pass
                if timeout or epoch_ended:
                    _, v, _ = ac.step(o)
                else:
                    v = 0
                buf.finish_path(v)
                mean_ep_ret = (ep_ret_idx/(ep_ret_idx+1))*mean_ep_ret + (1/(ep_ret_idx+1))*ep_ret
                mean_ep_len = (ep_len_idx/(ep_len_idx+1))*mean_ep_len + (1/(ep_len_idx+1))*ep_len
                ep_ret_idx += 1
                ep_len_idx += 1
                if ep_ret > max_ret:
                    max_ret = ep_ret
                if ep_ret < min_ret:
                    min_ret = ep_ret
                o, ep_ret, ep_len = env.reset(), 0, 0
                o = core.obs_preprocess(o, obs_shape)
                
        kl, loss_v, KL_theta = update()
        
        print(f'Epoch: {epoch+1}')
        print('------------------------------------')
        print(f'EpRet: {mean_ep_ret}')
        print(f'EpLen: {mean_ep_len}')
        print(f'KL: {kl}')
        print(f'KLtheta: {KL_theta}')
        print(f'LossV: {loss_v}')
        print(f'MaxRet: {max_ret}')
        print(f'MinRet: {min_ret}')
        print('------------------------------------\n')
        history['reward'].append(mean_ep_ret)
        
        if epoch%save_freq == 0 and save_path != "":
            ac.save_weights(save_path)
        
    return history


if __name__ == "__main__":
    historty = trpo(
        lambda : gym.make("CartPole-v1"),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(
            hidden_sizes=(128, 128, 128),
            activation=keras.activations.relu
        ),
        seed=0,
        steps_per_epoch=5000,
        epochs=1,
        gamma=0.99,
        vf_lr=5e-4,
        train_v_iters=80,
        lam=0.97,
        max_ep_len=500,
        delta = 0.01,
        damping_coeff=0.1,
        cg_iter = 20,
        line_search_iter = 10,
        line_search_coeff = 0.8,
        save_path="",
        save_freq=10
    )
    
    # env = gym.make("CartPole-v1")
    # # ac = core.MLPActorCritic(
    # #     env.observation_space,
    # #     env.action_space
    # # )
    # o = env.reset()
    # for _ in range(5000):
    #     o, _, d, _ = env.step(env.action_space.sample())
    #     o = core.obs_preprocess(o, (4,))
    #     if d:
    #         env.reset()