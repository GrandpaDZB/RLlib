import tensorflow as tf
import numpy as np
import tensorflow.keras as keras
import scipy
import gym
import tensorflow_probability.python as tfp
import tensorflow_probability.python.distributions as tfd
import scipy.signal as signal
from time import *


EPS = 1E-8

def obs_preprocess(obs, obs_shape):
    return np.reshape(obs, (1, *obs_shape)).astype(np.float32)
    

def combined_shape(length, shape=None):
    '''
    combined_shape(5) => (5,)\n
    combined_shape(5,3) => (5,3)\n
    combined_shape(5,(11,2)) => (5,11,2)\n
    '''
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation):    
    layers = [
        keras.layers.Dense(x, activation=activation) for x in sizes
    ]
    return keras.Sequential(layers)


def discount_cumsum(x, discount):
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Actor(keras.Model):
    def _distribution(self, obs):
        raise NotImplementedError()
    
    def _log_prob_from_distributions(self, pi, act):
        raise NotImplementedError()
    
    def call(self, obs, act=None):
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distributions(pi, act)
        return pi, logp_a
    
class MLPCategoricalActor(Actor):
    def __init__(self, obs_shape, act_dim, hidden_sizes, activation):
        super(MLPCategoricalActor, self).__init__()
        self.logits_net = mlp(list(hidden_sizes)+[act_dim], activation)
        self.logits_net(tf.random.normal((1, *obs_shape)))
        
        
    def _distribution(self, obs):
        ts = time()
        
        # logits = self.logits_net.call(obs, training=True)
        logits = self.logits_net(obs)
        
        print(f'{1}\t{time()-ts}')
        ts = time()
        
        d = tfd.Categorical(logits=logits)
        
        print(f'{2}\t{time()-ts}')
        ts = time()
        
        return d
    
    def _log_prob_from_distributions(self, pi, act):
        return pi.log_prob(act)
    
class MLPGaussianActor(Actor):
    def __init__(self, obs_shape, act_dim, hidden_sizes, activation):
        super(MLPGaussianActor, self).__init__()
        self.log_std = tf.Variable(tf.ones(act_dim, dtype=tf.float32))
        self.mu_net = mlp(list(hidden_sizes)+[act_dim], activation)
        self.mu_net(tf.random.normal((1, *obs_shape)))
        
    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = tf.exp(self.log_std)
        d = tfd.Normal(mu, std)
        return d
    
    def _log_prob_from_distributions(self, pi, act):
        return tf.reduce_sum(pi.log_prob(act))
    
class MLPCritic(keras.Model):
    def __init__(self, obs_shape, hidden_sizes, activation):
        super(MLPCritic, self).__init__()
        self.v_net = mlp(list(hidden_sizes)+[1], activation)
        self.v_net(tf.random.normal((1, *obs_shape)))
        
    def call(self, obs):
        return tf.squeeze(self.v_net(obs), -1)
    
class MLPActorCritic(keras.Model):
    def __init__(self, observation_space, action_space, hidden_sizes=(128, 128, 128), activation=keras.activations.relu):
        super(MLPActorCritic, self).__init__()
        obs_shape = observation_space.shape
        
        if isinstance(action_space, gym.spaces.Box):
            self.pi = MLPGaussianActor(obs_shape, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, gym.spaces.Discrete):
            self.pi = MLPCategoricalActor(obs_shape, action_space.n, hidden_sizes, activation)
        else:
            raise TypeError('Unknown action_space type') 
        self.v = MLPCritic(obs_shape, hidden_sizes, activation)

    def step(self, obs):
        # self.pi.trainable = False
        # self.v.trainable = False
        # ts = time()
        
        pi = self.pi._distribution(obs)
        
        # print(time()-ts)
        # print(1)
        # ts = time()
        
        a = pi.sample()
        
        
        # print(time()-ts)
        # print(2)
        # ts = time()
        
        logp_a = self.pi._log_prob_from_distributions(pi, a)
        
        
        # print(time()-ts)
        # print(3)
        # ts = time()
        
        v = self.v(obs)
        
        
        # print(time()-ts)
        # print(4)
        # ts = time()
        
        return a.numpy()[0], v.numpy(), logp_a.numpy()
    
    def act(self, obs):
        return self.step(obs)[0]
    




def flat_concat(xs):
    return tf.concat([tf.reshape(x, (-1,)) for x in xs], axis=0)


def hessian_vector_product(data, v, ac_pi):
    obs, act, logp_old = data['obs'], data['act'], data['logp']
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            _, logp = ac_pi(obs, act)
            kl_div_policy = tf.reduce_mean(logp_old - logp)
        g = tape1.gradient(kl_div_policy, ac_pi.trainable_variables)
        g = flat_concat(g)
        prod = tf.reduce_sum(g*v)
    h = tape2.gradient(prod, ac_pi.trainable_variables)
    return h

def cg(Hx, b, cg_iters):
    b = b.numpy()
    x = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    r_dot_old = np.dot(r,r)
    for _ in range(cg_iters):
        z = Hx(tf.convert_to_tensor(p))
        alpha = r_dot_old / (np.dot(p,z) + EPS)
        x += alpha * p
        r -= alpha * z 
        r_dot_new = np.dot(r, r)
        p = r + (r_dot_new / (r_dot_old + EPS)) * p
        r_dot_old = r_dot_new
    return x 

def compute_loss_pi(data, ac):
    obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
    _, logp = ac.pi(obs, act)
    loss_pi = tf.reduce_mean(logp*adv)
    
    # useful extra info
    approx_kl = tf.reduce_mean(logp_old-logp)
    #ent = pi.entropy().mean().item()
    return loss_pi, approx_kl

def compute_loss_v(data, ac):
    obs, ret = data['obs'], data['ret']
    return tf.reduce_mean((ac.v(obs) - ret)**2)

def concat_flat(x, shape_list):
    ls_list = [x.as_list() for x in shape_list]
    num_list = []
    for each in ls_list:
        tmp = 1
        for each2 in each:
            tmp *= each2
        num_list.append(tmp)
    pre = tf.split(x, num_list)
    concat_x = [tf.reshape(x, s) for x, s in zip(pre, shape_list)]
    return concat_x

