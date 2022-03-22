import numpy as np

class EnvSampler():
    def __init__(self, env, max_ep_len=1000):
        self.env = env
        self.t = 0
        self.s = None
        self.max_ep_len = max_ep_len
        self.mean_ep_len = 0
        
        
    def sample(self, agent, deterministic=False, rescale_factor = 1.0):
        if self.s is None:
            self.s = self.env.reset()
        
        s = self.s
        
        if np.random.rand() > 0.1:
            a = agent.act(s, deterministic)
        else:
            a = self.env.action_space.sample()

        s2, r, d, _ = self.env.step(a)
        
        self.t = self.t + 1
        d = False if self.t == self.max_ep_len else d
        
        
        

        # if r == -100:
        #     r, d = -1, True
        r *= rescale_factor
            
            
        
        
        if d or self.t >= self.max_ep_len:
            self.s = None
            self.t = 0
        else:
            self.s = s2
            
        return s, a, r, s2, d
    
    def sample_and_push(self, pool, agent, deterministic=False, rescale_factor= 1.0):
        s, a, r, s2, d = self.sample(agent, deterministic, rescale_factor)
        pool.push(s, a, r, s2, d)