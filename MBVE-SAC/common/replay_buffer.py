
import numpy as np
import torch
from . import core

device = 'cuda'

class ReplayBuffer:
    def __init__(self, env, size):
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape[0]
        self.s_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.s2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.a_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.r_buf = np.zeros(size, dtype=np.float32)
        self.d_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        
    def push(self, s, a, r, s2, d):
        self.s_buf[self.ptr] = s
        self.a_buf[self.ptr] = a
        self.r_buf[self.ptr] = r
        self.s2_buf[self.ptr] = s2
        self.d_buf[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size+1, self.max_size)
        
    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            s=self.s_buf[idxs],
            s2=self.s2_buf[idxs],
            a=self.a_buf[idxs],
            r=self.r_buf[idxs],
            d=self.d_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(device) for k,v in batch.items()}
    
    def sample_s(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        s = self.s_buf[idxs],
  
        return torch.as_tensor(np.array(s), dtype=torch.float32)
    
    def push_batch(self, s, a, r, s2, d):
        batch_size = s.shape[0]
        if batch_size + self.size >= self.max_size:
            self.ptr = 0
        else:
            self.size = min(self.size + batch_size, self.max_size)
        self.s_buf[self.ptr: batch_size + self.ptr] = s
        self.a_buf[self.ptr: batch_size + self.ptr] = a
        self.r_buf[self.ptr: batch_size + self.ptr] = torch.squeeze(r)
        self.s2_buf[self.ptr: batch_size + self.ptr] = s2
        self.d_buf[self.ptr: batch_size + self.ptr] = torch.squeeze(d)
        
        self.ptr = (self.ptr + batch_size) % self.max_size

    def clear(self):
        self.s_buf = np.zeros_like(self.s_buf)
        self.s2_buf = np.zeros_like(self.s2_buf)
        self.a_buf = np.zeros_like(self.a_buf)
        self.r_buf = np.zeros_like(self.r_buf)
        self.d_buf = np.zeros_like(self.d_buf)
        self.ptr, self.size = 0, 0