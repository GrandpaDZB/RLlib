import gym

class RandomAgent:
    def __init__(self, 
                 env = gym.make('HalfCheetah-v2'), 
                ):
        self.action_space = env.action_space
    
    def act(self, s, deterministic=False, std_scale = 1.0):
        return self.action_space.sample()
    
    
class RLAgent:
    def __init__(
        self,
        env = gym.make('HalfCheetah-v2'), 
    ):
        self.env = env
        pass
    
    def act(self, s):
        pass
    
    def update(self, batch):
        pass
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass
    
    def test(self, path):
        pass
    
    def print_log(self):
        pass