
import torch
import torch.nn as nn
from common.env_sampler import EnvSampler
from common.replay_buffer import ReplayBuffer
from common.networks import SAC
from common.networks import RandomAgent
import gym
import sys

env_fn = lambda : gym.make("Humanoid-v2")
env = env_fn()


env_sampler = EnvSampler(
    env_fn(),
)

env_pool = ReplayBuffer(
    env_fn(),
    int(1E6)
)



sac = SAC(
    env_fn()
)
rand_agent = RandomAgent(
    env_fn()
)

# sac.world.load_state_dict(torch.load("./world_pretrain.pt"))

for t in range(int(1E5)):
    env_sampler.sample_and_push(
        env_pool,
        #rand_agent
        sac
    )
# for t in range(int(1E5)):
#     batches = [env_pool.sample(256) for _ in range(4)]
#     sac.world.train(batches)
#     if t % 100 == 0:
#         print(f'Update time: {t}')
#         print(sac.world.loss)

# for i in range(sac.ensem_n):
#     torch.save(sac.world.Ts[i].state_dict(), f'./save/humanoid/pre_T{i}.pt')
#     torch.save(sac.world.Rs[i].state_dict(), f'./save/humanoid/pre_R{i}.pt')

# for i in range(sac.ensem_n):
#     torch.save(sac.ACs[i].state_dict(), f'./save/humanoid/AC{i}.pt')
#     torch.save(sac.ACs_targ[i].state_dict(), f'./save/humanoid/AC_tar{i}.pt')
    


for i in range(sac.ensem_n):
    sac.world.Ts[i].load_state_dict(torch.load(f'./save/humanoid/pre_T{i}.pt', map_location=torch.device('cuda')))
    sac.world.Rs[i].load_state_dict(torch.load(f'./save/humanoid/pre_R{i}.pt', map_location=torch.device('cuda')))
for i in range(sac.ensem_n):
    sac.ACs_targ[i].load_state_dict(torch.load(f'./save/humanoid/AC_tar{i}.pt', map_location=torch.device('cuda')))
    sac.ACs[i].load_state_dict(torch.load(f'./save/humanoid/AC{i}.pt', map_location=torch.device('cuda')))


b = env_pool.sample(10)
s = b['s']
a = b['a']
r = b['r']
s2 = b['s2']
d = b['d']

ps2, pd = sac.world.Ts[0](s, a)
pr = sac.world.Rs[0](s, a, s2)


timer = 1

for t in range(1000*40):
    env_sampler.sample_and_push(
        env_pool,
        sac
    )
    if t%50 == 0:
        # for _ in range(4):
        #     batches = [env_pool.sample(1024) for _ in range(4)]
        #     sac.world.train(batches)
        for _ in range(50):
            batches = [env_pool.sample(256) for _ in range(4)]    
            sac.update(batches, False)
        
    if t % 50 == 0:
        print(f'Loss_q: {sac.loss_q}')
    if t % 300 == 0:
        mean_ep_ret = 0
        mean_ep_len = 0
        max_ep_ret = 0
        min_ep_ret = 0

        sac.test(10)

        print(f'Timer: {timer}')
        sac.print_log()
        for i in range(sac.ensem_n):
            sac.ACs_targ[i].load_state_dict(torch.load(f'./save/humanoid/AC_tar{i}.pt', map_location=torch.device('cuda')))
            sac.ACs[i].load_state_dict(torch.load(f'./save/humanoid/AC{i}.pt', map_location=torch.device('cuda')))



    timer += 1
    
    
    
# mean_ep_ret = 0
# mean_ep_len = 0
# max_ep_ret = 0
# min_ep_ret = 0

# for j in range(10):
#     s, d, ep_ret, ep_len = env.reset(), False, 0, 0
#     while not (d or (ep_len == 1000)):
#         s = torch.as_tensor(s, dtype=torch.float32)
#         s, r, d, _ = env.step(sac.act(s, True))
#         ep_ret += r
#         ep_len += 1
#     mean_ep_ret = (j/(j+1))*mean_ep_ret + (1/(j+1))*ep_ret
#     mean_ep_len = (j/(j+1))*mean_ep_len + (1/(j+1))*ep_len
    
#     if ep_ret > max_ep_ret or j == 0:
#         max_ep_ret = ep_ret
#     if ep_ret < min_ep_ret or j == 0:
#         min_ep_ret = ep_ret

# print('------------------------------------')
# print(f'Timer: {timer}')
# print(f'EpRet: {mean_ep_ret}')
# print(f'EpLen: {mean_ep_len}')
# print(f'MaxRet: {max_ep_ret}')
# print(f'MinRet: {min_ep_ret}')
# print(f'LossQ: {sac.loss_q}')
# print(f'LossPi: {sac.loss_pi}')
# print('------------------------------------\n')