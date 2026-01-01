from tqdm import tqdm
import numpy as np
import torch
import collections
import random

# 不再调用 env.seed(0) —— 改为在第一次 reset 时或每个 episode reset 时传 seed
# 为兼容 gym 和 gymnasium 的 step/reset 接口，定义小 helper：
def env_reset(env, seed=None):
    # Gymnasium: env.reset(seed=...) -> (obs, info)
    # Old Gym: env.reset() -> obs
    if seed is not None:
        res = env.reset(seed=seed)
    else:
        res = env.reset()
    #print("res",res)，res类似(array([ 0.01369617, -0.02302133, -0.04590265, -0.04834723], dtype=float32), {})，所以是一个有两个元素的元组
    # 兼容：如果返回为 (obs, info)
    if isinstance(res, tuple) and len(res) == 2:#会执行
        obs, _info = res
        #print("obs, _info",res)
        return obs
    return res

def env_step(env, action):
    # Gymnasium: step -> (obs, reward, terminated, truncated, info)
    # Old Gym: step -> (obs, reward, done, info)
    res = env.step(action)
    # 如果是 5 元组（gymnasium）
    if isinstance(res, tuple) and len(res) == 5:
        obs, reward, terminated, truncated, info = res
        done = terminated or truncated
        return obs, reward, done, info
    # 否则（旧 gym），直接返回
    return res


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = [] #记录每个回合总回报R的列表
    for i in range(10):#把总回合数分成十个批次
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:#绘制进度条
            for i_episode in range(int(num_episodes/10)):#第i_episode回合
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}#定义一个字典，存放一个epsilon的全部过程
                #state = env.reset()
                state = env_reset(env, seed=0)
                done = False
                while not done:#在一个回合中不断交互
                    action = agent.take_action(state) #选择一个策略，具体选什么需要参考算法定义，在PPO中是按照概率采样
                    #next_state, reward, done, _ = env.step(action)
                    next_state, reward, done, _ = env_step(env,action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)#在线的意义就是现交互现训练
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                #state = env.reset()
                state = env_reset(env, seed=0)
                done = False
                while not done:
                    action = agent.take_action(state)
                    #next_state, reward, done, _ = env.step(action)
                    next_state, reward, done, _ = env_step(env,action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):#计算优势函数，把一序列的 TD 残差（td_delta）做折扣累加，#折扣因子γ=gamma*lmbda
    td_delta = td_delta.detach().numpy() #断开梯度,advantage 应当被视为目标权重/常数
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]: #逆序
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()#保持正确的时间顺序
    return torch.tensor(advantage_list, dtype=torch.float)
                