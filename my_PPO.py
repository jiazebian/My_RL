import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
import random

class PolicyNet(torch.nn.Module):#actor
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)#像是分类问题，输出的是选择每个动作的概率


class ValueNet(torch.nn.Module):#critic
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda#gamma*lmbda是真正的回合间的奖励折扣
        self.epochs = epochs  #训练批次
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):#state是一个四元列表
        state = torch.tensor([state], dtype=torch.float).to(self.device) #形状为(1,4)的二维张量
        probs = self.actor(state)#让演员策略网络根据情况输出动作概率组
        action_dist = torch.distributions.Categorical(probs)#用 probs 构造一个类别分布（Categorical distribution，适用于离散动作）
        action = action_dist.sample()#从这个分布里随机采样一个动作（按概率采样，不是取最大概率）
        return action.item()#把只包含一个数值的张量转换成 Python 标量（比如 int）

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -dones) #r+γ*V(t+1)
        td_delta = td_target - self.critic(states)#TD误差，r(s,a)+γ*V(st+1)-V(st)
        #评论员网络既要评论本状态的价值，又要评论下个状态的价值
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,td_delta.cpu()).to(self.device)
        #advantage（这里用 GAE 计算得到的）是对未来多个时步的TD误差加权累积,让数据更加平滑,同时优势函数指的是在s状态采取a动作的好坏，正就是好，负就是坏
        old_log_probs = torch.log(self.actor(states).gather(1,actions)).detach()
        # 计算“旧策略(old policy)对已选动作的对数概率 log π_old(a|s)”，在下面的训练过程中不变
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs) #log( (pΘ'(at|st)/(pΘ(at|st)) )，重要性采样
            surr1 = ratio * advantage #log( (pΘ'(at|st)/(pΘ(at|st)) ) *A（st,at)
            surr2 = torch.clamp(ratio, 1 - self.eps,1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))# 我们希望最大化期望目标J(Θ)，但是pytorch的优化器只能最小化一个损失，所以取负号
            #min 让最终使用的那一项永远是更保守（更小）的 surrogate，遇到好的，不要过于好，遇到坏的，不要过于坏
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))#最小化 MSE(V(s), td_target)，把 td_target 当作“常数目标”来回归
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 500 #总回合数
hidden_dim = 128 #隐藏层大小
gamma = 0.98 #折扣因子γ=gamma*lmbda
lmbda = 0.95 
epochs = 10 #训练批次
eps = 0.2  #PPO中截断范围的参数，设置变化的上限
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CartPole-v0'
env = gym.make(env_name)

# 全局随机种子（保持）
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# 给 action_space / observation_space 也设种子（向后兼容）
env.action_space.seed(0)
env.observation_space.seed(0)

state_dim  = env.observation_space.shape[0] #4
action_dim = env.action_space.n #2
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)

return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes) #在线策略学习
