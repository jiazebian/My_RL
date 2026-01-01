import random
import gymnasium as gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils

#首先定义一个网络1
class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)

#再定义一个buffer，可存，可取，可查看大小
class ReplayBuffer:
    def __init__(self,capacity):
        self.buffer=collections.deque(maxlen=capacity)#定义一个队列，它最大容量是capacity

    def add(self,state,action,reward,next_state,done): # 将数据加入buffer
        self.buffer.append((state,action,reward,next_state,done))#把输入数据打包成元组，存到队列

    def sample(self,batch_size):# 从buffer中采样数据,数量为batch_size
        transitions=random.sample(self.buffer,batch_size)
        state,action,reward,next_state,done=zip(*transitions)
        return np.array(state),action,reward,np.array(next_state),done


    def size(self): #目前buffer中的数据的数量
        return len(self.buffer)
    
#定义DQN算法
class DQN:
    def __init__(self,state_dim,hidden_dim,action_dim,learning_rate,gamma,epsilon,target_update,device):
        #先定义训练网络和目标网络
        self.action_dim=action_dim
        self.q_net=Qnet(state_dim,hidden_dim,action_dim)
        self.target_q_net=Qnet(state_dim,hidden_dim,action_dim)
        #定义训练用的优化器
        self.optimizer=torch.optim.Adam(self.q_net.parameters(),lr=learning_rate)
        #定义折扣因子、贪心策略、更新频率等
        self.gamma=gamma
        self.epsilon=epsilon
        self.target_update=target_update
        self.device=device
        self.q_net.to(device)#要把网络放到gpu
        self.target_q_net.to(device)
        self.count = 0  # 计数器,记录更新次数

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)#返回一个随机的，大小为2的数组
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item() #反函数，argmax(a)Q
        return action#一个大小为2的数组

    def update(self, transition_dict):#传进来一个共500个样本的字典
        #先tensor化并存在cuda上
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(self.device)
        q_values = self.q_net(states).gather(1, actions)  # Q值
        #print("actions:",actions),形状是（batchsize,1),因为每个动作是确定的，要么是0，要么是1
        #print("q_net(states):",self.q_net(states))形状是（batchsize,2),因为这个网络的输出大小就是动作空间的大小（2）,表示对应动作取得的价值
        #print("q_values:",q_values)形状是（batchsize,1),因为价值是已经采集到的，是一个数字
        #gather(1, actions) 的作用是：沿着第 1 维（列）从每一行中“按索引取值”，把每个样本对应动作的 Q 值挑出来，得到形状为 (batch_size, 1) 的张量（每行一个值）
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        #.max(1)[0].view(-1, 1)表示把形状为(batchsize,2)的Q值，沿着第1维（列）选取最大的值，也就是取每行（每个样本）最大的值，并把结果整理成(batchsize,1)的形式
        #print(states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape,max_next_q_values.shape)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数，这里dqn_loss是一个一维张量
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step() #利用梯度更新参数

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 定期更新目标网络
        self.count += 1

#设置初始状态
def env_reset(env,seed=None):
    if seed is not None:
        res=env.reset(seed=seed)
    else:
        res=env.reset()
    #res类似(array([ 0.01369617, -0.02302133, -0.04590265, -0.04834723], dtype=float32), {})，所以是一个有两个元素的元组
    if isinstance(res,tuple) and len(res)==2:
        obs,_info=res
        return obs
    return res

#用于返回下一个状态
def env_step(env,action):
    res=env.step(action)
    obs,reward,terminated,truncated,info=res
    done=terminated or truncated
    return obs,reward,done,info

#定义超参数
lr=2e-3
num_episodes=500
hidden_dim=128
gamma=0.98
epsilon=0.01
target_update=10
buffer_size=10000
minimal_size=500
batch_size=64
device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name='CartPole-v0'
env=gym.make(env_name)

#全局随机种子
random.seed(0)#必须
np.random.seed(0)#对应np.random
torch.manual_seed(0)#必须（控制 CPU RNG，以及默认的 CUDA RNG 的种子初始化）
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
#针对游戏环境中 reset(seed=...)的随机
env.action_space.seed(0)
env.observation_space.seed(0)

replay_buffer=ReplayBuffer(buffer_size)
state_dim=env.observation_space.shape[0]
action_dim=env.action_space.n
agent=DQN(state_dim,hidden_dim,action_dim,lr,gamma,epsilon,target_update,device)
#print(agent.device) #在cuda上
return_list=[]
# 训练循环
for outer_i in range(10):#分成十次跑，一次跑num_episodes/10个回合
    with tqdm(total=int(num_episodes/10),desc='Iteration %d' % outer_i) as pbar:#进度条
        for i_episode in range(int(num_episodes/10)):
            episode_return=0
            #可选择为每个episode传入指定的seed
            state=env_reset(env,seed=0)
            done=False
            while not done:
                action=agent.take_action(state)
                next_state,reward,done,_=env_step(env,action)
                replay_buffer.add(state,action,reward,next_state,done)
                state=next_state#迭代状态
                episode_return+=reward
                if replay_buffer.size()>minimal_size:
                    b_s,b_a,b_r,b_ns,b_d=replay_buffer.sample(batch_size)
                    transition_dict={
                        'states':b_s,
                        'actions':b_a,
                        'next_states':b_ns,
                        'rewards':b_r,
                        'dones':b_d
                    }
                    agent.update(transition_dict)
            
            return_list.append(episode_return)
            if(i_episode +1)%10==0:
                pbar.set_postfix({
                    'episode':'%d' % (num_episodes/10 * outer_i+i_episode+1),#目前总循环数
                    'return':'%.3f' % np.mean(return_list[-10:])#平滑处理
                })
            pbar.update(1)
            
episodes_list=list(range(len(return_list)))#横坐标，回合数
plt.plot(episodes_list,return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list,9)#再次平滑处理
plt.plot(episodes_list,mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()