import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import time


#####################  hyper parameters  ####################

MAX_EPISODES = 200               # 最大训练代数
MAX_EP_STEPS = 200               # episode最大持续帧数

RENDER = False
ENV_NAME = 'Pendulum-v0'         # 游戏名称
SEED = 123                       # 随机数种子


###############################  DDPG  ####################################
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound, ):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.a_bound = a_bound
        self.pointer = 0  # exp buffer指针
        self.lr_a = 0.001  # learning rate for actor
        self.lr_c = 0.002  # learning rate for critic
        self.gamma = 0.9  # reward discount
        self.tau = 0.01  # 软更新比例
        self.memory_capacity = 10000
        self.batch_size = 32
        self.memory = np.zeros((self.memory_capacity, s_dim * 2 + a_dim + 1), dtype=np.float32)

        class ANet(nn.Module):  # 定义动作网络
            def __init__(self, s_dim, a_dim, a_bound):
                super(ANet, self).__init__()
                self.a_bound = a_bound
                self.fc1 = nn.Linear(s_dim, 30)
                self.fc1.weight.data.normal_(0, 0.1)  # initialization

                self.out = nn.Linear(30, a_dim)
                self.out.weight.data.normal_(0, 0.1)  # initialization

            def forward(self, x):
                x = self.fc1(x)
                x = F.relu(x)
                x = self.out(x)
                x = F.tanh(x)
                actions_value = x * self.a_bound.item()
                return actions_value

        class CNet(nn.Module):  # 定义价值网络
            def __init__(self, s_dim, a_dim):
                super(CNet, self).__init__()
                self.fcs = nn.Linear(s_dim, 30)
                self.fcs.weight.data.normal_(0, 0.1)  # initialization

                self.fca = nn.Linear(a_dim, 30)
                self.fca.weight.data.normal_(0, 0.1)  # initialization

                self.out = nn.Linear(30, 1)
                self.out.weight.data.normal_(0, 0.1)  # initialization

            def forward(self, s, a):
                x = self.fcs(s)  # 输入状态
                y = self.fca(a)  # 输入动作
                net = F.relu(x + y)
                actions_value = self.out(net)  # 给出V(s,a)
                return actions_value

        self.Actor_eval = ANet(s_dim, a_dim, a_bound)  # 主网络
        self.Actor_target = ANet(s_dim, a_dim, a_bound)  # 目标网络
        self.Critic_eval = CNet(s_dim, a_dim)  # 主网络
        self.Critic_target = CNet(s_dim, a_dim)  # 当前网络
        self.ctrain = torch.optim.Adam(self.Critic_eval.parameters(), lr=self.lr_c)  # critic的优化器
        self.atrain = torch.optim.Adam(self.Actor_eval.parameters(), lr=self.lr_a)  # actor的优化器
        self.loss_td = nn.MSELoss()  # 损失函数采用均方误差

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        return self.Actor_eval(s)[0].detach()  # detach()不需要计算梯度

    def learn(self):

        for x in self.Actor_target.state_dict().keys():
            eval('self.Actor_target.' + x + '.data.mul_((1 - self.tau))')
            eval('self.Actor_target.' + x + '.data.add_(self.tau * self.Actor_eval.' + x + '.data)')
        for x in self.Critic_target.state_dict().keys():
            eval('self.Critic_target.' + x + '.data.mul_((1- self.tau))')
            eval('self.Critic_target.' + x + '.data.add_(self.tau * self.Critic_eval.' + x + '.data)')

        # soft target replacement

        indices = np.random.choice(self.memory_capacity, size=self.batch_size)  # 随机采样的index
        bt = self.memory[indices, :]  # 采样batch_size个sample
        bs = torch.FloatTensor(bt[:, :self.s_dim])  # state
        ba = torch.FloatTensor(bt[:, self.s_dim: self.s_dim + self.a_dim])  # action
        br = torch.FloatTensor(bt[:, -self.s_dim - 1: -self.s_dim])  # reward
        bs_ = torch.FloatTensor(bt[:, -self.s_dim:])  # next state

        a = self.Actor_eval(bs)
        q = self.Critic_eval(bs, a)  # loss=-q=-ce(s,ae(s))更新ae   ae(s)=a   ae(s_)=a_
        # 如果 a是一个正确的行为的话，那么它的Q应该更贴近0
        loss_a = -torch.mean(q)
        # print(q)
        # print(loss_a)
        self.atrain.zero_grad()
        loss_a.backward()
        self.atrain.step()

        a_ = self.Actor_target(bs_)  # 这个网络不及时更新参数, 用于预测 Critic 的 Q_target 中的 action
        q_ = self.Critic_target(bs_, a_)  # 这个网络不及时更新参数, 用于给出 Actor 更新参数时的 Gradient ascent 强度
        q_target = br + self.gamma * q_  # q_target = 负的
        # print(q_target)
        q_v = self.Critic_eval(bs, ba)
        # print(q_v)
        td_error = self.loss_td(q_target, q_v)
        # td_error = R + self.gamma * ct（bs_,at(bs_)）-ce(s,ba) 更新ce ,但这个ae(s)是记忆中的ba，让ce得出的Q靠近Q_target,让评价更准确
        # print(td_error)
        self.ctrain.zero_grad()
        td_error.backward()
        self.ctrain.step()

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.memory_capacity  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1  # 指示sample位置的指针+1

###############################  training  ####################################
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(SEED)                                          # 设置Gym的随机数种子
torch.manual_seed(SEED)                                 # 设置pytorch的随机数种子

s_dim = env.observation_space.shape[0]                  # 状态空间
a_dim = env.action_space.shape[0]                       # 动作空间
a_bound = env.action_space.high                         # 动作取值区间,对称区间，故只取上界
ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3                                                 # 动作服从的高斯分布的方差，控制探索程度
t1 = time.time()                                        # 开始时间
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)         # 为什么要对reward归一化

        if ddpg.pointer > ddpg.memory_capacity:              # 经验池已满
            var *= .9995                                # 学习阶段逐渐降低动作随机性decay the action randomness
            ddpg.learn()                                # 开始学习

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            # if ep_reward > -300:
                # RENDER = True
            break
print('Running time: ', time.time() - t1)