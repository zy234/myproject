
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import random
import time
from collections import deque

# Hyper Parameters for PG Network
GAMMA = 0.95  # discount factor
LR = 0.00001  # learning rate

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# torch.backends.cudnn.enabled = False  # 非确定性算法

class PGNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PGNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)  #nan
        self.fc2 = nn.Linear(20, action_dim)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            nn.init.normal_(m.weight.data, 0, 0.1)
            nn.init.constant_(m.bias.data, 0.01)
            # m.bias.data.zero_()


class PG(object):
    # dqn Agent
    def __init__(self, hidden_state_dim):  # 初始化
        # 状态空间和动作空间的维度
        self.state_dim = hidden_state_dim
        self.action_dim = 2

        # init N Monte Carlo transitions in one game
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        # init network parameters
        self.network = PGNetwork(state_dim=self.state_dim, action_dim=self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)

        # init some parameters
        self.time_step = 0

    def choose_action(self, observation):
        network_output = self.network.forward(observation)
        with torch.no_grad():
            prob_weights = F.softmax(network_output, dim=0).cuda().data.cpu().numpy()
            for data in prob_weights:
                if np.isnan(data):
                    pass
        # prob_weights = F.softmax(network_output, dim=0).detach().numpy()
        action = np.random.choice(range(prob_weights.shape[0]),
                                  p=prob_weights)  # select action w.r.t the actions prob
        return action

    # 将状态，动作，奖励这一个transition保存到三个列表中
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        self.time_step += 1

        # Step 1: 计算每一步的状态价值
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        # 注意这里是从后往前算的，所以式子还不太一样。算出每一步的状态价值
        # 前面的价值的计算可以利用后面的价值作为中间结果，简化计算；从前往后也可以
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * GAMMA + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)  # 减均值
        discounted_ep_rs /= np.std(discounted_ep_rs)  # 除以标准差
        discounted_ep_rs = torch.FloatTensor(discounted_ep_rs).to(device)

        # self.ep_obs从tensor列表转化为tensor
        b = np.random.randn(len(self.ep_obs), len(self.ep_obs[0]))
        b_tensor = torch.from_numpy(b).cuda()
        for i in range(len(self.ep_obs)):
            b_tensor[i] = self.ep_obs[i]
        b_tensor = b_tensor.type_as(self.ep_obs[0])
        # Step 2: 前向传播
        softmax_input = self.network.forward(b_tensor.to(device))
        # all_act_prob = F.softmax(softmax_input, dim=0).detach().numpy()
        neg_log_prob = F.cross_entropy(input=softmax_input, target=torch.LongTensor(self.ep_as).to(device),
                                       reduction='none')

        # Step 3: 反向传播
        loss = torch.mean(neg_log_prob * discounted_ep_rs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 每次学习完后清空数组
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []


# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 3000  # Episode limitation
STEP = 300  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode


def policy_gradient(x, y, model, seqlens, agent):

    mask_x_list = torch.zeros_like(x)
    index = 0
    for epi_x, epi_y, seqlen in zip(x, y, seqlens):
        with torch.no_grad():
            outputs = model(epi_x.unsqueeze(0), epi_y.unsqueeze(0))
            real_loss = outputs[0]
            hidden_state = outputs[2].squeeze()

        for step in range(seqlen):
            action = agent.choose_action(hidden_state[step])  # softmax概率选择action

            # 采取action的reward,暂时先都赋值成一样的reward，即采取一系列动作的最终loss差
            agent.store_transition(hidden_state[step], action, 0)  # 新函数 存取这个transition

        with torch.no_grad():
            mask_x = copy.deepcopy(epi_x)
            mask_x = torch.where(torch.LongTensor(agent.ep_as).cuda() == 1, mask_x[:seqlen],
                                 torch.full_like(mask_x[:seqlen], 103))
            expend = torch.zeros_like(epi_x)  # 将mask x 拼接成和原来一样的长度
            mask_x = torch.cat((mask_x, expend[len(mask_x):]), 0)
            mask_x_list[index] = mask_x
            index = index + 1

            outputs = model(mask_x.unsqueeze(0), epi_y.unsqueeze(0))
            mask_loss = outputs[0]

        reward = real_loss.item() - mask_loss.item()
        agent.ep_rs = np.full_like(np.array(agent.ep_rs), dtype=float, fill_value=reward)
        agent.ep_rs.tolist()

        agent.learn()  # 更新策略网络

    return mask_x_list


