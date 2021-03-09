import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from zy_utils import NerDataset

TARGET_REPLACE_ITER = 100   # Q 现实网络的更新频率
BATCH_SIZE = 32
GAMMA = 0.9

train_dataset = NerDataset(hp.trainset)

class State(nn.Module):
    def __init__(self):
        super(State, self).__init__()
        self.lstm = nn.LSTM(728, 364, bidirectional=True, dropout=0.1)
        self.out = nn.Linear(364, 10)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        s, _ = self.lstm(x)
        s = F.relu(s)
        s = self.out(s) #
        return s


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 384)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(384, 2)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net , self.target_net = Net(), Net()

        self.learn_step_counter = 0  # 用于target更新计时
        self.memory_counter = 0  # 记忆库计数
        self.memory = np.zeros((2000, 10 * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.0001)
        self.loss_func = nn.MSELoss()
        self.epsilon = 0.9

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # 这里只输入一个 sample
        if np.random.uniform() < self.epsilon:  # 选最优动作
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]  # return the argmax
        else:  # 选随机动作
            action = np.random.randint(0, 2)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % 2000
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(2000, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.config.hidden_state])
        b_a = torch.LongTensor(b_memory[:,
                               self.config.hidden_state:self.config.hidden_state+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:,
                                self.config.hidden_state+1:self.config.hidden_state+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.config.hidden_state:])

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + GAMMA * q_next.max(1)[0]   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()
sta = State()
for i_episode in range(400):
    input = torch.rand([32, 256, 768])
    states = sta(input) # 化成[x, 768] 将所有状态，除去pad部分存成2维
    for ten_s in states:
        a = dqn.choose_action(ten_s)
