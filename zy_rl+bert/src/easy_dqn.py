import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

TARGET_REPLACE_ITER = 100   # Q 现实网络的更新频率
MEMORY_CAPACITY = 1000
BATCH_SIZE = 32
GAMMA = 0.9
N_STATES = 10
MASK_NUM = 3
MASK_INDEX = 103

class State(nn.Module):
    def __init__(self):
        super(State, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(768, N_STATES)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.dropout(x)
        s = self.out(x) #
        return s


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 2)
        self.fc1.weight.data.normal_(0, 0.1)

    def forward(self, x):

        actions_value = self.fc1(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net , self.target_net = Net(), Net()
        self.sta = State().cuda()
        self.eval_net.cuda()
        self.target_net.cuda()

        self.learn_step_counter = 0  # 用于target更新计时
        self.memory_counter = 0  # 记忆库计数
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.0001)
        self.loss_func = nn.MSELoss()
        self.epsilon = 0.9

    def choose_action(self, x):
        x = torch.unsqueeze(x, 0)
        # 这里只输入一个 sample
        if np.random.uniform() < self.epsilon:  # 选最优动作
            actions_value = self.eval_net.forward(x)
            action = torch.argmax(actions_value, 1)  # return the argmax
            action_score = actions_value.squeeze()[action][0]
            action = action.cpu().numpy()[0]
        else:  # 选随机动作
            action = np.random.randint(0, 2)
            action_score = 0
        return action, action_score

    def store_transition(self, s, a, r):
        transition = np.hstack((s, a, r))
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target net 参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).cuda()
        b_a = torch.LongTensor(b_memory[:,N_STATES:N_STATES+1].astype(int)).cuda()
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).cuda()

        # 针对做过的动作b_a, 来选 q_eval 的值, (q_eval 原本有所有动作的值)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s).detach()     # q_next 不进行反向传递误差, 所以 detach
        q_target = b_r + GAMMA * q_next.max(1)[0].reshape(32, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # 计算, 更新 eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

def easy_dqn(model, x, y, seqlens, dqn):

    # 训练dqn
    with torch.no_grad():
        _, hidden_state = model(x)
        hidden_state = hidden_state.detach()
        output = model(x, y, detached_loss=True)
        loss = output[0].reshape(len(x), len(x[0]))
        baseline = loss.mean(dim=1, keepdim=True)

        for sent, label, hid_sent, base, seqlen in zip(x, y, hidden_state, baseline, seqlens):
            a_list, reward_list, state_list = [], [], []
            for idx in range(seqlen):
                if label[idx] != -100:
                    state = dqn.sta(hid_sent[idx])
                    state_list.append(state.cpu().numpy())
                    a, _ = dqn.choose_action(state)
                    a_list.append(a)

                    mask_sent = copy.deepcopy(sent)
                    mask_sent[idx] = MASK_INDEX
                    out = model(mask_sent.unsqueeze(0), label.unsqueeze(0))
                    reward = out[0] - base
                    reward_list.append(reward.cpu().numpy())

            for s, a, r in zip(state_list, a_list, reward_list):
                if a == 1:
                    dqn.store_transition(s, a, np.mean(reward_list))
                else:
                    dqn.store_transition(s, a, r)
                if dqn.memory_counter > MEMORY_CAPACITY:
                    with torch.enable_grad():
                        dqn.learn()  # 记忆库满了就进行学习

    # 得到dqn生成的数据
        sta = dqn.sta(hidden_state)
        for bs in range(len(x)):
            list_a, list_action_score = [], []
            for word_state in range(seqlens[bs]):
                a, action_score = dqn.choose_action(sta[bs][word_state])
                list_a.append(a)
                list_action_score.append(action_score)

            list_action_score = np.array(list_action_score)
            list_idx = np.array(np.where(np.array(list_a) == 0)).squeeze()
            if list_idx.size == 1:  # 单个值squeeze会把其变为一个数，len函数会报错
                list_idx = [list_idx]
            if len(list_idx) > MASK_NUM:
                mask_score = list_action_score[np.array(list_idx)]
                top_k_idx=mask_score.argsort()[::-1][0:MASK_NUM]
                mask_idx = list_idx[top_k_idx]
            else:
                mask_idx = list_idx
            x[bs][mask_idx] = MASK_INDEX

    return x
