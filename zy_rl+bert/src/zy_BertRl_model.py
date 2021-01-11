from transformers.modeling_utils import PreTrainedModel
from transformers import BertModel, BertPreTrainedModel
from transformers.file_utils import *
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from zy_actor_critic_model import Actor, Critic

eps = np.finfo(np.float32).eps.item()  # 非负的最小值，使得归一化时分母不为0


class MyBertRl(BertPreTrainedModel):
    def __init__(self, config):
        super(MyBertRl, self).__init__(config)

        self.num_labels = config.num_labels
        self.label2id = config.label2id
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.2)
        self.actor = Actor(config)
        self.critic = Critic(config)
        self.rl_trainable = None
        self.critic_trainable = None

        self.init_weights()

    def forward(self, **kwargs):
        input_ids = kwargs.pop("input_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        labels = kwargs.pop("labels", None)
        if kwargs:
            print(f"There are some parameters that not use:{kwargs.keys()}")

        # 得到句子表示(bs*seq_len*hs)
        if self.critic_trainable:
            outputs = self.bert(input_ids, attention_mask=attention_mask)
        else:
            with torch.no_grad():
                outputs = self.bert(input_ids, attention_mask=attention_mask)

        sequence_output = outputs[0]
        # (bs*seqlen*hs)维  1位置是pool后的(bs*hs)维的 用于句子分类
        sequence_output = self.dropout(sequence_output)  # dropout层

        if self.rl_trainable:

            # 得到action和相应的log prob (32, 128)
            action, action_log_prob = self.actor.get_action(hidden_state=sequence_output)
            # 使用critic生成分类概率
            class_prob = self.critic(hidden_state=sequence_output, attention_mask=attention_mask)
            # 得到对应action的句子mask，使用当前bert+critic得到新的分类概率
            active_mask = attention_mask * action  # 全为1的部分保留
            with torch.no_grad():
                new_outputs = self.bert(input_ids, attention_mask=active_mask)

            new_sequence_output = new_outputs[0]
            new_sequence_output = self.dropout(new_sequence_output)
            new_class_prob = self.critic(attention_mask=active_mask, new_hidden_state=new_sequence_output)
            # 两者相应位置相减，得到采取动作后概率预测准确率的增减
            reward_ = new_class_prob - class_prob
            # 对应label位置使用sigmoid函数映射到0，1之间，作为reward *2 负奖利为小于1，正奖励大于1
            # reward_list = F.sigmoid(reward_)
            reward_list = reward_.view(-1, self.config.num_labels)
            flat_active_mask = active_mask.view(-1)
            label = labels.view(-1)
            reward = torch.tensor(0.).cuda()
            for index, flag in enumerate(attention_mask.view(-1)):
                if flag == 0:
                    pass
                else:
                    if label[index] == -100:
                        # 如果对应label是-100， 跳过
                        continue
                    if flat_active_mask[index] == 1:
                        # 如果是被保留词的索引
                        reward += reward_list[index, label[index]]
                    else:
                        # 是被删除词的索引
                        if label[index] != self.label2id['O']:
                            # 如果把非O标签的词删除，此次采样的reward变为一个极小的值
                            reward = torch.tensor(-9999.)
                            break
                        else:
                            reward += 0.1  # 可以设置成超参数，删除一个O标签导致的reward增长
            reward = torch.sigmoid(reward) + eps  # 映射到01之间
            reward = torch.log(reward)
            # reward = reward * 2.0 + eps
            # reward乘以策略梯度取负作为loss
            action_log_prob = action_log_prob.view(-1)
            action_log_prob = torch.where(
                attention_mask.view(-1) == 1, action_log_prob, torch.tensor(0.).type_as(action_log_prob)
            )
            action_log_prob = action_log_prob.sum()
            loss = - action_log_prob * reward  # 最大化目标函数变为最小化loss

            logits = new_class_prob.view(-1, self.config.num_labels)
            # 得到shape
            bs, seq_len = input_ids.size()
            bool_active_mask = active_mask.reshape(bs * seq_len, 1).expand_as(logits)
            bool_active_mask = bool_active_mask.reshape(bs * seq_len * 5) == 0
            # 得到被删除词的位置，预测改为O
            a = torch.tensor([1, 0, 0, 0, 0]).type_as(logits).expand_as(logits).reshape(bs * seq_len * 5)
            logits = logits.view(-1)
            logits = torch.where(
                bool_active_mask, a, logits
            )
            logits = logits.view(bs, seq_len, self.config.num_labels)
            with torch.no_grad():
                if not self.critic_trainable:
                    seg_output = self.critic(hidden_state=sequence_output, attention_mask=attention_mask, label=labels)
                    c_loss = seg_output[0]
                else:
                    with torch.enable_grad():
                        seg_output = self.critic(hidden_state=sequence_output, attention_mask=attention_mask, label=labels)
                        c_loss = seg_output[0]

            # loss = 0.1 * loss + c_loss
            final_loss = c_loss + 0.1 * loss

            output = (final_loss, logits, active_mask, c_loss.item(), loss.item())


        else:
            output = self.critic(hidden_state=sequence_output, attention_mask=attention_mask, label=labels)

        return output


class MyBertRl_V2(nn.Module):
    def __init__(self, config):
        super(MyBertRl_V2, self).__init__()

        self.config = config
        self.bert = BertModel(config).from_pretrained("./pretrained_model", config=config)
        with torch.no_grad():
            self.bert_encoder = BertModel(config).from_pretrained('./pretrained_model', config=config)
        self.actor = Actor(config)
        self.critic = Critic(config)
        self.dropout = nn.Dropout(0.1)
        self.rl_trainable = None
        self.critic_trainable = None
        self.label2id = config.label2id

    def process_del_sent(self, action, input_ids, is_head):
        # 根据action删除词的函数
        # 被删除部分均为0
        bs, seq_len = input_ids.size()
        del_input_ids = torch.zeros_like(input_ids)
        for i in range(bs):
            for j in range(seq_len):
                if action[i][j] == 1 and is_head[i][j] == 1:
                    del_input_ids[i][j] = 1
                elif action[i][j] == 0 and is_head[i][j] == 1:
                    del_input_ids[i][j] = 0
                else:
                    del_input_ids[i][j] = del_input_ids[i][j - 1]
        del_input_ids = del_input_ids * input_ids

        out_ids = del_input_ids.cpu().numpy().tolist()
        # pad 到原来大小
        for i in range(bs):
            while 0 in out_ids[i]:
                out_ids[i].remove(0)
            out_ids[i] = out_ids[i] + [0] * (seq_len - len(out_ids[i]))
        out_ids = torch.tensor(out_ids).type_as(input_ids)
        return out_ids, del_input_ids

    def forward(self, **kwargs):

        input_ids = kwargs.pop("input_ids", None)
        is_head = kwargs.pop("is_heads", None)
        labels = kwargs.pop("labels", None)
        # if kwargs:
        #     print(f"There are some parameters that not use:{kwargs.keys()}")

        if self.rl_trainable:
            bert_embedding, _ = self.bert_encoder(input_ids)

            action, action_log_prob = self.actor.get_action(bert_embedding)
            # is_head == 0 的位置不能够删除，所以action == 1
            # 删除的位置就是action为0的部分
            action = torch.where(torch.tensor(is_head).type_as(action) == 0, torch.tensor(1).type_as(action), action)

            new_input_ids, del_input_ids = self.process_del_sent(action, input_ids, is_head)

            if self.critic_trainable:
                hidden_state, _ = self.bert(input_ids)
                critic_loss, logits = self.critic(hidden_state=hidden_state, label=labels)
                with torch.no_grad():
                    new_hidden_state, _ = self.bert(new_input_ids)
                class_prob = F.softmax(logits, -1)
                new_class_prob = self.critic(hidden_state=new_hidden_state)
            else:
                with torch.no_grad():
                    hidden_state, _ = self.bert(input_ids)
                    critic_loss, logits = self.critic(hidden_state=hidden_state, label=labels)
                    new_hidden_state, _ = self.bert(new_input_ids)
                    class_prob = F.softmax(logits, -1)
                    new_class_prob = self.critic(hidden_state=new_hidden_state)

            reward = torch.zeros_like(labels).type_as(logits)
            bs, seq_len = input_ids.size()
            # 得到rl动作后的 logit， 一定要确认O的位置
            out_logits = torch.tensor([1, 0, 0, 0, 0]).type_as(logits)\
                .repeat(bs, seq_len).reshape(bs, seq_len, self.config.num_labels)

            for b in range(bs):
                ptr = 0
                for s in range(seq_len):
                    prob2, prob1 = 0., 0.
                    if labels[b][s] == -100:
                        if del_input_ids[b][s] != 0:
                            ptr += 1
                        out_logits[b][s] = class_prob[b][s][:]
                        continue
                    prob1 = class_prob[b][s][labels[b][s]]
                    if del_input_ids[b][s] != 0:
                        prob2 = new_class_prob[b][ptr][labels[b][s]]
                        out_logits[b][s] = new_class_prob[b][ptr][:]  # 如果是保留的单词
                        ptr += 1
                    else:
                        if labels[b][s] == self.label2id['O']:
                            prob2 = 1
                        else:
                            prob2 = 0
                    reward[b][s] = prob2 - prob1

            # reward = F.sigmoid(reward)
            reward = F.logsigmoid(reward)
            # actor_loss = - reward * action_log_prob
            # actor_loss = action_log_prob.mean()

            # loss = critic_loss - 0.1 * actor_loss
            output = (critic_loss, out_logits, action, reward, action_log_prob)

        else:
            hidden_state, _ = self.bert(input_ids)
            output = self.critic(hidden_state=hidden_state, label=labels)

        return output
