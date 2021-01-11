import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class Critic(nn.Module):
    def __init__(self, config):
        super(Critic, self).__init__()

        self.config = config
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # self.hidden_state = hidden_state

    def forward(self, attention_mask=None, label=None, hidden_state=None):
        logits = self.classifier(hidden_state)
        outputs = (logits, )

        if label is not None:
            if attention_mask is not None:
                active_labels = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.config.num_labels)
                label = torch.where(
                    active_labels, label.view(-1), torch.tensor(-100).type_as(label)
                )

                loss = F.cross_entropy(active_logits, label)
            else:
                loss = F.cross_entropy(logits.view(-1, self.config.num_labels), label.view(-1))
            outputs = (loss,) + outputs
            return outputs

        else:
            class_prob = F.softmax(logits, -1)
            return class_prob


class Actor(nn.Module):
    def __init__(self, config):
        super(Actor, self).__init__()

        self.config = config
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size // 2, num_layers=2, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.action_logits = nn.Linear(config.hidden_size, 2)

    def get_action_prob(self, hidden_state):

        # get lstm hidden state
        lstm_feats, _ = self.lstm(hidden_state)
        lstm_feats = self.dropout(lstm_feats)

        action_scores = self.action_logits(lstm_feats)
        action_prob = F.softmax(action_scores, -1)

        return action_prob

    def get_action(self, hidden_state):
        action_prob = self.get_action_prob(hidden_state)  # (32, 128, 2)
        m = Categorical(action_prob)
        action = m.sample()  # (32, 128)
        action_log_prob = m.log_prob(action)  # (32, 128)

        return action, action_log_prob

    def forward(self, reward, action_log_prob):
        '''
        :param reward:
        :param action_log_prob:
        :return:
        '''

        return reward * action_log_prob