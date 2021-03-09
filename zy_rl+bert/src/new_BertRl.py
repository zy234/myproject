import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch.distributions import Categorical

class Bert_Rl(nn.Module):
    def __init__(self, tag2ix, hidden_size=768):
        super(Bert_Rl, self).__init__()
        self.tag2ix = tag2ix
        self.num_labels = len(self.tag2ix)
        self.hidden_size = hidden_size

        self.bert = BertModel.from_pretrained("../pretrained_model")
        self.critic_classifier = nn.Linear(self.hidden_size, self.num_labels)
        self.actor_classifier = nn.Linear(self.hidden_size, 2)
        self.dropout = nn.Dropout(0.5)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def bert_encoder(self, x, is_drop=True):

        bert_enc, _ = self.bert(x)
        if is_drop:
            bert_enc = self.dropout(bert_enc)
        return bert_enc

    def get_action_prop(self, hidden_state):
        action_scores = self.actor_classifier(hidden_state)
        action_prob = F.softmax(action_scores, -1)
        return action_prob

    def get_action(self, hidden_state):
        action_prob = self.get_action_prob()  # (32, 128, 2)
        m = Categorical(action_prob)
        action = m.sample()  # (32, 128)
        action_log_prob = m.log_prob(action)  # (32, 128)

        return action, action_log_prob

    def forward(self, x):
        #
        bert_hidden_state = self.bert_encoder(x)
        action =1