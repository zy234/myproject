from transformers.modeling_distilbert import DistilBertPreTrainedModel
from transformers import DistilBertTokenizer, DistilBertModel, BertModel
import torch.nn as nn
import torch
import numpy as np


class DistilForMaskPredict(DistilBertPreTrainedModel):
    def __init__(self, config):
        super(DistilForMaskPredict, self).__init__(config)
        self.num_labels = config.vocab_size
        self.distilbert = DistilBertModel(config)
        self.classifier = nn.Linear(config.dim, config.vocab_size)
        self.dropout = nn.Dropout(0.5)

        self.init_weights()

    def forward(self, input_ids, batch_pos, weight=None, attention_mask=None, head_mask=None, labels=None):
        distilbert_output = self.distilbert(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           head_mask=head_mask
                                           )
        hidden_state = distilbert_output[0]  # bs,seq_len, dim
        bs, seq_len, dim = hidden_state.shape[0], hidden_state.shape[1], hidden_state.shape[2]
        seq_out = self.dropout(hidden_state)

        batch_pos = batch_pos.unsqueeze(-1).expand(bs, dim).unsqueeze(-2)
        # 取出bs中对应token的隐藏状态
        token_hid = torch.gather(seq_out, 1, batch_pos)  # bs, 1, dim

        logits = self.classifier(token_hid)  # bs, 1, voc_size
        out_puts = (logits, ) + distilbert_output[:2]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0, reduction='mean')
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            out_puts = (loss, ) + out_puts

        return out_puts

def mask_input(input_ids, labels):
    # 传入的numpy数组
    new_sentences = []
    new_labels = []
    pos = []
    for index, sent_ids in enumerate(input_ids):
        for ce_id, label in enumerate(labels[index]):
            # 分别将mask后的数据加入新数据集
            if label == 2 or label == 4:
                id = sent_ids[ce_id]
                sent_ids[ce_id] = 103  # vocab中[MASK]对应的id
                new_sentences.append(sent_ids.copy())
                new_labels.append(id)
                pos.append(ce_id)
                # 复原sent_ids
                sent_ids[ce_id] = id
    return np.array(new_sentences), np.array(new_labels), np.array(pos)


# a = torch.randint(1, 40, size=(2, 3, 4))
# print(a.size())
# print(a)
# pos = torch.LongTensor([1, 2])
# pos = pos.unsqueeze(-1).expand(2, 4).unsqueeze(-2)
# print(pos)
# # index = torch.LongTensor([[[0, 0, 0, 0]], [[2, 2, 2, 2]]])
# print(torch.gather(a, 1, pos))
