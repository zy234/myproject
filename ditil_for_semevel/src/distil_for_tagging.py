from transformers.modeling_distilbert import DistilBertPreTrainedModel
from transformers import DistilBertTokenizer, DistilBertModel, BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F


class DistilForTagging(DistilBertPreTrainedModel):
    def __init__(self, config):
        super(DistilForTagging, self).__init__(config)
        self.num_labels = 6  # (O, BC, IC, BE, IE, pad)
        self.ditilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, 6)
        self.dropout = nn.Dropout(0.5)
        # label在loss函数的权重

        self.init_weights()

    def forward(self, input_ids, weight=None, attention_mask=None, head_mask=None, labels=None):
        distilbert_output = self.ditilbert(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           head_mask=head_mask
                                           )
        hidden_state = distilbert_output[0]  # bs,seq_len, dim
        seq_output = self.dropout(hidden_state)
        logits = self.classifier(seq_output)  # bs,seq_len, num_label
        # logits = F.softmax(logits, dim=2)
        # logits = logits[:, :-1, :]  # distilbert的输出有多一个不知道是什么，把最后一个去掉了
        # 直接往tokenizer传句子，有时候会将一个词分为两个

        outputs = (logits,) + distilbert_output[:2]

        if labels is not None:
            # crossentropy函数是softmax+交叉熵的结合体
            loss_fct = nn.CrossEntropyLoss(ignore_index=0, weight=weight, reduction='mean')
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


# out = torch.Tensor([[1, 2, 3], [3, 4, 1], [1, 2, 2]])
# target = torch.LongTensor([0, 1, 2])
# # w = torch.tensor([1, 1, 10], dtype=torch.float32)
#
# loss = F.cross_entropy(out, target,ignore_index=0)
# print(loss)