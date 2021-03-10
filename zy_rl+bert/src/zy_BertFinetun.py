from transformers import BertModel
import torch.nn as nn
import torch

class MyBertModel(nn.Module):
    def __init__(self):
        super(MyBertModel, self).__init__()
        self.bert = BertModel.from_pretrained("./bert+lstm+crf/pretrained_model")
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, 192)
        self.out = nn.Linear(192, 5)
        self.active_func = nn.ReLU()
        self.loss_func = nn.CrossEntropyLoss()
        self.detached_loss_func = nn.CrossEntropyLoss(reduction="none")

        self.num_labels = 5

    def forward(self, inputs, labels=None, attention_mask=None, detached_loss=False):
        outputs = self.bert(inputs, attention_mask=attention_mask)

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.fc1(sequence_output)
        sequence_output = self.active_func(sequence_output)
        logits = self.out(sequence_output)

        loss = None
        if detached_loss:
            loss_func = self.detached_loss_func
        else:
            loss_func = self.loss_func
        if labels is not None:
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(self.loss_func.ignore_index).type_as(labels)
                )
                loss = loss_func(active_logits, active_labels)
            else:
                loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不计算loss，返回logits和bert隐藏状态
        output = (loss, logits, outputs[0]) if loss is not None else (logits, outputs[0])
        return output