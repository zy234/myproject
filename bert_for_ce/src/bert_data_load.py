#
import torch
from transformers import BertTokenizer
import numpy as np


class MyDataSet():
    def __init__(self, path, max_len):
        self.path = path
        self.max_len = max_len
        self.input_ids = []
        self.labels_ids = []
        self.label_to_ix = {
            'O': 1, 'B-Cause': 2, 'I-Cause': 3, 'B-Effect': 4, 'I-Effect': 5, 'pad': 0}
        self.ix_to_label = {
            0: 'pad', 1: 'O', 2: 'B-Cause', 3: 'I-Cause', 4: 'B-Effect', 5: 'I-Effect'
        }
        self.tokenizer = None

    def label_trans(self, labels, ix_to_label=True):
        '''
        :param labels: 待转换的序列，['O', 'B-Cause', 'O', 'O']或[1, 2, 3, 4]
        :param ix_to_label: 默认将id转换为标签， 为false将标签转换为id
        :return: 转换后的序列
        '''
        l = []
        try:
            if ix_to_label:
                for label in labels:
                    l.append(self.ix_to_label[label])
            else:
                for label in labels:
                    l.append(self.label_to_ix[label])
        except KeyError:
            print('请确认id或标签是否正确')
        else:
            return l

    def load_data(self):
        '''读路径中的文件，返回句子列表和标签列表'''
        sentences = []
        labels = []
        with open(self.path) as f:
            for index, line in enumerate(f.readlines()):
                if index % 2 == 0:
                    sentences.append(line.strip())  # sentences:['sentence a', 'sentence b', ...]
                else:
                    labels.append(line.strip())  # labels: ['O O', 'O O', ...]
        return sentences, labels

    def tokenize_data_ids(self):
        pretained_weights = './pretrained_model'
        tokenizer = BertTokenizer.from_pretrained(pretained_weights)
        self.tokenizer = tokenizer

        sentences, labels = self.load_data()
        #pad
        for sentence in sentences:
            split_tokens = sentence.split()
            if len(split_tokens) > self.max_len:
                split_tokens = split_tokens[:self.max_len]
            else:
                # 加入pad字符
                split_tokens += ['[PAD]'] * (self.max_len - len(split_tokens))
                # input_ids:[[10,20,0,0,0],[23,43,34,0,0]...]
            self.input_ids.append(
                tokenizer.encode(split_tokens, add_special_tokens=False))

        for label in labels:
            split_label = label.split()
            split_label_id = []
            for index, single_label in enumerate(split_label):
                split_label_id.append(self.label_to_ix[single_label])
            if len(split_label_id) > self.max_len:
                split_label_id = split_label_id[:self.max_len]
            else:
                # pad
                split_label_id += [0] * (self.max_len - len(split_label_id))
            self.labels_ids.append(split_label_id)  # label_ids[[1,1,0,0,0], [1,4,2,0,0], ...]
        return self.input_ids, self.labels_ids


# batch_data = np.random.randn(2, 5)
# pad = np.zeros([2, 3])
# batch_data = np.append(batch_data, pad, axis=1)
# batch_data = torch.tensor(batch_data)
# batch_data[0][4] = 0
def get_attention_mask(batch_data):
    # 输入是一个batch的数据，shape为（bs，seq_len）
    #        返回一个对应得attention_mask
    attention_mask = torch.ones_like(batch_data)
    bs, seq_len = batch_data.size()
    sent_len_list = []

    for sent_ix in range(bs):
        for word_ix in range(seq_len):
            if batch_data[sent_ix][word_ix] == 0:
                sent_len_list.append(word_ix)
                break
    for sent_ix in range(bs):
        attention_mask[sent_ix, sent_len_list[sent_ix]:] = 0

    return attention_mask
    # print(attention_mask)


# get_attention_mask(batch_data)

