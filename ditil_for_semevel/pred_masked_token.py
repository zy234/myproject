import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import argparse
import time
import os

from data_load import MyDataSet, get_attention_mask
from distil_for_mask_pred import mask_input, DistilForMaskPredict
from eval_metrics import eval_masked_model
from find_substitude_word import find_sub_word

def train(hp, train_input_ids, train_label_ids, pos, model):
    optimizer = None
    if hp.optim not in ['SGD', 'adam']:
        print('优化器不再可选择的范围内')
    else:
        if hp.optim == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=hp.lr)
        elif hp.optim == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=hp.lr)

    if hp.from_checkpoint is not None:
        checkpoint_path = './checkpoint/'
        checkpoint = torch.load(checkpoint_path + hp.from_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if hp.debug:
        hp.batch_size = 1
        train_input_ids = train_input_ids[:10]
        train_label_ids = train_label_ids[:10]
        pos = pos[:10]

    # 转化为longtensor
    train_input_ids = torch.Tensor(train_input_ids).long()
    train_label_ids = torch.Tensor(train_label_ids).long()
    pos = torch.Tensor(pos).long()
    dataset = TensorDataset(train_input_ids, train_label_ids, pos)
    train_loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, drop_last=True)

    print('-' * 50)
    print('训练开始, batch_size为：{}, epoch为{}'.format(hp.batch_size, hp.epochs))
    patience = 3
    best_loss = 999999
    pat = 0  # early stopping
    for epoch in range(hp.epochs):
        iter_num = 0
        total_loss = 0

        model.train()  # 目的是开启dropout
        if torch.cuda.is_available():
            model.cuda(device='cuda:0')  # 调用gpu 实际上是把参数放入GPU

        print('epoch:{}'.format(epoch))
        for i, data in enumerate(train_loader):
            batch_input_ids, labels, batch_pos = data  # (bs,seq_len) (bs) (bs)
            if torch.cuda.is_available():
                batch_input_ids = batch_input_ids.to('cuda')
                labels = labels.to('cuda')
                batch_pos = batch_pos.to('cuda')
                attention_mask = get_attention_mask(batch_input_ids).to('cuda')
            else:
                attention_mask = get_attention_mask(batch_input_ids)
            # 梯度清零
            optimizer.zero_grad()
            # 将数据放入模型
            outputs = model(batch_input_ids, labels=labels, batch_pos=batch_pos, attention_mask=attention_mask)
            loss, scores = outputs[:2]

            total_loss += loss
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('迭代{}次，平均loss为{}'.format(i * hp.batch_size, loss))
            iter_num = i
        print('epoch{}训练完毕，loss 为：{}'.format(epoch, total_loss / (iter_num + 1)))
        if total_loss < best_loss:
            best_loss = total_loss
            pat = 0
            best_model_dict = model.state_dict().copy()
            best_optimizer_dict = optimizer.state_dict().copy()
        else:
            pat += 1

        if pat > patience:  # 如果超过patience，不再训练
            break

        # 测试
        if hp.do_eval:
            model.eval()  # 关闭dropout和batch_norm，但autograd还是在计算，只是不进行反传
            # with torch.no_grad():  # 关闭autograd功能，节省gpu和现存
            accurancy = eval_masked_model(model, train_input_ids, train_label_ids, pos)
            if accurancy > 0.8:
                # 感觉还是不要太拟合了吧
                best_model_dict = model.state_dict().copy()
                best_optimizer_dict = optimizer.state_dict().copy()
                break
            print("第{}个epoch训练结束，准确率为{}".format(epoch, accurancy))
        print('-' * 50)

    print('-' * 50)
    print('训练在第{}个epoch结束，保存模型'.format(epoch))

    # 保存模型
    if hp.debug:
        pass
    else:
        if not os.path.exists('./checkpoint'):
            os.makedirs('./checkpoint')
        torch.save({
            'model_state_dict': best_model_dict,
            'optimizer_state_dict': best_optimizer_dict,
            'total_loss': total_loss
        }, './checkpoint/' + time.strftime("%y_%m_%d_%H_%M_%S", time.localtime()) + '.pt')
    # 加载模型时记得用eval()函数设置dropout和batchNomalization


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--do_eval", type=bool, default=True)
    parser.add_argument("--optim", type=str, default='adam',
                        help="必须是[adam, SGD]之一")
    parser.add_argument("--from_checkpoint", default='20_05_31_17_37_28.pt',
                        help="如果不是none，则从给定的checkpoint文件名中读取模型参数")
    parser.add_argument("--train", default=False)
    parser.add_argument("--predict", default=True)
    hp = parser.parse_args()

    print("载入数据......")
    train_data_set = MyDataSet('./data/log_SEtrain.csv', max_len=110)
    raw_sentences, raw_labels = train_data_set.load_data()
    train_input_ids, train_label_ids = train_data_set.tokenize_data_ids()

    # 删除不含有因果的句子
    no_ce_index = []
    ce_index = []
    for index, label_ids in enumerate(train_label_ids):
        if 2 and 4 in label_ids:
            ce_index.append(index)
        else:
            no_ce_index.append(index)
    train_input_ids = np.delete(train_input_ids, no_ce_index, axis=0)
    train_label_ids = np.delete(train_label_ids, no_ce_index, axis=0)
    no_ce_sentences = np.delete(raw_sentences, ce_index, axis=0)
    no_ce_labels = np.delete(raw_labels, ce_index, axis=0)
    raw_sentences = np.delete(raw_sentences, no_ce_index, axis=0)
    raw_labels = np.delete(raw_labels, no_ce_index, axis=0)
    print("数据载入完毕")
    print('-' * 50)
    print("训练集共有{}个因果句子".format(train_input_ids.shape[0]))

    print("mask单词并将其id作为新的label......")
    # bs, seq_len    bs               bs 都是numpy数组
    train_input_ids, train_label_ids, pos = mask_input(train_input_ids, train_label_ids)
    print("处理后的数据集大小为{}，训练数据形状：{}， label形状{}".format(train_input_ids.shape[0],
                                                     train_input_ids.shape, train_label_ids.shape))
    print('-' * 50)

    model = DistilForMaskPredict.from_pretrained('./pretrained_model/')

    if hp.train:
        train(hp, train_input_ids, train_label_ids, pos, model)
    elif hp.predict:
        # 制作新的数据集
        checkpoint_path = './checkpoint/'
        checkpoint = torch.load(checkpoint_path + hp.from_checkpoint, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        train_input_ids = torch.Tensor(train_input_ids).long()
        pos = torch.Tensor(pos).long()
        new_set, new_labels = \
            find_sub_word(model, train_input_ids, pos, raw_labels, raw_sentences, train_data_set.tokenizer)
        new_set.extend(no_ce_sentences)
        new_labels.extend(no_ce_labels)
        with open('./data/new_SEtrain_set.csv', 'w') as f:
            for index in range(len(new_labels)):
                f.write(new_set[index])
                f.write('\n')
                f.write(new_labels[index])
                f.write('\n')

