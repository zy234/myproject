import torch
import torch.nn as nn
import torch.optim as optim
# from model_evaluation_utils import display_model_performance_metrics
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import argparse
import os

from data_load import MyDataSet, get_attention_mask
from distil_for_tagging import DistilForTagging
from eval_metrics import metrics, eval_model

parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--weight", type=list, default=[1, 1, 19, 370, 19, 370],
                    help="列表需要有六个权重对应六个标签")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--do_eval", type=bool, default=True)
parser.add_argument("--optim", type=str, default='adam',
                    help="必须是[adam, SGD]之一")

hp = parser.parse_args()

patience = 3  # early stopping
# train_input_ids = torch.tensor([tokenizer.encode(["here is some text to encode","there is another"],\
#                                            # 加了3个
#                                            add_special_tokens=False,\
#                                            # 不会自动pad
#                                            max_length=4)])
# 读数据
train_data_set = MyDataSet('./data/new_SEtrain_set.csv', max_len=110)
train_input_ids, train_label_ids = train_data_set.tokenize_data_ids()
test_data_set = MyDataSet('./data/log_SEtest.csv', max_len=110)
test_input_ids, test_label_ids = test_data_set.tokenize_data_ids()

model = DistilForTagging.from_pretrained('./pretrained_model/')
if hp.optim not in ['SGD', 'adam']:
    print('优化器不再可选择的范围内')
else:
    if hp.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=hp.lr)
    elif hp.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=hp.lr)

#数据集分割
#train_train_input_ids, test_train_input_ids, train_label_ids, test_label_ids = train_test_split(train_input_ids, train_labels_ids)
print('-'*50)
print('数据加载完毕')
print('训练集数量：{}， 测试集数量：{}'.format(len(train_input_ids), len(test_input_ids)))
# labels = torch.tensor([1, 2, 3, 0, 1, 3])

if hp.debug:
    hp.batch_size = 2
    train_input_ids = train_input_ids[:10]
    train_label_ids = train_label_ids[:10]
    test_input_ids = test_input_ids[:10]
    test_label_ids = test_label_ids[:10]
    # test_input_ids = test_input_ids[:10]
    # test_label_ids = test_label_ids[:10]

train_input_ids = torch.tensor(train_input_ids)
train_label_ids = torch.tensor(train_label_ids)
test_input_ids = torch.tensor(test_input_ids)
test_label_ids = torch.tensor(test_label_ids)

dataset = TensorDataset(train_input_ids, train_label_ids)
train_loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, drop_last=True)
print('-'*50)
print('训练开始, batch_size为：{}, epoch为{}'.format(hp.batch_size, hp.epochs))
best_model_dict = {}
best_optimizer_dict = {}
best_cm = []
best_f1 = 0
total_loss = 0

for epoch in range(hp.epochs):
    pat = 0  # early stopping
    iter_num = 0
    total_loss = 0

    model.train()  # 目的是开启dropout
    if torch.cuda.is_available():
        model.cuda()  # 调用gpu 实际上是把参数放入GPU

    print('epoch:{}'.format(epoch))
    for i, data in enumerate(train_loader):
        batch_input_ids, labels = data
        if torch.cuda.is_available():
            batch_input_ids = batch_input_ids.to('cuda')
            labels = labels.to('cuda')
            weight = torch.tensor(hp.weight, dtype=torch.float32).to('cuda')
            attention_mask = get_attention_mask(batch_input_ids).to('cuda')
        else:
            weight = torch.tensor(hp.weight, dtype=torch.float32)
            attention_mask = get_attention_mask(batch_input_ids)
        # 梯度清零
        optimizer.zero_grad()
        # 将数据放入模型
        outputs = model(batch_input_ids, labels=labels, weight=weight, attention_mask=attention_mask)
        loss, scores = outputs[:2]

        total_loss += loss
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print('迭代{}次，平均loss为{}'.format(i * hp.batch_size, loss))
        iter_num = i
    print('epoch{}训练完毕，loss 为：{}'.format(epoch, total_loss / (iter_num + 1)))

    # 测试
    if hp.do_eval:
        model.eval()  # 关闭dropout
        t_p, t_r, t_f1, t_cm = eval_model(model, train_input_ids, train_label_ids)
        p, r, f1, cm = eval_model(model, test_input_ids, test_label_ids)

        print("训练集因果的平均precision:{}, recall:{}, f1:{}".format(t_p, t_r, t_f1))
        print("测试集因果的平均precision:{}, recall:{}, f1:{}".format(p, r, f1))
        print("训练集混淆矩阵：\n{}".format(t_cm))
        print("测试集混淆矩阵：\n{}".format(cm))

        if f1 > best_f1:
            best_f1 = f1
            pat = 0
            best_model_dict = model.state_dict().copy()
            best_optimizer_dict = optimizer.state_dict().copy()
            best_cm = cm
        else:
            pat += 1

    if pat > patience:  # 如果超过patience，不再训练
        break

    print('-' * 50)


print('-' * 50)
print('训练结束，保存模型')


# 保存模型
if hp.debug:
    pass
else:
    if not os.path.exists('./checkpoint'):
        os.makedirs('./checkpoint')
    print('-' * 50)
    print('训练结束，最好的f1值为{}， 混淆矩阵为{}'.format(best_f1, best_cm))
    torch.save({
        'model_state_dict': best_model_dict,
        'optimizer_state_dict': best_optimizer_dict,
        'total_loss': total_loss
    }, './checkpoint/' + time.strftime("%y_%m_%d_%H_%M_%S", time.localtime()) + '.pt')
    print('模型保存完毕')
# 加载模型时记得用eval()函数设置dropout和batchNomalization



