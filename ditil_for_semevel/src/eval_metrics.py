import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix,accuracy_score
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from data_load import get_attention_mask

# pred = torch.tensor([[1,2,3,4,1,1,0], [1,2,2,2,3,4,1]])
# labels = torch.tensor([[1,1,3,4,1,1,0], [1,2,3,2,3,4,0]])


def metrics(pred_ids, label_ids, labels=[]):
    # 将pad的label去除

    indexs = []  # 将要删除的位置列表
    for index, label_id in enumerate(label_ids):
        if label_id == 0:
            indexs.append(index)
    label_ids = np.delete(label_ids, indexs)
    pred_ids = np.delete(pred_ids, indexs)

    # pred_ids = pred_ids.view(-1)
    # label_ids = label_ids.view(pred_ids.numel())
    f1 = f1_score(label_ids, pred_ids, labels=labels, average='micro')
    precision = precision_score(label_ids, pred_ids, labels=labels, average='micro')
    recall = recall_score(label_ids, pred_ids, labels=labels, average='micro')
    cm = confusion_matrix(label_ids, pred_ids, labels=labels)

    # label_num, pred_num = label_ids.size, pred_ids.size
    # print('数据总数为{}'.format(label_num))

    return f1, cm


def metrics_except_o(pred_ids, label_ids, labels=[]):
    # 将pad的label去除, 计算除了o标签的prf，之后平均
    # 返回评价指标的列表和详细的字典

    indexs = []  # 将要删除的位置列表
    for index, label_id in enumerate(label_ids):
        if label_id == 0:
            indexs.append(index)
    label_ids = np.delete(label_ids, indexs)
    pred_ids = np.delete(pred_ids, indexs)

    # 对每个标签求prf，存在字典中
    prf_dic = {}
    for label in labels:
        if label != 1:
            # arr_pre = np.copy(pred_ids)
            # arr_lab = np.copy(label_ids)
            arr_lab = np.where(pred_ids == label, 1, 0)
            arr_pre = np.where(label_ids == label, 1, 0)
            precision = precision_score(arr_lab, arr_pre)
            recall = recall_score(arr_lab, arr_pre)
            f1 = f1_score(arr_lab, arr_pre)
            prf_dic[label] = [precision, recall, f1]

    # 求平均
    arr = list(prf_dic.values())
    arr = np.mean(arr, axis=0)

    return arr, prf_dic


def eval_model(model, input_ids, label_ids):
    if torch.cuda.is_available():
        model.cuda()
        label_ids = label_ids.to('cuda')
        input_ids = input_ids.to('cuda')
    total_examples = np.array([0])
    total_label = np.array([0])
    dataset = TensorDataset(input_ids, label_ids)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            batch_input_ids, labels = data
            attention_mask = get_attention_mask(batch_input_ids)

            outputs = model(batch_input_ids, attention_mask=attention_mask)
            scores = outputs[0]
            scores = torch.argmax(scores, dim=2)
            # cuda上的tensor不能转为numpy
            scores = scores.cpu()
            labels = labels.cpu()
            scores = np.array(scores).flatten()
            labels = np.array(labels).flatten()
            total_examples = np.append(total_examples, scores)
            total_label = np.append(total_label, labels)
            # 删除第一个填充的位置
            total_examples = np.delete(total_examples, 0)
            total_label = np.delete(total_label, 0)

    _, cm = metrics(total_examples, total_label, labels=[1, 2, 3, 4, 5])
    # 原本是算每个分类的平均prf，但是对于I-C和I-E来说，个数太少，对整体评价影响太大
    # arr, dic = metrics_except_o(total_examples, total_label, labels=[1, 2, 3, 4, 5])
    # 根据cm计算prf，按分类为除了O之外的标签的数量计算prf
    cm_row = cm.sum(axis=1)[1:5]  # 对行求和
    cm_col = cm.sum(axis=0)[1:5]
    cm_row = cm_row.sum()
    cm_col = cm_col.sum()
    tp = 0  # 真正例的个数
    for i in range(1, 5):
        tp += cm[i, i]
    precision = tp / cm_col  # 预测为真正例的个数除以所有预测为正例的个数
    recall = tp / cm_row  # 预测为真正例的个数除以所有真正例
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, cm


def eval_masked_model(model, input_ids, label_ids, pos):
    # 只用accurancy指标了，种类太多
    if torch.cuda.is_available():
        model.cuda()
        label_ids = label_ids.to('cuda')
        input_ids = input_ids.to('cuda')
        pos = pos.to('cuda')
    total_examples = np.array([0])
    total_label = np.array([0])
    dataset = TensorDataset(input_ids, label_ids, pos)
    data_loader = DataLoader(dataset, batch_size=50, shuffle=False, drop_last=False)

    with torch.no_grad():  # 关闭autograd，节省计算和显存，
        for i, data in enumerate(data_loader):
            batch_input_ids, labels, batch_pos = data
            attention_mask = get_attention_mask(batch_input_ids)

            outputs = model(batch_input_ids, batch_pos=batch_pos, attention_mask=attention_mask)
            logits = outputs[0]
            logits = torch.argmax(logits, dim=2)
            # cuda上的tensor不能转为numpy
            logits = logits.cpu()
            labels = labels.cpu()
            logits = np.array(logits).flatten()
            labels = np.array(labels).flatten()
            total_examples = np.append(total_examples, logits)
            total_label = np.append(total_label, labels)
            # 删除第一个填充的位置
            total_examples = np.delete(total_examples, 0)
            total_label = np.delete(total_label, 0)

    accuracy = accuracy_score(total_label, total_examples)
    return accuracy
