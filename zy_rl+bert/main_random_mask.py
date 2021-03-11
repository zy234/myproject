# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
from torch.utils import data
from seqeval.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from zy_utils import NerDataset, pad, idx2tag
from zy_BertFinetun import MyBertModel
from easy_dqn import easy_dqn, DQN

MASK_PROB = 0.8
MASK_NUM = 4
MASK_ID = 103

def train(model, iterator, optimizer, criterion, device):
    model.train()
    loss = 0
    for i, batch in enumerate(tqdm(iterator)):
        words, x, is_heads, tags, y, seqlens = batch
        x = x.to(device)
        y = y.to(device)
        _y = y # for monitoring
        optimizer.zero_grad()

        for bs in range(len(x)):
            if np.random.random() > MASK_PROB:
                pass
            else:
                mask_idx = [np.random.randint(0, seqlens[bs]) for i in range(MASK_NUM)]
                x[bs][mask_idx] = MASK_ID

        outputs = model(x, y)  # logits: (N, T, VOCAB), y: (N, T)

        if len(outputs) == 3:
            loss = outputs[0]
            loss.backward()

        optimizer.step()

    print(f"loss: {loss.item()}")


def eval(model, iterator, f, device, save_result=False):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch
            x = x.to(device)
            # y = y.to(device)

            logits, _ = model(x)  # y_hat: (N, T)
            y_hat = torch.argmax(logits, 2)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    # gets results and save
    y_true, y_pred = [], []
    with open("temp", 'w', encoding='utf-8') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds)==len(words.split())==len(tags.split())

            temp_t, temp_p = [], []
            for w, t, p in zip(words.split()[0:-1], tags.split()[0:-1], preds[0:-1]):
                fout.write(f"{w} {t} {p}\n")
                temp_t.append(t)
                temp_p.append(p)

            y_pred.append(temp_p)
            y_true.append(temp_t)
            fout.write("\n")


    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # os.remove("temp")

    print("precision=%.2f"%precision)
    print("recall=%.2f"%recall)
    print("f1=%.3f"%f1)
    return precision, recall, f1

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.00001)
    parser.add_argument("--n_epochs", type=int, default=120)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true", default=True)
    # parser.add_argument("--top_rnns", dest="top_rnns", action="store_true")
    parser.add_argument("--logdir", type=str, default="./result/bert+random_mask")
    parser.add_argument("--trainset", type=str, default="./data/train.txt")
    parser.add_argument("--validset", type=str, default="./data/test.txt")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    hp = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MyBertModel().cuda()
    dqn = DQN()
    print('Initial model Done')
    # model = nn.DataParallel(model)

    train_dataset = NerDataset(hp.trainset)
    eval_dataset = NerDataset(hp.validset)
    print('Load Data Done')

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=pad)
    eval_iter = data.DataLoader(dataset=eval_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr = hp.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    best_f1 = 0.
    if hp.checkpoint_dir:
        print("model load from checkpoint dir " + hp.checkpoint_dir)
        checkpoint = torch.load(hp.checkpoint_dir)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # best_f1 = checkpoint["best_f1"]

    print('Start Train...,')
    best_result = best_f1
    patience = 0
    for epoch in range(hp.n_epochs):  # 每个epoch对dev集进行测试
        patience += 1
        save_result = False

        train(model, train_iter, optimizer, criterion, device)

        print(f"=========eval at epoch={epoch}=========")
        if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
        fname = os.path.join(hp.logdir, 'result' + str(epoch))
        precision, recall, f1 = eval(model, eval_iter, fname, device)

        if best_result < f1:
            best_result = f1
            patience = 0
            save_result = True

        if save_result:
            # 存储eval结果
            print("save current best model at {}, best f1 is {}.".format(fname, best_result))

            # 删除上一步保存的文件
            for file in os.listdir(hp.logdir):
                if file.startswith('result'):
                    os.remove(hp.logdir + '/' + file)

            final = fname + ".F%.3f_R%.2f_P%.2f" % (f1, recall, precision)
            with open(final, 'w', encoding='utf-8') as fout:
                result = open("temp", "r", encoding='utf-8').read()
                fout.write(f"{result}\n")

                fout.write(f"precision={precision}\n")
                fout.write(f"recall={recall}\n")
                fout.write(f"f1={f1}\n")

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_result
            }, f"{fname}.pt")
            print(f"weights were saved to {fname}.pt")

        if patience > 20:
            print("early stop at epoch {}, best f1 is {}, other results in file.".format(epoch, best_result))
            break
    print("training is end, best f1 is {}, other results in file.".format(best_result))