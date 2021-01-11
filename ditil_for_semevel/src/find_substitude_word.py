import torch
from transformers import DistilBertTokenizer

from data_load import get_attention_mask

def sub_word(tokenizer, word_id, pos, ori_sent):
    # 根据id替换原句子中单词
    # 传入一个原始字符串，返回一个替换后的句子字符串
    word_id = [word_id]
    word = tokenizer.decode(word_id)

    ori_sent_split = ori_sent.split()
    ori_sent_split[pos] = word

    sentence = str()
    for w in ori_sent_split:
        sentence += w
        sentence += ' '
    sentence.strip()
    return sentence


def find_sub_word(model, train_input_ids, pos, raw_labels, raw_sentences, tokenizer):
    # 输入：预训练好的model，准确率在80以上，mask后的训练集，对应的mask位置，原始序列标注的标签
    # 返回：一个新的大数据集，包括原始语句

    db_ori_label = []  # 处理成和训练集一样的数量，双倍
    for label in raw_labels:
        db_ori_label.append(label)
        db_ori_label.append(label)
    db_raw_sentences = []
    for sent in raw_sentences:
        db_raw_sentences.append(sent)
        db_raw_sentences.append(sent)

    new_set = []
    new_labels = []
    maxk = 3
    model.eval()
    model.cuda()
    pos = pos.to('cuda')
    for index, sentence_id in enumerate(train_input_ids):
        sentence_id = sentence_id.unsqueeze(0)
        sentence_id = sentence_id.to('cuda')
        attention_mask = get_attention_mask(sentence_id).to('cuda')
        logits = model(sentence_id, pos[index], attention_mask=attention_mask)[0]  # (1, 1, voc_size30522)
        _, topk = logits.topk(maxk, dim=2, largest=True, sorted=True)
        for word_id in topk.squeeze():
            new_sentence = sub_word(tokenizer, word_id, pos[index], db_raw_sentences[index])
            new_set.append(new_sentence)
            new_labels.append(db_ori_label[index])

    return new_set, new_labels
