#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import torch
import numpy as np
from tqdm import tqdm
import time
import logging
from sklearn.model_selection import StratifiedKFold
import os
import pandas as pd
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from sklearn.metrics import accuracy_score, f1_score
from pytorch_transformers.modeling_bert import BertModel, BertConfig
from pytorch_transformers import AdamW
from pytorch_transformers.tokenization_bert import BertTokenizer
import torch.nn as nn
import torch.nn.functional as F

"""
该执行文件主要用于训练基于Bert预训练模型-Model_1_bert_att_drop_42,并生成测试集的预测概率文件
"""


# 设置参数及文件路径
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # 程序可调用的GPU的ID
max_seq_length = 512  # 输入文本最大长度
learning_rate = 1e-5  # 模型学习率
num_epochs = 10  # 训练最大迭代次数
batch_size = 8  # 训练时每个batch中的样本数
patience = 5   # 早停轮数
file_name = 'model_1_bert_42'  # 指定输出文件的名字
model_name_or_path = './pretrain_weight/bert/'  # 预训练模型权重载入路径
train_input = './datasets/preprocess_round_1_2_train_data.csv'  # 完成预处理的训练集载入路径
test_input = './datasets/preprocess_round2_test.csv'  # 完成预处理的测试集载入路径
raw_train = './datasets/round_1_2_train_data.csv'  # 原始训练集载入路径
raw_test = './datasets/round2_test.csv'  # 原始测试集载入路径  
random_seed = 42  # 随机种子

def seed_everything(seed=random_seed):
    '''
    固定随机种子
    :param random_seed: 随机种子数目
    :return: 
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything()

# 创建一个logger
file_path = './log/'
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(file_path + 'log_model1.txt')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, id, text, entity=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.id = id
        self.text = text
        self.entity = entity
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

                 ):
        self.example_id = example_id
        _, input_ids, input_mask, segment_ids = choices_features[0]
        self.choices_features = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'segment_ids': segment_ids
        }
        self.label = label


def read_examples(input_file, is_training):
    df = pd.read_csv(input_file)
    if not is_training:
        df['negative'] = np.zeros(len(df), dtype=np.int64)
    examples = []
    for val in df[['id', 'text', 'clean_entity', 'negative']].values:
        if type(val[2]) == float:
            print(val[0])
        examples.append(InputExample(id=val[0], text=val[1], entity=val[2], label=val[3]))
    return examples, df


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 is_training):
    # 将文本输入样例，转换为数字特征，用于模型计算
    features = []
    for example_index, example in enumerate(examples):

        text = tokenizer.tokenize(example.text)
        entity = tokenizer.tokenize(example.entity)
        MAX_TEXT_LEN = max_seq_length - len(entity) - 3
        text = text[:MAX_TEXT_LEN]

        choices_features = []

        tokens = ["[CLS]"] + text + ["[SEP]"] + entity + ["[SEP]"]
        segment_ids = [0] * (len(text) + 2) + [1] * (len(entity) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += ([0] * padding_length)
        input_mask += ([0] * padding_length)
        segment_ids += ([0] * padding_length)
        choices_features.append((tokens, input_ids, input_mask, segment_ids))

        label = example.label
        if example_index < 1 and is_training:
            logger.info("*** Example ***")
            logger.info("idx: {}".format(example_index))
            logger.info("id: {}".format(example.id))
            logger.info("tokens: {}".format(' '.join(tokens).replace('\u2581', '_')))
            logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
            logger.info("input_mask: {}".format(len(input_mask)))
            logger.info("segment_ids: {}".format(len(segment_ids)))
            logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                example_id=example.id,
                choices_features=choices_features,
                label=label
            )
        )
    return features


def select_field(features, field):
    return [
        feature.choices_features[field] for feature in features
    ]


class NeuralNet(nn.Module):
    def __init__(self, model_name_or_path, hidden_size=768, num_class=2):
        super(NeuralNet, self).__init__()

        self.config = BertConfig.from_pretrained(model_name_or_path, num_labels=4)
        self.config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(model_name_or_path, config=self.config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.weights = nn.Parameter(torch.rand(13, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, input_ids, input_mask, segment_ids):
        last_hidden_states, pool, all_hidden_states = self.bert(input_ids, token_type_ids=segment_ids,
                                                                attention_mask=input_mask)
        batch_size = input_ids.shape[0]
        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(
            13, batch_size, 1, 768)
        atten = torch.sum(ht_cls * self.weights.view(
            13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.fc(dropout(feature))
            else:
                h += self.fc(dropout(feature))
        h = h / len(self.dropouts)
        return h


def metric(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    return acc, f1


def postprocess(raw, df, prefix=''):
    # 初步的后处理。包括：相同ID结果合并，取最长实体，去除重复实体
    negatives = []
    key_entities = []

    for raw_id in tqdm(raw['id'].tolist()):
        result = df[df['id'] == raw_id]
        if len(result) > 0:
            negative = 0
            key_entity = []
            for n, e in zip(result[prefix + 'negative'].tolist(), result['entity']):
                if len(e) < 2:
                    continue
                if '?' in e:
                    n = 1
                if n == 1:
                    negative = 1
                    repeat = False
                    for k_e in key_entity.copy():
                        if e in k_e:
                            repeat = True
                            break
                        elif k_e in e:
                            key_entity.remove(k_e)
                            key_entity.append(e)
                            repeat = True
                            break
                    if not repeat:
                        key_entity.append(e)
            negatives.append(negative)
            key_entities.append(';'.join(key_entity))
        else:
            negatives.append(0)
            key_entities.append('')

    raw[prefix + 'negative'] = negatives
    raw[prefix + 'key_entity'] = key_entities
    return raw


def metric_weight(train):
    # 计算情感和实体的带权重F1值
    negative_true = train['negative'].tolist()
    negative_pred = train['pred_negative'].tolist()
    negative_f1 = f1_score(negative_true, negative_pred)

    key_entities_true = train['key_entity'].tolist()
    key_entities_pred = train['pred_key_entity'].tolist()
    A, B, C = 1e-10, 1e-10, 1e-10
    for e_true, e_pred in zip(key_entities_true, key_entities_pred):
        if type(e_true) == float:
            e_true = ''
        if type(e_pred) == float:
            e_pred = ''
        e_true = set(e_true.split(';'))
        e_pred = set(e_pred.split(';'))
        A += len(e_true & e_pred)
        B += len(e_pred)
        C += len(e_true)
    entities_f1 = 2 * A / (B + C)
    return 0.4 * negative_f1, 0.6 * entities_f1, 0.4 * negative_f1 + 0.6 * entities_f1


# 加载数据
tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
train_examples, train_df = read_examples(train_input, is_training=True)
labels = train_df['negative'].astype(int).values
train_features = convert_examples_to_features(
    train_examples, tokenizer, max_seq_length, True)
all_input_ids = np.array(select_field(train_features, 'input_ids'))
logger.info('shape: {}'.format(all_input_ids.shape))
all_input_mask = np.array(select_field(train_features, 'input_mask'))
all_segment_ids = np.array(select_field(train_features, 'segment_ids'))
all_label = np.array([f.label for f in train_features])

test_examples, test_df = read_examples(test_input, is_training=False)
test_features = convert_examples_to_features(
    test_examples, tokenizer, max_seq_length, True)
test_input_ids = torch.tensor(select_field(test_features, 'input_ids'), dtype=torch.long)
test_input_mask = torch.tensor(select_field(test_features, 'input_mask'), dtype=torch.long)
test_segment_ids = torch.tensor(select_field(test_features, 'segment_ids'), dtype=torch.long)

# 七折交叉训练
skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=random_seed)
oof_train = np.zeros((len(train_df), 2), dtype=np.float32)
oof_test = np.zeros((len(test_df), 2), dtype=np.float32)

for fold, (train_index, valid_index) in enumerate(skf.split(all_label, all_label)):
    logger.info('================     fold {}        ==============='.format(fold))

    # 处理模型输入数据
    train_input_ids = torch.tensor(all_input_ids[train_index], dtype=torch.long)
    train_input_mask = torch.tensor(all_input_mask[train_index], dtype=torch.long)
    train_segment_ids = torch.tensor(all_segment_ids[train_index], dtype=torch.long)
    train_label = torch.tensor(all_label[train_index], dtype=torch.long)

    valid_input_ids = torch.tensor(all_input_ids[valid_index], dtype=torch.long)
    valid_input_mask = torch.tensor(all_input_mask[valid_index], dtype=torch.long)
    valid_segment_ids = torch.tensor(all_segment_ids[valid_index], dtype=torch.long)
    valid_label = torch.tensor(all_label[valid_index], dtype=torch.long)

    train = torch.utils.data.TensorDataset(train_input_ids, train_input_mask, train_segment_ids, train_label)
    valid = torch.utils.data.TensorDataset(valid_input_ids, valid_input_mask, valid_segment_ids, valid_label)
    test = torch.utils.data.TensorDataset(test_input_ids, test_input_mask, test_segment_ids)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)

    model = NeuralNet(model_name_or_path)
    model.cuda()
    loss_fn = torch.nn.CrossEntropyLoss()

    # 优化器定义
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-6)
    model.train()

    best_f1 = 0.
    valid_best = np.zeros((valid_label.size(0), 2))

    early_stop = 0
    # for epoch in range(num_epochs):
    #     train_loss = 0.
    #     for batch in tqdm(train_loader):
    #         batch = tuple(t.cuda() for t in batch)
    #         x_ids, x_mask, x_sids, y_truth = batch
    #         y_pred = model(x_ids, x_mask, x_sids)
    #         loss = loss_fn(y_pred, y_truth)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item() / len(train_loader)
    #
    #     # 计算在验证集上的结果
    #     model.eval()
    #     val_loss = 0.
    #     valid_preds_fold = np.zeros((valid_label.size(0), 2))
    #     with torch.no_grad():
    #         for i, batch in tqdm(enumerate(valid_loader)):
    #             batch = tuple(t.cuda() for t in batch)
    #             x_ids, x_mask, x_sids, y_truth = batch
    #             y_pred = model(x_ids, x_mask, x_sids).detach()
    #             val_loss += loss_fn(y_pred, y_truth).item() / len(valid_loader)
    #             valid_preds_fold[i * batch_size:(i + 1) * batch_size] = F.softmax(y_pred, dim=1).cpu().numpy()
    #
    #     acc, f1 = metric(all_label[valid_index], np.argmax(valid_preds_fold, axis=1))
    #     if best_f1 < f1:
    #         early_stop = 0
    #         best_f1 = f1
    #         valid_best = valid_preds_fold
    #         torch.save(model.state_dict(), './model_save/bert_cv_ ' + file_name + '_{}.bin'.format(fold))
    #     else:
    #         early_stop += 1
    #     logger.info(
    #         'epoch: %d, train loss: %.8f, valid loss: %.8f, acc: %.8f, f1: %.8f, best_f1: %.8f\n' %
    #         (epoch, train_loss, val_loss, acc, f1, best_f1))
    #     torch.cuda.empty_cache()  # 每个epoch结束之后清空显存，防止显存不足
    #
    #     # 检测早停
    #     if early_stop >= patience:
    #         break

    # 得到一折模型对测试集的预测结果
    test_preds_fold = np.zeros((len(test_df), 2))
    valid_preds_fold = np.zeros((valid_label.size(0), 2))
    model.load_state_dict(torch.load('./best_model_save/bert_cv_ ' + file_name + '_{}.bin'.format(fold)))
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(valid_loader)):
            batch = tuple(t.cuda() for t in batch)
            x_ids, x_mask, x_sids, y_truth = batch
            y_pred = model(x_ids, x_mask, x_sids).detach()
            valid_preds_fold[i * batch_size:(i + 1) * batch_size] = F.softmax(y_pred, dim=1).cpu().numpy()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            batch = tuple(t.cuda() for t in batch)
            x_ids, x_mask, x_sids = batch
            y_pred = model(x_ids, x_mask, x_sids).detach()
            test_preds_fold[i * batch_size:(i + 1) * batch_size] = F.softmax(y_pred, dim=1).cpu().numpy()
    valid_best = valid_preds_fold
    oof_train[valid_index] = valid_best
    acc, f1 = metric(all_label[valid_index], np.argmax(valid_best, axis=1))
    logger.info('epoch: best, acc: %.8f, f1: %.8f, best_f1: %.8f\n' %
                (acc, f1, best_f1))
    oof_test += test_preds_fold / 7


# 保存概率文件
np.savetxt('./submit/train_prob/train_bert_' + file_name + '.txt', oof_train)
np.savetxt('./submit/test_prob/test_bert_' + file_name + '.txt', oof_test)
f1 = f1_score(labels, np.argmax(oof_train, axis=1))
logger.info(f1_score(labels, np.argmax(oof_train, axis=1)))
train_df['pred_negative'] = np.argmax(oof_train, axis=1)
test_df['negative'] = np.argmax(oof_test, axis=1)
logger.info(test_df['negative'].value_counts())

# 后处理
raw_train = pd.read_csv(raw_train)
pred_train = postprocess(raw_train, train_df, prefix='pred_')
negative_f1, entity_f1, weight_f1 = metric_weight(pred_train)
pred_train.to_csv('./submit/train_' + file_name + '.csv', index=False)
logger.info('negative_f1: %.8f, entity_f1: %.8f, weight_f1: %.8f\n' %
            (negative_f1, entity_f1, weight_f1))
submit = pd.read_csv(raw_test)
submit = postprocess(submit, test_df)
submit[['id', 'negative', 'key_entity']].to_csv('./submit/' + file_name + '_test_predict_{}.csv'.format(weight_f1),
                                                index=False)
