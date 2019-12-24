#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
该代码主要执行以下五个模型的融合：
1.model_1_bert_att_drop_42.py
2.model_2_bert_att_drop_further_pretrain.py
3.model_3_roberte_wwm_ext_att_drop_42.py.py
4.model_4_bert_att_drop_420.py
5.model_5_bert_att_drop_1001001.py

融合方法：五个模型概率求平均
"""


import numpy as np
from tqdm import tqdm
import time
import logging
import os
import pandas as pd
from sklearn.metrics import f1_score

# 创建一个logger
file_path = './log/'
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
timestamp = time.strftime("%Y.%m.%d_%H.%M.%S", time.localtime())
fh = logging.FileHandler(file_path + 'log_fusion_model.txt')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

file_name = 'Fusion_model_6'


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


def read_examples(input_file, is_training):
    df = pd.read_csv(input_file)
    if not is_training:
        df['negative'] = np.zeros(len(df), dtype=np.int64)
    examples = []
    for val in df[['id', 'text', 'entity', 'negative']].values:
        examples.append(InputExample(id=val[0], text=val[1], entity=val[2], label=val[3]))
    return examples, df


def postprocess(raw, df, prefix=''):
    """
   将多条预测结果数据拼接成一条
   :param raw:
   :param df:
   :param prefix:
   :return:
   """
    negatives = []
    key_entities = []

    for raw_id in tqdm(raw['id'].tolist()):
        result = df[df['id'] == raw_id]
        if len(result) > 0:
            negative = 0
            key_entity = []
            for n, e in zip(result[prefix+'negative'].tolist(), result['entity']):
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

    raw[prefix+'negative'] = negatives
    raw[prefix+'key_entity'] = key_entities
    return raw


def metric(train):
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
    logger.info('precission: %.8f, recall: %.8f, f1: %.8f' % (A/B, A/C, entities_f1))
    return 0.4*negative_f1, 0.6*entities_f1, 0.4*negative_f1 + 0.6*entities_f1


if __name__ == '__main__':

    # 加载数据
    train_examples, train_df = read_examples('./datasets/preprocess_round_1_2_train_data.csv', is_training=True)
    test_examples, test_df = read_examples('./datasets/preprocess_round2_test.csv', is_training=False)
    raw_train = pd.read_csv('./datasets/round_1_2_train_data.csv')
    raw_test = pd.read_csv('./datasets/round2_test.csv')

    # 计算训练集的平均融合的概率
    oof_train_total = 0.
    for i, file_name in enumerate(sorted(os.listdir('./submit/train_prob'))):
        file = os.path.join('./submit/train_prob', file_name)
        oof_train = np.loadtxt(file)
        oof_train_total += oof_train
    oof_train_ave = oof_train_total / 5

    # 计算测试集的平均融合的概率
    oof_test_total = 0.
    for i, file_name in enumerate(sorted(os.listdir('./submit/test_prob'))):
        file = os.path.join('./submit/test_prob', file_name)
        oof_test = np.loadtxt(file)
        oof_test_total += oof_test
    oof_test_ave = oof_test_total / 5

    labels = train_df['negative'].astype(int).values
    train_df['pred_negative'] = np.argmax(oof_train_ave, axis=1)
    test_df['negative'] = np.argmax(oof_test_ave, axis=1)

    pred_train = postprocess(raw_train, train_df, prefix='pred_')
    pred_train.to_csv('./submit/train_5_model_ave_predict.csv', index=False)
    negative_f1, entity_f1, weight_f1 = metric(pred_train)
    logger.info('negative_f1: %.8f, entity_f1: %.8f, weight_f1: %.8f\n' %
                    (negative_f1, entity_f1, weight_f1))

    submit = postprocess(raw_test, test_df)
    submit[['id', 'negative', 'key_entity']].to_csv('./submit/Fusion_model_test_predict.csv', index=False)
