#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from tqdm import tqdm
import re

def post_dictionary(submit_file):
    train = pd.read_csv('./datasets/round_1_2_train_data.csv')
    test = pd.read_csv('./datasets/round2_test.csv')

    train = train.fillna('')
    test = test.fillna('')

    # 维护训练集中{实体：负面实体}字典
    result = dict()
    for negative, entity, key_entity in zip(train['negative'].tolist(), train['entity'].tolist(), train['key_entity'].tolist()):
        if entity == '':
            continue
        else:
            entity = entity.split(';')
            entity = ';'.join(list(sorted(entity)))
            if entity not in result.keys():
                result[entity] = {0: set(), 1: set()}
            if key_entity != '':
                key_entity = key_entity.split(';')
                key_entity = ';'.join(list(sorted(key_entity)))
                result[entity][negative].add(key_entity)
            else:
                result[entity][negative].add(entity)

    # 测试集的{实体：set()}字典
    test_result = dict()
    for entity in test['entity'].tolist():
        if entity != '':
            entity = entity.split(';')
            entity = ';'.join(list(sorted(entity)))
            if entity not in test_result.keys():
                test_result[entity] = set()

    # 取训练集和测试集共同出现的实体对
    intersection = set(result.keys()).intersection(test_result.keys())
    one2one = dict()

    '''
    只取实体对和负面实体对一一对应的，比如
        富金利;理财时代,"{0: {'富金利;理财时代'}, 1: set()} ———— 表示只要出现“富金利;理财时代”实体对，那么这两个实体就一对是非负面的
        旺旺贷;旺贷,"{0: set(), 1: {'旺旺贷'}}" ———— 表示只要出现“旺旺贷;旺贷”实体对，只存在一个负面实体“旺旺贷”
    一对多的过滤掉，比如下面的例子
    小资钱包;资易贷,"{0: set(), 1: {'小资钱包;资易贷', '小资钱包'}}" ————表示出现“小资钱包;资易贷”实体对，可能存在两个负面实体组：“小资钱包;资易贷”和“小资钱包”
    '''
    for intersection_id in intersection:
        # 实体：负面实体对
        if len(result[intersection_id][0]) == 0 and len(result[intersection_id][1]) == 1:
            one2one[intersection_id] = result[intersection_id]
        # 实体：非负面实体对
        elif len(result[intersection_id][0]) == 1 and len(result[intersection_id][1]) == 0:
            one2one[intersection_id] = result[intersection_id]

    submit = pd.read_csv(submit_file)
    submit['title'] = test['title']
    submit['text'] = test['text']
    submit['entity'] = test['entity']
    submit = submit.fillna('')
    negatives = []
    key_entities = []
    for n, e, ke, text in tqdm(zip(submit['negative'].tolist(), submit['entity'].tolist(), submit['key_entity'].tolist(), submit['text'].tolist())):
        if type(e) == float or e == '':
            negatives.append(n)
            key_entities.append(ke)
            continue
        e = e.split(';')
        e = ';'.join(list(sorted(e)))
        if e in one2one.keys():
            # 实体：负面实体对
            if len(one2one[e][0]) == 0 and len(one2one[e][1]) == 1:
                # 如果之前的情感值为1，则替换之前的负面实体对，否则不进行替换
                if n == 1:
                    negatives.append(1)
                    key_entities.append(list(one2one[e][1])[0])
                else:
                    negatives.append(n)
                    key_entities.append(ke)
            # 实体：非负面实体对
            elif len(one2one[e][0]) == 1 and len(one2one[e][1]) == 0:
                if n == 0:
                    negatives.append(0)
                    key_entities.append('')
                else:
                    negatives.append(n)
                    key_entities.append(ke)
        else:
            negatives.append(n)
            key_entities.append(ke)
    submit['negative'] = negatives
    submit['key_entity'] = key_entities

    # submit[['id', 'negative', 'key_entity']].to_csv('./result/voting.csv', index=False)
    return submit

def paralleling_solve(test):
    """
    处理存在并列关系的金融实体，如果并列的实体70%都被模型预测是负面实体，那么剩余的实体也将被认为是负面实体
    :param test:
    :return:
    """
    key_entitys = []
    negs = []
    for index, text in enumerate(test['text']):
        title = test['title'][index]
        negative = test['negative'][index]
        key_entity = test['key_entity'][index]
        text = str(title) + str(text)
        if negative == 1:
            entity = test['entity'][index]
            entity_splits = list(set(str(entity).split(';')))
            entity_splits.sort(key=lambda i: len(i), reverse=True)
            key_entity_splits = set(str(key_entity).split(';'))
            new_key_entity_splits = key_entity_splits
            pattern_dun = re.compile(r'(,(.{2,10},){3,100})')  # 查找英文逗号并列
            pattern_dou = re.compile(r'(、(.{2,10}、){3,100})')  # 查找顿号并列
            pattern_zn_dou = re.compile(r'(，(.{2,10}，){3,100})')  # 查找中文逗号并列
            pattern_zn_dian = re.compile(r'(\s(.{2,10}\s){3,100})')  # 查找点号并列
            results = pattern_dun.findall(str(text)) + pattern_dou.findall(str(text)) \
                      + pattern_zn_dou.findall(str(text)) + pattern_zn_dian.findall(text)

            # 如果存在并列实体
            if len(results) > 0:
                for result in results:
                    paralleling_string = result[0]
                    for item in entity_splits:
                        if text.find(item + paralleling_string) != -1:
                            paralleling_string = item + paralleling_string
                    for item in entity_splits:
                        if text.find(paralleling_string + item) != -1:
                            paralleling_string = paralleling_string + item
                    paralleling_entities = set(re.split('，|、|,|\s', paralleling_string))    # 切割实体
                    if '' in paralleling_entities:
                        paralleling_entities.remove('')
                    paralleling_entities_in = []
                    for item in paralleling_entities:
                        if item in entity_splits:
                            paralleling_entities_in.append(item)
                    if len(paralleling_entities_in) > 3:    # 如果并列实体的个数超过3个，说明是数量比较大的并列实体
                        count_in_key = 0
                        for item in paralleling_entities_in:
                            if item in key_entity_splits:
                                count_in_key += 1
                        in_ratio = count_in_key / len(paralleling_entities_in)
                        if in_ratio >= 0.7: # 如果并列的实体70%都被模型预测为负面金融实体，则剩余的实体也属于负面实体
                            for item in paralleling_entities_in:
                                if item not in new_key_entity_splits:
                                    new_key_entity_splits.add(item)
            if len(key_entity_splits) > 0:
                negative = 1
                key_entity = ';'.join(new_key_entity_splits)
            else:
                negative = 0
                key_entity = ''
        key_entitys.append(key_entity)
        negs.append(negative)

    test['key_entity'] = key_entitys
    test['negative'] = negs
    return test


def find_ignore_entity(result_df):
    '''
    根据训练集中的先验，补充嵌套短实体
    :param result_df: 预测结果的DataFrame
    :return: 该规则处理完成后的DataFrame
    '''

    # Prepare
    train_df = pd.read_csv('./datasets/round_1_2_train_data.csv')
    train_df = train_df[train_df['negative'] == 1]
    es = train_df['entity']
    kes = train_df['key_entity']

    iiset = set()  # 记录会被忽略的嵌套实体对
    nnset = set()  # 记录不会被忽略的嵌套实体对

    for e, ke in zip(es, kes):
        e = e.split(';')
        ke = ke.split(';')
        if len(e) != len(ke):
            for ee in e:
                ee = ee.strip(' ')
                for kee in ke:
                    kee = kee.strip(' ')
                    if ee in kee and ee != kee:  # 判断两实体是否嵌套
                        if ee in ke:  # 判断是否有嵌套忽略
                            nnset.add((ee, kee))
                        else:
                            iiset.add((ee, kee))
    s = iiset & nnset
    nnset = nnset.difference(s)  # 去除交集，有些忽略现象在不同样本中不一致

    ignore_dict = dict()
    for e1, e2 in nnset:  # 转换为dict形式，方便后续操作
        if (e1, e2) not in s:
            if len(e1) > 0 and len(e2) > 0:
                ignore_dict[e2] = e1

    new_key_entity = []
    for i in range(len(result_df)):
        e = result_df['entity'][i]
        ke = result_df['key_entity'][i]
        n = result_df['negative'][i]
        nke = ke

        if n == 1:  # 只对预测结果为负的样本进行嵌套短实体补充
            nke = nke.split(';')
            ke = ke.split(';')
            e = e.split(';')
            for kee in ke:
                if kee in ignore_dict.keys() and ignore_dict[kee] in e:
                    nke.append(ignore_dict[kee])
            nke = list(set(nke))  # 实体去重
            nke = list(sorted(nke))
            nke = ';'.join(nke)

        new_key_entity.append(nke)

    result_df['key_entity'] = new_key_entity
    return result_df


if __name__ == '__main__':
    submit_path = './submit/Fusion_model_test_predict.csv'

    submit = post_dictionary(submit_path)
    # 处理实体并列
    submit = paralleling_solve(submit)
    # 根据训练集先验补充短实体
    submit = find_ignore_entity(submit)

    submit[['id', 'negative', 'key_entity']].to_csv('./submit/best_result.csv', index=False)
