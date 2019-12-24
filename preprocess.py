#!/usr/bin/python
# -*- coding: utf-8 -*-

import csv
import re
import distance
import sys
import pandas as pd

"""
该文件为预处理文件，主要进行以下几个预处理：
1.清除无用的信息
2.如果待预测实体不在文本的前512中，将预测实体所在的文本提前到前512中
3.将文本中出现的实体，添加上“<”，“>”，来突出实体
4.将含有多条实体的数据切分成多条只预测一个实体的数据
5.截断文本（取前512）
"""

max_seq_length = 512


def clean_space(text):
    """"
    处理多余的空格
    """
    match_regex = re.compile(u'[\u4e00-\u9fa5。\.,，:：《》、\(\)（）]{1} +(?<![a-zA-Z])|\d+ +| +\d+|[a-z A-Z]+')
    should_replace_list = match_regex.findall(text)
    order_replace_list = sorted(should_replace_list,key=lambda i:len(i),reverse=True)
    for i in order_replace_list:
        if i == u' ':
            continue
        new_i = i.strip()
        text = text.replace(i,new_i)
    return text


def clean(text):
    """
    清除无用的信息
    :param text:
    :return:
    """
    if type(text) != str:
        return text
    text = clean_space(text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\{IMG:[0-9]+\}', '', text)
    text = re.sub(r'\?{2,}', '', text)
    text = re.sub(r'[0-9a-zA-Z]{100,}', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http(s?):[/a-zA-Z0-9.=&?_#]+', '', text)
    text = re.sub(r'&ldquo;', '', text)
    text = re.sub(r'&rdquo;', '', text)
    text = re.sub(r'—{5,}', '', text)

    text = re.sub(r'？{2,}', '', text)
    text = re.sub(r'●', '', text)
    text = re.sub(r'【图】', '', text)
    text = re.sub(r'[0-9]+[-|.|/|年][0-9]{2}[-|.|/|月][0-9]{2}日?', '', text)
    text = re.sub(r'&nbsp;', '', text)
    text = re.sub(r'[0-9]{15,}', '', text)
    text = re.sub(r'&quot;', '', text)
    return text


def match(title, text):
    if type(title) == float or type(text) == float:
        return 1
    strs = list(distance.lcsubstrings(title, text))
    if len(strs) > 0:
        return len(strs[0]) / len(title)
    else:
        return 0


def get_entity_sentence(entity, text):
    """
    找出预测实体所在的文本
    :param entity:
    :param text:
    :return:
    """
    index = text.find(entity)
    if index > 512:
        split_symbol = ['.', '。', '!', '！', '?', '？', '；', ';', ' ', '\t', '\n']
        for i in range(50):
            if text[index - 20 - i - 1] in split_symbol:
                return ''.join(text[index - 20 - i: len(text)])
    else:
        return None


def process(filename, data_path, mode='train'):
    """
    数据预处理主函数
    :param filename:
    :param mode:
    :return:
    """
    header = []
    rows = []
    with open(filename, 'r', encoding='utf-8-sig') as f:
        f_csv = csv.reader(f)
        for i, row in enumerate(f_csv):
            if i == 0:
                if mode == 'train':
                    header = [row[0], row[2], 'clean_entity'] + row[3:5]
                else:
                    header = [row[0], row[2], 'clean_entity', row[3]]
            else:
                text_id, title, text = row[0:3]
                if row[3] != '':
                    entities = row[3].split(';')
                else:
                    entities = ['']
                if mode == 'train':
                    if row[5] != '':
                        key_entities = clean(row[5]).split(';')
                    else:
                        key_entities = []
                if len(text) == 0 or type(text) == float:
                    text = title
                text = clean(text)
                for index, entity in enumerate(entities):
                    clean_entity = clean(entity)
                    entity_in_text = get_entity_sentence(clean_entity, text)
                    new_text = text
                    if entity_in_text != None:
                        new_text = entity_in_text
                    if clean_entity not in text:
                        new_text = clean_entity + '。' + text
                    if entity == '' or entity != entity or type(entity) == float:
                        continue
                    if clean_entity == '' or clean_entity != clean_entity or type(clean_entity) == float:
                        continue
                    new_text = new_text.replace(clean_entity, '<' + clean_entity + '>')
                    if mode == 'train':
                        if clean_entity in key_entities:
                            negative = 1
                        else:
                            negative = 0
                    if mode == 'train':
                        new_entities = entities.copy()
                        new_entities[index] = '<' + clean_entity + '>'

                        new_entities_sub = []
                        for i in range(index-10, index + 10):
                            if i >= 0 and i < len(new_entities):
                                new_entities_sub.append(new_entities[i])
                        new_entity = ';'.join(new_entities_sub)
                        rows.append([text_id, new_text, new_entity, entity, negative])
                    else:
                        new_entities = entities.copy()
                        new_entities[index] = '<' + clean_entity + '>'
                        new_entities_sub = []
                        for i in range(index-10, index + 10):
                            if i >= 0 and i < len(new_entities):
                                new_entities_sub.append(new_entities[i])
                        new_entity = ';'.join(new_entities_sub)
                        rows.append([text_id, new_text, new_entity, entity])

    with open(data_path, 'w', encoding='utf-8-sig', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(header)
        for row in rows:
            f_csv.writerow(row)


def merge_round_1_round_2():
    """
    合并初赛和复赛的数据
    :return:
    """
    train_round_1 = pd.read_csv('./datasets/Train_Data.csv')
    train_round_2 = pd.read_csv('./datasets/Round2_train.csv')
    merge_data = pd.concat([train_round_2, train_round_1], ignore_index=True)
    merge_data = merge_data.drop_duplicates(keep='first', subset=['text'])
    merge_data.to_csv('./datasets/round_1_2_train_data.csv', encoding='utf-8', index=False)


if __name__ == '__main__':
    merge_round_1_round_2()

    if len(sys.argv) < 2:

        # 训练集的预处理
        process('./datasets/round_1_2_train_data.csv', './datasets/preprocess_round_1_2_train_data.csv', 'train')
        # 测试集的预处理
        process('./datasets/round2_test.csv', './datasets/preprocess_round2_test.csv', 'test')

    else:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
        # 训练集的预处理
        process(train_path, './datasets/preprocess_round_1_2_train_data.csv', 'train')
        # 测试集的预处理
        process(test_path, './datasets/preprocess_round2_test.csv', 'test')

