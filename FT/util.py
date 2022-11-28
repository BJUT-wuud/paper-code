#encoding=utf-8
import json

from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate, ManualVerbalizer, ManualTemplate, ProtoVerbalizer
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
from datasets import load_dataset, load_from_disk
from openprompt.data_utils import InputExample
from transformers import AdamW
import torch
import matplotlib.pyplot as plt
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.utils.reproduciblity import set_seed
import random

class TnewsDataset(torch.utils.data.Dataset):
    def __init__(self, train_file, val_file, test_file, split):
       datasets = {}
       datasets['train'] = load_dataset("json", data_files=train_file)
       datasets['valid'] = load_dataset("json", data_files=val_file)
       datasets['test'] = load_dataset("json", data_files=test_file)
       self.dataset = datasets[split]['train']
       print(self.dataset)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text_a']
        label = int(self.dataset[i]['label'])

        return text, label

class CnewsDataset(torch.utils.data.Dataset):

    def __init__(self, train_file, val_file, test_file,split):
       datasets = {}
       datasets['train'] = load_dataset("json", data_files=train_file)
       datasets['valid'] = load_dataset("json", data_files=val_file)
       datasets['test'] = load_dataset("json", data_files=test_file)
       self.dataset = datasets[split]['train']
       # print(self.dataset[0:10])
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        # print(self.dataset[i])
        text = self.dataset[i]['text_a']
        label = int(self.dataset[i]['label'])
        return text, label


class CsldcpDataset(torch.utils.data.Dataset):
    def __init__(self, train_file, val_file, test_file, split):
       datasets = {}
       datasets['train'] = load_dataset("json", data_files=train_file)
       datasets['valid'] = load_dataset("json", data_files=val_file)
       datasets['test'] = load_dataset("json", data_files=test_file)
       self.dataset = datasets[split]['train']
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        text = self.dataset[i]['text_a']
        label = int(self.dataset[i]['label'])

        return text, label


def get_tnew_data():
    train_file = 'E:\datasets\\few-tnews\\train_few_all.json'
    test_file = "E:\datasets\\tnews\\test_public.json"

    def trans_label(label):
        map = {100: 0, 101: 1, 102: 2, 103: 3, 104: 4, 106: 5, 107: 6, 108: 7, 109: 8, 110: 9, 112: 10, 113: 11,
               114: 12,
               115: 13, 116: 14}
        return map[label]

    my_data_set = {}

    def get_data_set(file, split):
        my_data_set[split] = []
        with open(file, encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = json.loads(line)
                input = InputExample(guid=idx, label=trans_label(int(line['label'])), text_a=line['sentence'])
                my_data_set[split].append(input)

    get_data_set(train_file, 'train')
    get_data_set(test_file, 'test')
    return my_data_set


def get_csldcp_data():
    train_file = 'E:\\datasets\\csldcp\\train_few_all.json'
    test_file = "E:\datasets\\csldcp\\test_public.json"

    label_to_index = {'材料科学与工程': 0, '作物学': 1, '口腔医学': 2, '药学': 3, '教育学': 4, '水利工程': 5, '理论经济学': 6, '食品科学与工程': 7,
                      '畜牧学/兽医学': 8, '体育学': 9, '核科学与技术': 10, '力学': 11, '园艺学': 12, '水产': 13, '法学': 14,
                      '地质学/地质资源与地质工程': 15, '石油与天然气工程': 16, '农林经济管理': 17, '信息与通信工程': 18, '图书馆、情报与档案管理': 19, '政治学': 20,
                      '电气工程': 21, '海洋科学': 22, '民族学': 23, '航空宇航科学与技术': 24, '化学/化学工程与技术': 25, '哲学': 26, '公共卫生与预防医学': 27,
                      '艺术学': 28, '农业工程': 29, '船舶与海洋工程': 30, '计算机科学与技术': 31, '冶金工程': 32, '交通运输工程': 33, '动力工程及工程热物理': 34,
                      '纺织科学与工程': 35, '建筑学': 36, '环境科学与工程': 37, '公共管理': 38, '数学': 39, '物理学': 40, '林学/林业工程': 41,
                      '心理学': 42, '历史学': 43, '工商管理': 44, '应用经济学': 45, '中医学/中药学': 46, '天文学': 47, '机械工程': 48, '土木工程': 49,
                      '光学工程': 50, '地理学': 51, '农业资源利用': 52, '生物学/生物科学与工程': 53, '兵器科学与技术': 54, '矿业工程': 55, '大气科学': 56,
                      '基础医学/临床医学': 57, '电子科学与技术': 58, '测绘科学与技术': 59, '控制科学与工程': 60, '军事学': 61, '中国语言文学': 62,
                      '新闻传播学': 63, '社会学': 64, '地球物理学': 65, '植物保护': 66}

    my_data_set = {}

    def get_data_set(file, split):
        my_data_set[split] = []
        with open(file, encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = json.loads(line)
                input = InputExample(guid=idx, label=int(label_to_index[line['label']]), text_a=line['content'])
                my_data_set[split].append(input)

    get_data_set(train_file, 'train')
    get_data_set(test_file, 'test')
    return my_data_set


def get_cnews_data():
    # 定义数据集
    train_file = 'E:\datasets\cnews\\cnews.train.txt'
    test_file = "E:\datasets\cnews\\cnews.test.txt"

    map = {'体育': 0, '财经': 1, '房产': 2, '家居': 3, '教育': 4, '科技': 5, '时尚': 6, '时政': 7, '游戏': 8, '娱乐': 9}
    my_data_set = {}

    def get_data_set(file, split):
        my_data_set[split] = []
        with open(file, encoding='utf-8') as f:
            for idx, line in enumerate(f):
                label = line[:2]
                text = line[3:]
                input = InputExample(guid=idx, label=map[label], text_a=text)
                my_data_set[split].append(input)

    get_data_set(train_file, 'train')
    get_data_set(test_file, 'test')
    return my_data_set

def few_shot_sample(seed, shot, dataset, dataset_name):
    train_file = f'./{dataset_name}/train_res.txt'
    val_file = f'./{dataset_name}/val_res.txt'
    sampler = FewShotSampler(num_examples_per_label=shot, also_sample_dev=True, num_examples_per_label_dev=shot)
    train_dataset, valid_dataset = sampler(dataset['train'], seed=seed)
    train_str = ''
    for i in range(len(train_dataset)):
        dic = train_dataset[i].to_dict()
        train_str += json.dumps(dic, ensure_ascii=False)
        train_str += '\n'
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(train_str)

    valid_str = ''
    for i in range(len(valid_dataset)):
        dic = valid_dataset[i].to_dict()
        valid_str += json.dumps(dic, ensure_ascii=False)
        valid_str += '\n'
    with open(val_file, 'w', encoding='utf-8') as f:
        f.write(valid_str)

    return train_file, val_file

def trans_test_dataset(dataset, dataset_name):
    test_dataset = dataset['test']
    test_file = f'./{dataset_name}/test_res.txt'
    test_str = ''
    for i in range(len(test_dataset)):
        dic = test_dataset[i].to_dict()
        dic_str = json.dumps(dic, ensure_ascii=False)
        test_str += dic_str
        test_str += '\n'
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_str)
    return test_file