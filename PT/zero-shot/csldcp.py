# coding=utf-8

import json

from openprompt.plms import load_plm
from openprompt.prompts import MixedTemplate, ManualVerbalizer,ManualTemplate,ProtoVerbalizer
from openprompt import PromptDataLoader
from openprompt import PromptForClassification
from datasets import load_dataset, load_from_disk
from openprompt.data_utils import InputExample
from transformers import AdamW
import torch
import matplotlib.pyplot as plt
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.utils.reproduciblity import set_seed

# model_path = 'E://models//gpt2-chinese-cluecorpussmall'
model_path = 'E://models//chinese-roberta-wwm-ext'
batch_size = 4
EPOCH = 20
max_length = 256
device = 'cuda'
shot = 16
learning_rate = 1e-6


# plm, tokenizer, model_config, wrapper_class = load_plm("bert", 'bert-base-chinese') # huggingface 仓库

#定义数据集
train_file = 'E:\\datasets\\csldcp\\train_few_all.json'
# val_file = 'E:\datasets\\tnews\\dev.json'
val_file = 'E:\datasets\\csldcp\\dev_few_all.json'
test_file = 'E:\datasets\\csldcp\\test_public.json'

label_to_index = {'材料科学与工程': 0, '作物学': 1, '口腔医学': 2, '药学': 3, '教育学': 4, '水利工程': 5, '理论经济学': 6, '食品科学与工程': 7, '畜牧学/兽医学': 8, '体育学': 9, '核科学与技术': 10, '力学': 11, '园艺学': 12, '水产': 13, '法学': 14, '地质学/地质资源与地质工程': 15, '石油与天然气工程': 16, '农林经济管理': 17, '信息与通信工程': 18, '图书馆、情报与档案管理': 19, '政治学': 20, '电气工程': 21, '海洋科学': 22, '民族学': 23, '航空宇航科学与技术': 24, '化学/化学工程与技术': 25, '哲学': 26, '公共卫生与预防医学': 27, '艺术学': 28, '农业工程': 29, '船舶与海洋工程': 30, '计算机科学与技术': 31, '冶金工程': 32, '交通运输工程': 33, '动力工程及工程热物理': 34, '纺织科学与工程': 35, '建筑学': 36, '环境科学与工程': 37, '公共管理': 38, '数学': 39, '物理学': 40, '林学/林业工程': 41, '心理学': 42, '历史学': 43, '工商管理': 44, '应用经济学': 45, '中医学/中药学': 46, '天文学': 47, '机械工程': 48, '土木工程': 49, '光学工程': 50, '地理学': 51, '农业资源利用': 52, '生物学/生物科学与工程': 53, '兵器科学与技术': 54, '矿业工程': 55, '大气科学': 56, '基础医学/临床医学': 57, '电子科学与技术': 58, '测绘科学与技术': 59, '控制科学与工程': 60, '军事学': 61, '中国语言文学': 62, '新闻传播学': 63, '社会学': 64, '地球物理学': 65, '植物保护': 66}


my_data_set = {}
def get_data_set(file, split):
    my_data_set[split] = []
    with open(file, encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = json.loads(line)
            input = InputExample(guid=idx, label=int(label_to_index[line['label']]), text_a=line['content'])
            my_data_set[split].append(input)
get_data_set(train_file, 'train')
get_data_set(val_file, 'validation')
get_data_set(test_file, 'test')

print('train dataset length: ', len(my_data_set['train']))
# print(my_data_set['train'])
print('validation dataset length: ', len(my_data_set['validation']))
# print('test dataset length: ', len(my_data_set['test']))
# 定义 PLM
plm, tokenizer, model_config, wrapper_class = load_plm("bert", model_path)  # 本地路径
sampler = FewShotSampler(num_examples_per_label=shot, also_sample_dev=True, num_examples_per_label_dev=shot)
train_dataset, valid_dataset = sampler(my_data_set['train'])
# print(train_dataset)
# template

# freq > 3, and top40(下同): 0.5328358208955224
prompt_template = ManualTemplate(text='{"placeholder":"text_a" }。好好思考一下，这包括{"mask"}。', tokenizer=tokenizer) # template1



train_dataloader = PromptDataLoader(dataset=train_dataset, template=prompt_template, tokenizer=tokenizer,
    tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length)
valid_dataloader = PromptDataLoader(dataset=valid_dataset, template=prompt_template, tokenizer=tokenizer,
    tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length, decoder_max_length=3,
    batch_size=batch_size,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")
test_dataloader = PromptDataLoader(dataset=my_data_set['test'], template=prompt_template, tokenizer=tokenizer,
    tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length, decoder_max_length=30,
    batch_size=batch_size, shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

classes = ['材料科学与工程', '作物学', '口腔医学', '药学', '教育学', '水利工程', '理论经济学', '食品科学与工程', '畜牧学/兽医学', '体育学', '核科学与技术', '力学', '园艺学', '水产', '法学', '地质学/地质资源与地质工程', '石油与天然气工程', '农林经济管理', '信息与通信工程', '图书馆、情报与档案管理', '政治学', '电气工程', '海洋科学', '民族学', '航空宇航科学与技术', '化学/化学工程与技术', '哲学', '公共卫生与预防医学', '艺术学', '农业工程', '船舶与海洋工程', '计算机科学与技术', '冶金工程', '交通运输工程', '动力工程及工程热物理', '纺织科学与工程', '建筑学', '环境科学与工程', '公共管理', '数学', '物理学', '林学/林业工程', '心理学', '历史学', '工商管理', '应用经济学', '中医学/中药学', '天文学', '机械工程', '土木工程', '光学工程', '地理学', '农业资源利用', '生物学/生物科学与工程', '兵器科学与技术', '矿业工程', '大气科学', '基础医学/临床医学', '电子科学与技术', '测绘科学与技术', '控制科学与工程', '军事学', '中国语言文学', '新闻传播学', '社会学', '地球物理学', '植物保护']
label_words = ['材料科学与工程', '作物学', '口腔医学', '药学', '教育学', '水利工程', '理论经济学', '食品科学与工程', '畜牧学/兽医学', '体育学', '核科学与技术', '力学', '园艺学', '水产', '法学', '地质学/地质资源与地质工程', '石油与天然气工程', '农林经济管理', '信息与通信工程', '图书馆、情报与档案管理', '政治学', '电气工程', '海洋科学', '民族学', '航空宇航科学与技术', '化学/化学工程与技术', '哲学', '公共卫生与预防医学', '艺术学', '农业工程', '船舶与海洋工程', '计算机科学与技术', '冶金工程', '交通运输工程', '动力工程及工程热物理', '纺织科学与工程', '建筑学', '环境科学与工程', '公共管理', '数学', '物理学', '林学/林业工程', '心理学', '历史学', '工商管理', '应用经济学', '中医学/中药学', '天文学', '机械工程', '土木工程', '光学工程', '地理学', '农业资源利用', '生物学/生物科学与工程', '兵器科学与技术', '矿业工程', '大气科学', '基础医学/临床医学', '电子科学与技术', '测绘科学与技术', '控制科学与工程', '军事学', '中国语言文学', '新闻传播学', '社会学', '地球物理学', '植物保护']
#
prompt_verbalizer = ManualVerbalizer(classes=classes, tokenizer=tokenizer, label_words=label_words)
prompt_model = PromptForClassification(plm=plm,template=prompt_template, verbalizer=prompt_verbalizer, freeze_plm=False).to(device)

#################################################################################################
# calibrate
from openprompt.utils.calibrate import calibrate
# support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
support_sampler = FewShotSampler(num_examples_per_label=10, also_sample_dev=False)
my_data_set['support'] = support_sampler(my_data_set['train'], seed=1)
for example in my_data_set['support']:
    example.label = -1 # remove the labels of support set for classification
support_dataloader = PromptDataLoader(dataset=my_data_set["support"], template=prompt_template, tokenizer=tokenizer,
    tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length,
    batch_size=4,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")
cc_logits = calibrate(prompt_model, support_dataloader)
prompt_model.verbalizer.register_calibrate_logits(cc_logits)

#################################################################################################


val_accs = []
def evaluate(data_loader, type='validation'):
    all_preds = []
    all_labels = []
    for step, inputs in enumerate(data_loader):
        inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        # print('-' * 100)
        # print('labels is : ', all_labels)
        # print('prediction is : ',all_preds)
    acc = sum([int(i == j) for i, j in zip(all_preds, all_labels)]) / len(all_preds)
    if type == 'validation':
        val_accs.append(acc)
    print(type, ' accuracy: ', acc)
    return acc

# evaluate(valid_dataloader)
test_acc = evaluate(test_dataloader, 'test')
