# coding=utf-8
# freeze plm

# !pip install transformers --quiet
# !pip install datasets==2.0 --quiet
# !pip install openprompt --quiet
# !pip install torch --quiet

# function ConnectButton(){
#     console.log("Connect pushed");
#     document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click()
# }
# setInterval(ConnectButton,60000);
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

# model_path = 'E://models//wobert_chinese_base'
model_path = 'E://models//chinese-roberta-wwm-ext'
batch_size = 4
EPOCH = 20
max_length = 50
device = 'cuda'
shot = 16
learning_rate = 1e-6

'''
应考虑，soft token个数对学习率的影响
1. 对不同soft token 进行对比
1.1 soft token的数量
1.2 soft token + hard prompt
2. 对不同 verbalizer 进行对比
2.1 单个verbalizer 和 每个标签多个 verbalizer 进行对比


zero-shot few-shot 对比
'''


# plm, tokenizer, model_config, wrapper_class = load_plm("bert", 'bert-base-chinese') # huggingface 仓库

# 定义数据集
train_file = 'E:\datasets\\few-tnews\\train_few_all.json'
# train_file = 'E:\datasets\\tnews\\train.json'
# val_file = 'E:\datasets\\tnews\\dev.json'
val_file = 'E:\datasets\\tnews\\dev_few_all.json'
test_file = 'E:\datasets\\few-tnews\\test_public.json'


def trans_label(label):
    map = {100: 0, 101: 1, 102: 2, 103: 3, 104: 4, 106: 5, 107: 6, 108: 7, 109: 8, 110: 9, 112: 10, 113: 11, 114: 12,
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
get_data_set(val_file, 'validation')
get_data_set(test_file, 'test')

print('train dataset length: ', len(my_data_set['train']))
# print(my_data_set['train'])
print('validation dataset length: ', len(my_data_set['validation']))

# 0.41691542288557215
seed = 144
set_seed(seed)
# 定义 PLM
plm, tokenizer, model_config, wrapper_class = load_plm("bert", model_path)  # 本地路径

# template
prompt_template = ManualTemplate(text='{"placeholder":"text_a" }。好好思考一下，这包括{"mask"}。', tokenizer=tokenizer)  # template1


test_dataloader = PromptDataLoader(dataset=my_data_set['test'], template=prompt_template, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length,
                                   decoder_max_length=3,
                                   batch_size=batch_size, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="head")

classes = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "证券", "农业", "游戏"]
label_words = ["故事", "文化", "娱乐", "体育", "财经", "房产", "汽车", "教育", "科技", "军事", "旅游", "国际", "证券", "农业", "游戏"]

# 0.5348258706467661
prompt_verbalizer = ManualVerbalizer(classes=classes, tokenizer=tokenizer, label_words=label_words)
prompt_model = PromptForClassification(plm=plm, template=prompt_template, verbalizer=prompt_verbalizer,
                                       freeze_plm=False).to(device)

#################################################################################################
# calibrate
from openprompt.utils.calibrate import calibrate

# support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
support_sampler = FewShotSampler(num_examples_per_label=100, also_sample_dev=False)
my_data_set['support'] = support_sampler(my_data_set['train'], seed=1)
print('support length: ', len(my_data_set['support']))
for example in my_data_set['support']:
    example.label = -1  # remove the labels of support set for classification
support_dataloader = PromptDataLoader(dataset=my_data_set["support"], template=prompt_template, tokenizer=tokenizer,
                                      tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length,
                                      batch_size=8, shuffle=False, teacher_forcing=False, predict_eos_token=False,
                                      truncate_method="tail")
cc_logits = calibrate(prompt_model, support_dataloader)
print('cc logits shape: ', cc_logits.shape)
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
        # print('labels is : ', labels.tolist())
        # print('prediction is : ',torch.argmax(logits, dim=-1).tolist())
        # print(logits.shape)
    acc = sum([int(i == j) for i, j in zip(all_preds, all_labels)]) / len(all_preds)
    if type == 'validation':
        val_accs.append(acc)
    print(type, ' accuracy: ', acc)
    return acc



# evaluate(valid_dataloader)
test_acc = evaluate(test_dataloader, 'test')
print(test_acc)
# with open(f'./tnews_res/seed{seed}_shot{shot}.txt', 'w') as f:
#     f.write('seed = {}, shot = {}, acc = {}'.format(seed, shot, test_acc))
