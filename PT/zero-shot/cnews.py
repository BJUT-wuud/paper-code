# coding=utf-8

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

model_path = 'E://models//chinese-roberta-wwm-ext'
batch_size = 4
EPOCH = 20
max_length = 512
device = 'cuda'
shot = 16
learning_rate = 1e-6


# plm, tokenizer, model_config, wrapper_class = load_plm("bert", 'bert-base-chinese') # huggingface 仓库

#定义数据集
train_file = 'E:\datasets\cnews\\cnews.train.txt'
val_file = 'E:\datasets\\cnews\\cnews.val.txt'
test_file = 'E:\datasets\\cnews\\cnews.test.txt'
map = {'体育':0, '财经':1, '房产':2, '家居':3, '教育':4, '科技':5, '时尚':6, '时政':7, '游戏':8, '娱乐':9}
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
# get_data_set(val_file, 'validation')
get_data_set(test_file, 'test')

print('train dataset length: ', len(my_data_set['train']))
# print('validation dataset length: ', len(my_data_set['validation']))
print('test dataset length: ', len(my_data_set['test']))


# 定义 PLM
plm, tokenizer, model_config, wrapper_class = load_plm("bert", model_path) # 本地路径
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
    tokenizer_wrapper_class=wrapper_class, max_seq_length=max_length, decoder_max_length=3,
    batch_size=batch_size,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

classes = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
label_words = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

prompt_verbalizer = ManualVerbalizer(classes=classes, tokenizer=tokenizer, label_words=label_words)
prompt_model = PromptForClassification(plm=plm,template=prompt_template, verbalizer=prompt_verbalizer, freeze_plm=False).to(device)

#################################################################################################
from openprompt.utils.calibrate import calibrate
# support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
# 500:0.723, 200:0.7234,
support_sampler = FewShotSampler(num_examples_per_label=200, also_sample_dev=False)
my_data_set['support'] = support_sampler(my_data_set['train'], seed=1)
print('support length: ', len(my_data_set['support']))
for example in my_data_set['support']:
    example.label = -1 # remove the label s of support set for classification
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
