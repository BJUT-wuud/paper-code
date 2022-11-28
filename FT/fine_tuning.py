#encoding=utf-8
import torch
from datasets import load_from_disk, load_dataset
from transformers import BertTokenizer
from transformers import BertModel
from transformers import AdamW
from openprompt.utils.reproduciblity import set_seed
import util as my_util

epoch = 20
batch_size = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'E://models//chinese-roberta-wwm-ext'
model_name = 'bert'
learn_rate = 1e-3
shot = 16
seed = 2
set_seed(seed)

# tnews
# seed = 2, 1: 0.23057768924302788, 4: 0.4437250996015936, 8: 0.4900398406374502, 16:  0.50199203187251
# seed = 144, 1:  0.26095617529880477, 4:0.3610557768924303, 8: 0.424800796812749, 16: 0.4910358565737052
# seed = 145, 1: 0.2793824701195219, 4: 0.44721115537848605, 8: 0.4945219123505976, 16: 0.5189243027888446
# cnews
# seed = 2, 1:  0.5428, 4: 0.794, 8:0.9008, 16:0.9082
# seed = 143, 1: 0.4165, 4: 0.7681, 8:0.8245, 16:0.9303
# seed = 144, 1: 0.6787, 4: 0.891, 8:0.9154, 16:0.924
# csldcp
# seed=2, 1: 0.25, 4: 0.42096412556053814, 8:0.5011210762331838, 16: 0.531390134529148
# seed=144, 1: 0.2875560538116592, 4:  0.43385650224215244, 8:0.4646860986547085, 16: 0.5179372197309418
# seed=145, 1: 0.24663677130044842, 4: 0.4316143497757848, 8:0.4899103139013453, 16: 0.5358744394618834

# csldcp
# max_length = 256
# num_classes = 67
# my_dataset = my_util.get_csldcp_data()
# train_file, val_file = my_util.few_shot_sample(seed, shot, my_dataset, 'csldcp')
# test_file = my_util.trans_test_dataset(my_dataset, 'csldcp')
# Dataset = my_util.CsldcpDataset
# cnews
# max_length = 512
# num_classes = 10
# my_dataset = my_util.get_cnews_data()
# train_file, val_file = my_util.few_shot_sample(seed, shot, my_dataset, 'cnews')
# test_file = my_util.trans_test_dataset(my_dataset, 'cnews')
# Dataset = my_util.CnewsDataset
# # tnews
max_length = 50
num_classes = 15
my_dataset = my_util.get_tnew_data()
train_file, val_file = my_util.few_shot_sample(seed, shot, my_dataset, 'tnews')
test_file = my_util.trans_test_dataset(my_dataset, 'tnews')
Dataset = my_util.TnewsDataset


dataset = Dataset(train_file, val_file, test_file, 'train')
print(dataset)
print(len(dataset))


token = BertTokenizer.from_pretrained(model_path)
# token = BertTokenizer.from_pretrained(model_name) # huggingface 仓库

def collate_fn(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]


    data = token.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=max_length,
                                   return_tensors='pt',
                                   return_length=True)

    #input_ids:
    #attention_mask:
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)

    #print(data['length'], data['length'].max())

    return input_ids, attention_mask, token_type_ids, labels



loader = torch.utils.data.DataLoader(dataset=dataset,
                                     batch_size=batch_size,
                                     collate_fn=collate_fn,
                                     shuffle=True,
                                     drop_last=True)




pretrained = BertModel.from_pretrained(model_path).to(device)
# pretrained = BertModel.from_pretrained(model_name).to(device)

#
# for param in pretrained.parameters():
#     param.requires_grad_(False)

#
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):

        out = pretrained(input_ids=input_ids,
                   attention_mask=attention_mask,
                   token_type_ids=token_type_ids)

        out = self.fc(out.last_hidden_state[:, 0])
#         print("1: ", out)
        out = out.softmax(dim=1)
#         print("2: ", out)

        return out


model = Model().to(device)



#
optimizer = AdamW(model.parameters(), lr=learn_rate)
criterion = torch.nn.CrossEntropyLoss()

model.train()

print('--------------start training-----------------')
best_val_acc = 0
for poch in range(epoch):
    tot_loss = 0
    idx = 0
    correct = 0
    total = 0
    for i, (input_ids, attention_mask, token_type_ids,
        labels) in enumerate(loader):
        input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
    #     print(labels.shape)
    #     print(out.shape)
        loss = criterion(out, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer.step()
        optimizer.zero_grad()
        out = out.argmax(dim=1)
        idx = i
        correct += (out == labels).sum().item()
        total += len(labels)
    val_acc = correct / total
    # if val_acc >= best_val_acc:
    #     torch.save(model.state_dict(), "./best_val.ckpt")
    #     best_val_acc = val_acc
    print("Epoch {}, average loss: {}, valid acc: {}".format(poch, tot_loss / (idx + 1), val_acc), flush=True)
    # print('correct : total: ', correct, total)

def test():
    # best_model = model.load_state_dict(torch.load("./best_val.ckpt"))
    # print(type(torch.load("./best_val.ckpt")))
    # best_model = best_model.to(device)
    model.eval()
    correct = 0
    total = 0

    #
    loader_test = torch.utils.data.DataLoader(dataset=Dataset(train_file, val_file, test_file, 'test'),
                                              batch_size=batch_size,
                                              collate_fn=collate_fn,
                                              shuffle=True,
                                              drop_last=True)

    print('test dataset length : ',len(loader_test))
    for i, (input_ids, attention_mask, token_type_ids,
            labels) in enumerate(loader_test):
        input_ids, attention_mask, token_type_ids, labels = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device), labels.to(device)
        # if i == 50:
        #     break

#         print(i)

        with torch.no_grad():
            out = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids)

        out = out.argmax(dim=1)
        correct += (out == labels).sum().item()
        total += len(labels)

    print('test accuracy : ', correct / total)


test()