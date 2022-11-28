from logging import exception
from openprompt.data_utils.text_classification_dataset import AgnewsProcessor, YahooProcessor, DBpediaProcessor
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt import PromptDataLoader
from openprompt.prompts import SoftVerbalizer
from openprompt.data_utils import InputExample
import json
import torch
from openprompt import PromptForClassification
from transformers import AdamW, get_linear_schedule_with_warmup
import os
import random
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm, trange
import utils as my_util
from openprompt.data_utils.data_sampler import FewShotSampler
from openprompt.utils.reproduciblity import set_seed

model_path = 'E://models//chinese-roberta-wwm-ext'
batch_size = 4
EPOCH = 20
device = 'cuda'
shot = 1
# tnews
# seed = 2, 1: 0.24875621890547264, 4: 0.39502487562189054, 8: 0.4955223880597015, 16: 0.5194029850746269
# seed = 144, 1: 0.17761194029850746, 4: 0.4154228855721393, 8: 0.43134328358208956, 16: 0.5154228855721393
# seed = 145, 1: 0.2064676616915423, 4: 0.2930348258706468 , 8: 0.43283582089552236, 16: 0.5203980099502488
# cnews
# seed = 2, 1: 0.5028, 4: 0.7553 , 8: 0.8976, 16: 0.9422
# seed = 145, 1: 0.5445, 4: 0.7955 , 8: 0.9377, 16: 0.9423
# seed = 144, 1: 0.5695, 4: 0.8837 , 8: 9269, 16: 0.9397
# csldcp
# seed = 2, 1: 0.1709641255605381, 4: 0.38621076233183854 , 8: 0.5179372197309418, 16: 0.5246636771300448
# seed = 144, 1: 0.2040358744394619, 4: 0.4282511210762332 , 8: 0.5162556053811659, 16: 0.5392376681614349
# seed = 145, 1: 0.1446188340807175, 4: 0.413677130044843 , 8: 0.5263452914798207, 16: 0.5257847533632287
seed = 145

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(seed)
    output_dir = 'output'
    # set_seed(args)

    plm, tokenizer, model_config, WrapperClass = load_plm("bert", model_path)

    prompt_template = ManualTemplate(text='{"placeholder":"text_a" }。好好思考一下，这包括{"mask"}。', tokenizer=tokenizer) # template1


    dataset = {}
    dataset,max_length, num_classes = my_util.get_csldcp_data()
    # dataset,max_length, num_classes = my_util.get_tnews_data()
    # dataset,max_length, num_classes = my_util.get_cnews_data()
    # dataset['test'] = data_processor.get_examples("./dataset/agnews/", "test_sample_100_1")

    sampler = FewShotSampler(num_examples_per_label=shot, also_sample_dev=True, num_examples_per_label_dev=shot)
    train_dataset, valid_dataset = sampler(dataset['train'])
    train_dataloader = PromptDataLoader(dataset=train_dataset, template=prompt_template, tokenizer=tokenizer,
                                        tokenizer_wrapper_class=WrapperClass, max_seq_length=max_length,
                                        decoder_max_length=3,
                                        batch_size=batch_size, shuffle=True, teacher_forcing=False,
                                        predict_eos_token=False,
                                        truncate_method="tail")
    print(len(dataset['train']))
    print(len(train_dataloader))
    verbalizer = SoftVerbalizer(tokenizer, plm, num_classes=num_classes)

    model = PromptForClassification(plm=plm, template=prompt_template, verbalizer=verbalizer, freeze_plm=False)
    model = model.cuda()

    # training
    loss_func = torch.nn.CrossEntropyLoss()

    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 5e-5, "lr": 3e-5},
        {'params': [p for n, p in model.plm.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 5e-5, "lr": 3e-5}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer_grouped_parameters2 = [
        {'params': model.verbalizer.group_parameters_1, 'weight_decay': 5e-5, "lr": 3e-5},
        {'params': model.verbalizer.group_parameters_2, 'weight_decay': 5e-5, "lr": 3e-5},
    ]

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5, eps=1e-8)
    optimizer2 = AdamW(optimizer_grouped_parameters2, lr=3e-5, eps=1e-8)

    global_step = 0
    epochs_trained = 0
    best_score = 0.0
    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(EPOCH), desc="Epoch"
    )
    # set_seed(args)
    epoch_num = 1
    for _ in train_iterator:
        print('-' * 50, epoch_num, '-' * 50)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, inputs in enumerate(epoch_iterator):
            model.train()
            inputs = inputs.cuda()
            logits = model(inputs)
            labels = inputs['label']
            loss = loss_func(logits, labels)
            try:
                loss.backward()
            except RuntimeError:
                print(loss)
            tr_loss += loss.item()
            epoch_iterator.set_description('Loss: {}'.format(round(loss.item(), 6)))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer1.step()
            optimizer1.zero_grad()
            optimizer2.step()
            optimizer2.zero_grad()
            model.zero_grad()
            global_step += 1
        epoch_num += 1

    output_dir = os.path.join(output_dir, "last_checkpoint")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    torch.save(model.state_dict(), os.path.join(output_dir, "model"))
    print("saving model to {}".format(output_dir))


if __name__ == "__main__":
    main()
