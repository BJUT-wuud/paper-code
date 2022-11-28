from openprompt.data_utils.text_classification_dataset import AgnewsProcessor, YahooProcessor, DBpediaProcessor
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt import PromptDataLoader
from openprompt.prompts import SoftVerbalizer
from openprompt.data_utils import InputExample
import torch
from openprompt import PromptForClassification
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaTokenizer
import os
import random
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm, trange
import json
import utils as my_util


model_path = 'E://models//chinese-roberta-wwm-ext'
batch_size = 4
device = 'cuda'


# tnews效果0.508
def main():

    output_dir = 'output'
    plm, tokenizer, model_config, WrapperClass = load_plm("bert", model_path)

    prompt_template = ManualTemplate(text='{"placeholder":"text_a" }。好好思考一下，这包括{"mask"}。', tokenizer=tokenizer) # template1


    # dataset, max_length, num_classes = my_util.get_tnews_data()
    # dataset, max_length, num_classes = my_util.get_cnews_data()
    dataset, max_length, num_classes = my_util.get_csldcp_data()

    verbalizer = SoftVerbalizer(tokenizer, plm, num_classes=num_classes)

    model = PromptForClassification(plm=plm, template=prompt_template, verbalizer=verbalizer, freeze_plm=False)
    model = model.cuda()

    checkpoint = os.path.join(output_dir, 'last_checkpoint')
    # tokenizer = RobertaTokenizer.from_pretrained(checkpoint)
    state_dict = torch.load(os.path.join(checkpoint, "model"))
    model.load_state_dict(state_dict)

    model.eval()

    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=prompt_template, tokenizer=tokenizer,
                                       tokenizer_wrapper_class=WrapperClass, max_seq_length=max_length,
                                       decoder_max_length=3,
                                       batch_size=batch_size, shuffle=False, teacher_forcing=False,
                                       predict_eos_token=False,
                                       truncate_method="tail")
    allpreds = []
    alllabels = []
    for inputs in tqdm(test_dataloader, desc="Evaluating"):
        inputs = inputs.cuda()
        with torch.no_grad():
            logits = model(inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    allpreds = np.array(allpreds)
    alllabels = np.array(alllabels)
    acc = sum([int(i == j) for i, j in zip(allpreds, alllabels)]) / len(allpreds)
    print(acc)
    # micro_f1, macro_f1, f1 = get_acc_f1(alllabels, allpreds, num_classes)
    # results = {}
    # results['eval_loss'] = 0
    # results['micro_f1'] = micro_f1
    # results['macro_f1'] = macro_f1
    # print("***** Eval results  *****")
    # result_str = "Eval loss is {}\nMicro f1 is {}\nMacro f1 is {}\n".format(0, micro_f1, macro_f1)
    # for i in range(num_classes):
    #     result_str += "f1 score of Class {} is: {}\n".format(i, f1[i])
    # result_str += "\n\n"
    # print(result_str)

    # output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    # with open(output_eval_file, "a") as f:
    #     f.write('***** Predict Result for {} sample_num {} seed {} prompt index {} *****\n'.format(args.task_name,
    #                                                                                                args.sample_num,
    #                                                                                                args.seed,
    #                                                                                                args.prompt_index))
    #     f.write(result_str)


if __name__ == "__main__":
    main()