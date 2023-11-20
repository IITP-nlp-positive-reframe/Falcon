# falcon evaluate

import argparse
import pandas as pd
import numpy as np
import torch
import random
# from sentence_transformers import SentenceTransformer, util
from transformers import DataCollatorForLanguageModeling
import os
from transformers import Trainer, pipeline, set_seed, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainingArguments
import nltk
nltk.download('punkt')
import csv
import transformers
from datasets import load_dataset, load_metric
from peft import LoraConfig,PeftConfig,get_peft_model,prepare_model_for_kbit_training
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

from datasets import load_dataset
root = "./"
train_path = root + "data/wholetrain_gpt.txt"
# train_path = root + "data/wholetrain.csv"
# train_dataset = load_dataset('csv', data_files=train_path)
dev_path =  root+"data/wholedev.csv"
dev_dataset = load_dataset('csv', data_files=dev_path)
test_path =  root + "data/wholetest.csv"
test_dataset = load_dataset('csv', data_files=test_path)

output_dir = root+"falcon-rw-1b"
f_name = "falcon_rw_1b_predict.txt"

model = AutoModelForCausalLM.from_pretrained(output_dir+"/output/reframer", load_in_4bit=True)
config = PeftConfig.from_pretrained(output_dir+"/output/reframer")
model = prepare_model_for_kbit_training(model)
tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path)
tokenizer.bos_token = "<startoftext>"
tokenizer.eos_token = "<endoftext>"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
# num_return_sequences 추가해서 돌려보기
reframer = pipeline('text-generation', model=model, tokenizer=tokenizer, stop_sequence="<endoftext>",eos_token_id=tokenizer.eos_token_id,  num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

bleu = load_metric('sacrebleu')
rouge = load_metric("rouge")

preds = []
gts = []

import csv
with open (test_path, newline="") as data:
    annotations = csv.DictReader(data, delimiter=',', quotechar='"')
    annotations_list = list(annotations)
    reframed_phrases = []
    answer_phrases = []
    for i in range(len(annotations_list)):
        prefix = "<startoftext> " + annotations_list[i]['original_text'] + "\nreframed:"
        gen_text = reframer(prefix, max_length=100)[0]['generated_text']
        if "<endoftext>" in gen_text:
            eos_idx = gen_text.index("<endoftext>")
            gen_text = gen_text[:eos_idx-1]
        sdx = gen_text.index("reframed:")
        pred_text = gen_text[sdx+10:]    
        print("-----------------------------------------------------------------------")
        print(pred_text)
        print("-----------------------------------------------------------------------")
        print(annotations_list[i]['reframed_text'])
        print("-----------------------------------------------------------------------")
        # reframed_phrases.append(gen_text+"\n---------------")
        gts.append([annotations_list[i]['reframed_text']])
        preds.append([pred_text])
        # answer_phrases.append(annotations_list[i]['original_text'] + "\nreframed:"+ annotations_list[i]['reframed_text']+"\n-------------")

bleu_scores = bleu.compute(predictions=preds, references =gts)['score']
rouge_scores = rouge.compute(predictions=preds, references=gts,use_stemmer=True)
for key, value in rouge_scores.items():
    print(key, value.mid.fmeasure*100)
# result = {key: value.mid.fmeasure * 100 for key, value in rouge_scores.items()}
print("load_metric('sacrebleu')", bleu_scores)

# with open(os.path.join(root, f_name), 'w') as f:
#     for item in reframed_phrases:
#         f.write("%s\n" % item)
# print("write complete!")
# with open(os.path.join(root, "total_reframe.txt"),'w') as f:
#     for item in answer_phrases:
#         f.write("%s\n"%item)
# print("answer written")

