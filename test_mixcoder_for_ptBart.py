import datasets
# from transformers import BartModel, BartConfig, BartForConditionalGeneration, BertModel
from transformers import AdamW, get_scheduler
import torch
from datasets import load_dataset
import custom_datasets
import custom_tokenizer
# from modeling_mixcoder import MixCoderForConditionalGeneration, MixCoderConfig
# from modeling_mc import BartForConditionalGeneration, BartConfig
import evaluate

from tqdm import tqdm
# from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
import os
import argparse
import numpy as np

import json
import wandb

#set seed function
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

argparser = argparse.ArgumentParser()
argparser.add_argument("--gpu", type=int, default=0)
argparser.add_argument("--save_path", type=str, default="")

tmp_args = argparser.parse_args()

args = argparse.Namespace(**json.load(open(os.path.join(tmp_args.save_path+"/..", "args.json"), "r", encoding="utf8")))
args.gpu = tmp_args.gpu
args.save_path = tmp_args.save_path
set_seed(args.seed)

data_name = args.data_name
subset = args.subset
batch_size = args.batch_size
tokenizer_path = args.tokenizer_path
gpu = args.gpu
device = "cuda:"+str(gpu)
learning_rate = args.learning_rate
epoch = args.epoch
full_step = args.full_step
eval_step = args.eval_step
next_token_type = args.next_token_type
share_self_attention_module = args.share_self_attention_module
pass_hidden_to_cross_att = args.pass_hidden_to_cross_att
max_norm = args.max_norm
share_cross_attention_module = args.share_cross_attention_module
indi_self_query = args.indi_self_q
indi_self_output = args.indi_self_out
indi_cross_query = args.indi_cross_q
indi_cross_output = args.indi_cross_out
share_ffnn = args.share_ffnn
save_path = args.save_path

# if args.baseline:
#     save_path = "baseline-" + args.save_path
# elif args.pre_trained_baseline:
#     save_path = "pre_trained_baseline-" + args.save_path
# else:
#     save_path = args.save_path
#     save_path += "-" + next_token_type 
#     if share_self_attention_module:
#         save_path += "-share_att"
#     if indi_self_query:
#         save_path += "-indi_self_q"
#     if indi_self_output:
#         save_path += "-indi_self_out"
#     if share_cross_attention_module:
#         save_path += "-share_cross_att"
#     if indi_cross_query:
#         save_path += "-indi_cross_q"
#     if indi_cross_output:
#         save_path += "-indi_cross_out"
#     if pass_hidden_to_cross_att:
#         save_path += "-hidden_cross_att"
#     if share_ffnn:
#         save_path += "-share_ffnn"

# save_path = os.path.join("results",f"{args.data_name}_{args.src_lang}-{args.tgt_lang}", save_path)

# # if os.path.exists(save_path):
# #     input("this path already exists. press enter to continue.")

# os.makedirs(save_path, exist_ok=True)
# json.dump(vars(args), open(os.path.join(save_path, "args.json"), "w", encoding="utf8"), indent=2)

# wandb.init(project=f"MixCoder_{args.data_name}_{args.subset}", name=save_path, config=vars(args))

# data_name = "wmt14"
# subset = "de-en"
# batch_size = 16
# tokenizer_path = "tokenizer/wmt14_de-en_BPEtokenizer.json"
# gpu = 1
# device = "cuda:"+str(gpu)
# learning_rate = 5e-5
# epoch = 10
# full_step = 1000000
# eval_step = 10000
# # next_token_type = "new_token"
# next_token_type = "avg_prev_token"
# share_self_attention_module = True
# pass_hidden_to_cross_att = False

# wmt 14 train bart model
dataset = load_dataset(data_name, subset)
print("before filtering:")
print(dataset)

dataset["test"] = dataset["test"].filter(lambda x: len(x["translation"][args.src_lang]) < 1024 and len(x["translation"][args.tgt_lang]) < 1024)
print("after filtering:")
print(dataset)

pre_train_path = "facebook/bart-base"

if args.baseline:
    from transformers import BartTokenizer, BartForConditionalGeneration
    tokenizer = BartTokenizer.from_pretrained(pre_train_path)
    model = BartForConditionalGeneration.from_pretrained(save_path, local_files_only=True)
    model.to(device)

else:
    from transformers import BartTokenizer
    from modeling_mc_with_pre_trained_bart import BartForConditionalGeneration, BartConfig
    tokenizer = BartTokenizer.from_pretrained(pre_train_path)
    len_tokenizer = len(tokenizer)
    
    if next_token_type == "new_token":
        tokenizer.add_tokens("<next>", special_tokens=True)
        next_token_id = tokenizer.convert_tokens_to_ids("<next>")
    else:
        next_token_id = None
    print(len(tokenizer))

    print("load_pre trained model")
    model = BartForConditionalGeneration.from_pretrained(save_path, local_files_only=True)

    print(model.config)

    # if args.copy_qo:
    #     print("copy qo")
    #     model.model.deepcopy_indi_qo()

    if next_token_type == "new_token":
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)



print(model)

# train_dataset = custom_datasets.WmtDataset(dataset["train"], tokenizer=tokenizer, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
# val_dataset = custom_datasets.WmtDataset(dataset["validation"], tokenizer=tokenizer, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
test_dataset = custom_datasets.WmtDataset(dataset["test"], tokenizer=tokenizer, src_lang=args.src_lang, tgt_lang=args.tgt_lang)

# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, num_workers=4, shuffle=True, drop_last=True)
# val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, collate_fn=val_dataset.collate_fn, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=test_dataset.collate_fn)

cur_step = 0

model.eval()

matric_scarebleu = evaluate.load("sacrebleu")
matric_bleu = evaluate.load("bleu")
matric_scarebleu_v14 = evaluate.load("sacrebleu")
matric_bleu_v14 = evaluate.load("bleu")
with torch.no_grad():
    refers = []
    preds = []
    for batch in tqdm(test_dataloader):
        for i in batch.keys():
            batch[i] = batch[i].to(device)

        out = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False, num_beams=5, do_sample=True, max_new_tokens=512)
        print(out)
        pred_str = tokenizer.batch_decode(out, skip_special_tokens=True)
        print(pred_str)
        # out = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=True)
        # print(out)
        # # out = model(**batch)
        # # pred = out.logits.argmax(dim=-1)
        # pred_str = tokenizer.batch_decode(out, skip_special_tokens=True)
        # print(pred_str)

        refer = tokenizer.batch_decode(torch.where(batch["labels"] == -100, tokenizer.pad_token_id, batch["labels"]), skip_special_tokens=True)
        refers.extend(refer)
        preds.extend(pred_str)
        print(refer, "\n\n\n")

        matric_scarebleu.add_batch(predictions=pred_str, references=refer)
        matric_bleu.add_batch(predictions=pred_str, references=refer)
        matric_scarebleu_v14.add_batch(predictions=pred_str, references=refer)
        matric_bleu_v14.add_batch(predictions=pred_str, references=refer)
        # print(pred_str)

    # matric.add_batch(predictions=preds, references=refers)
    # matric_result=matric_scarebleu.compute(predictions=preds, references=refers)
    matric_scarebleu_result = matric_scarebleu.compute()
    print(matric_scarebleu_result)
    matric_bleu_result = matric_bleu.compute()
    print(matric_bleu_result)
    matric_scarebleu_v14_result = matric_scarebleu_v14.compute(tokenize="intl")
    print(matric_scarebleu_v14_result)

        
