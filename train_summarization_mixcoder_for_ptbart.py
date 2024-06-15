import datasets
from transformers import AdamW, get_scheduler
import torch
from datasets import load_dataset
import custom_datasets
import custom_tokenizer
import evaluate

from tqdm import tqdm
import os
import argparse
import numpy as np

import json
import wandb

import pandas as pd

#set seed function
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

argparser = argparse.ArgumentParser()
argparser.add_argument("--next_token_type", type=str, default="avg_prev_token", choices=["new_token", "avg_prev_token"])
argparser.add_argument("--share_self_attention_module", default=False, action="store_true")
argparser.add_argument("--indi_self_q", default=False, action="store_true")
argparser.add_argument("--indi_self_out", default=False, action="store_true")
argparser.add_argument("--share_cross_attention_module", default=False, action="store_true")
argparser.add_argument("--indi_cross_q", default=False, action="store_true")
argparser.add_argument("--indi_cross_out", default=False, action="store_true")
argparser.add_argument("--pass_hidden_to_cross_att", default=False, action="store_true")
argparser.add_argument("--share_ffnn", default=False, action="store_true")
argparser.add_argument("--setting", type=str, default=None)
argparser.add_argument("--base", default=False, action="store_true")
argparser.add_argument("--copy_qo", default=False, action="store_true")
argparser.add_argument("--copy_f", default=False, action="store_true")

argparser.add_argument("--data_name", type=str, default="xsum")
argparser.add_argument("--batch_size", type=int, default=16)
argparser.add_argument("--gpu", type=int, default=0)
argparser.add_argument("--learning_rate", type=float, default=5e-5)
argparser.add_argument("--weight_decay", type=float, default=0.0)
argparser.add_argument("--epoch", type=int, default=10)
argparser.add_argument("--num_beam", type=int, default=5)
argparser.add_argument("--full_step", type=int, default=1000010)
argparser.add_argument("--eval_step", type=int, default=50000)
argparser.add_argument("--save_path", type=str, default="")
argparser.add_argument("--baseline", default=False, action="store_true")
argparser.add_argument("--pre_trained_baseline", default=False, action="store_true")
argparser.add_argument("--max_norm", type=float, default=1.0)
argparser.add_argument("--seed", type=int, default=42)
argparser.add_argument("--logging_step", type=int, default=1000)

args = argparser.parse_args()
set_seed(args.seed)

setting_dict = {
    "1":
        {
            "share_self_attention_module":True,
            "indi_self_q":False,
            "indi_self_out":False,
            "share_cross_attention_module":True,
            "indi_cross_q":False,   
            "indi_cross_out":False,
            "pass_hidden_to_cross_att":True,
            "share_ffnn":True
        },
    "2":
        {
            "share_self_attention_module":True,
            "indi_self_q":True,
            "indi_self_out":True,
            "share_cross_attention_module":True,
            "indi_cross_q":True,   
            "indi_cross_out":True,
            "pass_hidden_to_cross_att":True,
            "share_ffnn":False
        }
}

if args.setting is not None:
    args.share_self_attention_module = setting_dict[args.setting]["share_self_attention_module"]
    args.indi_self_q = setting_dict[args.setting]["indi_self_q"]
    args.indi_self_out = setting_dict[args.setting]["indi_self_out"]
    args.share_cross_attention_module = setting_dict[args.setting]["share_cross_attention_module"]
    args.indi_cross_q = setting_dict[args.setting]["indi_cross_q"]
    args.indi_cross_out = setting_dict[args.setting]["indi_cross_out"]
    args.pass_hidden_to_cross_att = setting_dict[args.setting]["pass_hidden_to_cross_att"]
    args.share_ffnn = setting_dict[args.setting]["share_ffnn"]

DATA_INFO = {
    "cnn_dailymail": {"data_name":"abisee/cnn_dailymail","subset":"3.0.0","src_lang":"article", "tgt_lang":"highlights"},
    "xsum": {"data_name":"EdinburghNLP/xsum","subset":None,"src_lang":"document", "tgt_lang":"summary"},
}

data_name = DATA_INFO[args.data_name]["data_name"]
subset = DATA_INFO[args.data_name]["subset"]
src_lang = DATA_INFO[args.data_name]["src_lang"]
tgt_lang = DATA_INFO[args.data_name]["tgt_lang"]
batch_size = args.batch_size
tokenizer_path = f"tokenizer/{args.data_name}_{subset}_BPEtokenizer.json"
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

if args.baseline:
    save_path = "baseline-" + args.save_path
elif args.pre_trained_baseline:
    save_path = "pre_trained_baseline-" + args.save_path
else:
    save_path = args.save_path
    save_path += "-" + next_token_type 
    if share_self_attention_module:
        save_path += "-share_att"
    if indi_self_query:
        save_path += "-indi_self_q"
    if indi_self_output:
        save_path += "-indi_self_out"
    if share_cross_attention_module:
        save_path += "-share_cross_att"
    if indi_cross_query:
        save_path += "-indi_cross_q"
    if indi_cross_output:
        save_path += "-indi_cross_out"
    if pass_hidden_to_cross_att:
        save_path += "-hidden_cross_att"
    if share_ffnn:
        save_path += "-share_ffnn"
    if args.copy_qo:
        save_path += "-copy_qo"
    if args.copy_f:
        save_path += "f"

if args.base:
    pre_train_path = "facebook/bart-base"
    n_layer=6
    d_model=768
    decoder_layers=6
    decoder_attention_heads=12
    decoder_ffn_dim=3072
    encoder_layers=6
    encoder_attention_heads=12
    encoder_ffn_dim=3072
    max_position_embeddings=1024

    save_path = os.path.join("results_ptBart_base",f"{args.data_name}_{src_lang}-{tgt_lang}", save_path)
    wandb.init(project=f"MixCoder_ptBart_base_{args.data_name}_{subset}_{src_lang}-{tgt_lang}", name=save_path, config=vars(args))

else:
    n_layer=6
    d_model=512
    decoder_layers=6
    decoder_attention_heads=8
    decoder_ffn_dim=2048
    encoder_layers=6
    encoder_attention_heads=8
    encoder_ffn_dim=2048
    max_position_embeddings=512

    save_path = os.path.join("results_ptBart",f"{args.data_name}_{src_lang}-{tgt_lang}", save_path)
    wandb.init(project=f"MixCoder_ptBart_{args.data_name}_{subset}_{src_lang}-{tgt_lang}", name=save_path, config=vars(args))
    
os.makedirs(save_path, exist_ok=True)
json.dump(vars(args), open(os.path.join(save_path, "args.json"), "w", encoding="utf8"), indent=2)

# wmt 14 train bart model
dataset = load_dataset(data_name, subset)
print(dataset)

if args.baseline:
    from transformers import BartTokenizer, BartForConditionalGeneration
    tokenizer = BartTokenizer.from_pretrained(pre_train_path)
    model = BartForConditionalGeneration.from_pretrained(pre_train_path)
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

    mixcoder_config = BartConfig(n_layer=n_layer,
                                    d_model=d_model,
                                    decoder_layers=decoder_layers,
                                    decoder_attention_heads=decoder_attention_heads,
                                    decoder_ffn_dim=decoder_ffn_dim,
                                    encoder_layers=encoder_layers,
                                    encoder_attention_heads=encoder_attention_heads,
                                    encoder_ffn_dim=encoder_ffn_dim,
                                    max_position_embeddings=max_position_embeddings,
                                    pad_token_id=tokenizer.pad_token_id, 
                                    eos_token_id=tokenizer.eos_token_id, 
                                    bos_token_id=tokenizer.bos_token_id, 
                                    decoder_start_token_id=tokenizer.eos_token_id, 
                                    is_encoder_decoder=True, 
                                    forced_bos_token_id=tokenizer.bos_token_id, 
                                    forced_eos_token_id=tokenizer.eos_token_id, 
                                    vocab_size=len_tokenizer, #pre trained model is not have <next> token
                                    next_token_type=next_token_type,
                                    next_token_id=next_token_id,
                                    share_self_attention_module=share_self_attention_module,
                                    pass_hidden_to_cross_att=pass_hidden_to_cross_att,
                                    share_cross_attention_module=share_cross_attention_module,
                                    indi_self_query=indi_self_query,
                                    indi_self_output=indi_self_output,
                                    indi_cross_query=indi_cross_query,
                                    indi_cross_output=indi_cross_output,
                                    share_ffnn=share_ffnn
                                    )
    print(mixcoder_config)
    model = BartForConditionalGeneration(config=mixcoder_config)
    print("load_pre trained model")
    model = model.from_pretrained(pre_train_path, config=mixcoder_config)

    print(model.config)

    if args.copy_qo:
        print("copy qo")
        model.model.deepcopy_indi_qo()

    if args.copy_f:
        print("copy f")
        model.model.deepcopy_f()

    if next_token_type == "new_token":
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)

print(model)

train_dataset = custom_datasets.SummarizationDataset(dataset["train"], tokenizer=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang)
val_dataset = custom_datasets.SummarizationDataset(dataset["validation"], tokenizer=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang)
test_dataset = custom_datasets.SummarizationDataset(dataset["test"], tokenizer=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=train_dataset.collate_fn, num_workers=4, shuffle=True, drop_last=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, collate_fn=val_dataset.collate_fn, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=test_dataset.collate_fn)

num_training = len(train_dataloader) * epoch
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
scheduler = get_scheduler("linear", optimizer, num_warmup_steps=100, num_training_steps=num_training)

cur_step = 0

refers = []
preds = []
model.train()
result_dict_rouge = {}
logging_losses = []
best_rouge = 0
best_step = 0
for E in range(epoch):
    print(f"Epoch {E}")

    td = tqdm(train_dataloader)
    for batch in td:
        for i in batch.keys():
            batch[i] = batch[i].to(device)
        
        out = model(**batch)
        out.loss.backward()
        logging_losses.append(out.loss.item())
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        td.set_postfix(loss=out.loss.item())

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        cur_step += 1

        if cur_step % args.logging_step == 0:
            wandb.log({"loss":np.mean(logging_losses), "_step":cur_step})
            logging_losses = []
            
        if cur_step > full_step:
            break

    model.eval()
    rouge = evaluate.load("rouge")
    with torch.no_grad():
        refers = []
        preds = []
        for batch in tqdm(val_dataloader):
            for i in batch.keys():
                batch[i] = batch[i].to(device)

            out = model(**batch)
            pred = out.logits.argmax(dim=-1)
            pred_str = tokenizer.batch_decode(pred, skip_special_tokens=True)

            refer = tokenizer.batch_decode(torch.where(batch["labels"] == -100, tokenizer.pad_token_id, batch["labels"]), skip_special_tokens=True)
            refers.extend(refer)
            preds.extend(pred_str)

            rouge.add_batch(predictions=pred_str, references=refer)

        matric_rouge_result = rouge.compute()
        result_dict_rouge[str(cur_step)] = matric_rouge_result
        print(matric_rouge_result)

        if matric_rouge_result["rouge2"] > best_rouge:
            best_rouge = matric_rouge_result["rouge2"]
            best_step = cur_step
        
        
        os.makedirs(os.path.join(save_path,str(cur_step)), exist_ok=True)
        model.save_pretrained(os.path.join(save_path,str(cur_step)), safe_serialization=False)

        result_str_dict = dict()
        for idx,(r,p) in enumerate(zip(refers, preds)):
            result_str_dict[str(idx)] = {"ref":r, "pred":p}

        json.dump(result_str_dict, open(os.path.join(save_path,str(cur_step),"validation_result.json"), "w", encoding="utf8"), indent=2)
        json.dump(result_dict_rouge, open(os.path.join(save_path,"result_scareBLEU.json"), "w", encoding="utf8"), indent=2)

        wandb.log({"loss":np.mean(logging_losses), "_step":cur_step, "ROUGE-1":matric_rouge_result["rouge1"], "ROUGE-2":matric_rouge_result["rouge2"], "ROUGE-L":matric_rouge_result["rougeL"], "ROUGE-Lsum":matric_rouge_result["rougeLsum"]})
        logging_losses = []
    model.train()
        


if os.path.exists(os.path.join("results_rouge.csv")):
    result_df = pd.read_csv("results_rouge.csv", index_col = 0)
else:
    result_df = pd.DataFrame(columns=["save_path", "rouge1", "rouge2", "rougeL", "rougeLsum"])  

model.from_pretrained(os.path.join(save_path,str(best_step)), local_files_only=True)
model.eval()

rouge = evaluate.load("rouge")
with torch.no_grad():
    refers = []
    preds = []
    for batch in tqdm(test_dataloader):
        for i in batch.keys():
            batch[i] = batch[i].to(device)

        out = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False, num_beams=args.num_beam, do_sample=True, max_new_tokens=512)
        print(out)
        pred_str = tokenizer.batch_decode(out, skip_special_tokens=True)
        print(pred_str)

        refer = tokenizer.batch_decode(torch.where(batch["labels"] == -100, tokenizer.pad_token_id, batch["labels"]), skip_special_tokens=True)
        refers.extend(refer)
        preds.extend(pred_str)
        print(refer, "\n\n\n")

        rouge.add_batch(predictions=pred_str, references=refer)

    rouge_result = rouge.compute()
    print(rouge_result)
    
    result_str_dict = dict()
    for idx,(r,p) in enumerate(zip(refers, preds)):
        result_str_dict[str(idx)] = {"ref":r, "pred":p}

    json.dump(result_str_dict, open(os.path.join(save_path,"test_result.json"), "w", encoding="utf8"), indent=2)

    result_df.loc[len(result_df.index)] = {"save_path":save_path, "rouge1":rouge_result["rouge1"], "rouge2":rouge_result["rouge2"], "rougeL":rouge_result["rougeL"], "rougeLsum":rouge_result["rougeLsum"]}
    result_df.to_csv("results.csv")
    
    wandb.log({"test_ROUGE1":rouge_result["rouge1"], "test_ROUGE2":rouge_result["rouge2"], "test_ROUGEL":rouge_result["rougeL"], "test_ROUGELsum":rouge_result["rougeLsum"]})