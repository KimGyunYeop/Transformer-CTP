import datasets
from transformers import AdamW, get_scheduler, BartConfig, BartForConditionalGeneration, BartTokenizer
import torch
from datasets import load_dataset
import custom_datasets
import custom_tokenizer
from modeling_mc import MixcoderForConditionalGeneration, MixcoderConfig
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

argparser.add_argument("--data_name", type=str, default="wmt14")
argparser.add_argument("--subset", type=str, default="de-en")
argparser.add_argument("--src_lang", type=str, default="en")
argparser.add_argument("--tgt_lang", type=str, default="de")
argparser.add_argument("--batch_size", type=int, default=16)
argparser.add_argument("--gpu", type=int, default=0)
argparser.add_argument("--learning_rate", type=float, default=5e-5)
argparser.add_argument("--weight_decay", type=float, default=0.01)
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

data_name = args.data_name
subset = args.subset
batch_size = args.batch_size
tokenizer_path = f"tokenizer/{args.data_name}_{args.subset}_BPEtokenizer.json"
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

if args.base:
    n_layer=6
    d_model=768
    decoder_layers=6
    decoder_attention_heads=12
    decoder_ffn_dim=3072
    encoder_layers=6
    encoder_attention_heads=12
    encoder_ffn_dim=3072
    max_position_embeddings=1024

    save_path = os.path.join("results_base",f"{args.data_name}_{args.src_lang}-{args.tgt_lang}", save_path)
    wandb.init(project=f"MixCoder_base_{args.data_name}_{args.subset}_{args.src_lang}-{args.tgt_lang}", name=save_path, config=vars(args))

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

    save_path = os.path.join("results",f"{args.data_name}_{args.src_lang}-{args.tgt_lang}", save_path)
    wandb.init(project=f"MixCoder_{args.data_name}_{args.subset}_{args.src_lang}-{args.tgt_lang}", name=save_path, config=vars(args))


# if os.path.exists(save_path):
#     input("this path already exists. press enter to continue.")

os.makedirs(save_path, exist_ok=True)
json.dump(vars(args), open(os.path.join(save_path, "args.json"), "w", encoding="utf8"), indent=2)

# wandb.log({"loss":np.mean(logging_losses), "_step":cur_step, "BLEU":matric_bleu_result["bleu"], "sacreBLEU":matric_scarebleu_result["score"], "sacreBLEU_v14":matric_scarebleu_v14_result["score"]})
wandb.define_metric("BLEU", summary="max")
wandb.define_metric("sacreBLEU", summary="max")
wandb.define_metric("sacreBLEU_v14", summary="max")
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

dataset = dataset.filter(lambda x: len(x["translation"][args.src_lang]) < 1024 and len(x["translation"][args.tgt_lang]) < 1024)
print("after filtering:")
print(dataset)


if args.baseline:
    tokenizer = custom_tokenizer.get_tokenizer(tokenizer_path)
    bartconfig = BartConfig(n_layer=n_layer,
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
                            vocab_size=len(tokenizer),
                            )

    model = BartForConditionalGeneration(config=bartconfig)
    model.to(device)

elif args.pre_trained_baseline:
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    model.to(device)

else:
    tokenizer = custom_tokenizer.get_tokenizer(tokenizer_path)
    print(len(tokenizer))
    if next_token_type == "new_token":
        tokenizer.add_tokens("<next>", special_tokens=True)
        next_token_id = tokenizer.convert_tokens_to_ids("<next>")
    else:
        next_token_id = None
    print(len(tokenizer))

    mixcoder_config = MixcoderConfig(n_layer=n_layer,
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
                                    vocab_size=len(tokenizer),
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
                            
    model = MixcoderForConditionalGeneration(config=mixcoder_config)

    if next_token_type == "new_token":
        model.resize_token_embeddings(len(tokenizer))

    model.to(device)

print(model)
with open(os.path.join(save_path, "model.txt"), "w", encoding="utf8") as f:
    f.write(str(model))

train_dataset = custom_datasets.WmtDataset(dataset["train"], tokenizer=tokenizer, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
val_dataset = custom_datasets.WmtDataset(dataset["validation"], tokenizer=tokenizer, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
test_dataset = custom_datasets.WmtDataset(dataset["test"], tokenizer=tokenizer, src_lang=args.src_lang, tgt_lang=args.tgt_lang)

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
result_dict_sacre_bleu = {}
result_dict_bleu = {}
result_dict_bleu_v14 = {}
result_dict_sacre_bleu_v14 = {}
logging_losses = []
best_bleu = 0
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


        if cur_step%eval_step == 0:
            model.eval()
            
            matric_scarebleu = evaluate.load("sacrebleu")
            matric_bleu = evaluate.load("bleu")
            matric_scarebleu_v14 = evaluate.load("sacrebleu")
            matric_bleu_v14 = evaluate.load("bleu")
            with torch.no_grad():
                refers = []
                preds = []
                for batch in tqdm(val_dataloader):
                    for i in batch.keys():
                        batch[i] = batch[i].to(device)

                    # out = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    out = model(**batch)
                    pred = out.logits.argmax(dim=-1)
                    pred_str = tokenizer.batch_decode(pred, skip_special_tokens=True)

                    refer = tokenizer.batch_decode(torch.where(batch["labels"] == -100, tokenizer.pad_token_id, batch["labels"]), skip_special_tokens=True)
                    refers.extend(refer)
                    preds.extend(pred_str)

                    matric_scarebleu.add_batch(predictions=pred_str, references=refer)
                    matric_bleu.add_batch(predictions=pred_str, references=refer)
                    matric_scarebleu_v14.add_batch(predictions=pred_str, references=refer)
                    matric_bleu_v14.add_batch(predictions=pred_str, references=refer)
                    # print(pred_str)

                # matric.add_batch(predictions=preds, references=refers)
                # matric_result=matric_scarebleu.compute(predictions=preds, references=refers)
                matric_scarebleu_result = matric_scarebleu.compute()
                result_dict_sacre_bleu[str(cur_step)] = matric_scarebleu_result
                matric_bleu_result = matric_bleu.compute()
                result_dict_bleu[str(cur_step)] = matric_bleu_result
                matric_scarebleu_v14_result = matric_scarebleu_v14.compute(tokenize="intl")
                result_dict_bleu_v14[str(cur_step)] = matric_scarebleu_v14_result
                # matric_bleu_v14_result = matric_bleu_v14.compute(tokenizer="intl")
                # result_dict_sacre_bleu_v14[str(cur_step)] = matric_bleu_v14_result
                if matric_scarebleu_v14_result["score"] > best_bleu:
                    best_bleu = matric_scarebleu_v14_result["score"]
                    best_step = cur_step
                
                os.makedirs(os.path.join(save_path,str(cur_step)), exist_ok=True)
                model.save_pretrained(os.path.join(save_path,str(cur_step)), safe_serialization=False)

                result_str_dict = dict()
                for idx,(r,p) in enumerate(zip(refers, preds)):
                    result_str_dict[str(idx)] = {"ref":r, "pred":p}

                json.dump(result_str_dict, open(os.path.join(save_path,str(cur_step),"validation_result.json"), "w", encoding="utf8"), indent=2)
                json.dump(result_dict_sacre_bleu, open(os.path.join(save_path,"result_scareBLEU.json"), "w", encoding="utf8"), indent=2)
                json.dump(result_dict_bleu, open(os.path.join(save_path,"result_BLEU.json"), "w", encoding="utf8"), indent=2)
                json.dump(result_dict_bleu_v14, open(os.path.join(save_path,"result_scareBLEU_v14.json"), "w", encoding="utf8"), indent=2)
                json.dump(result_dict_sacre_bleu_v14, open(os.path.join(save_path,"result_BLEU_v14.json"), "w", encoding="utf8"), indent=2)

                wandb.log({"loss":np.mean(logging_losses), "_step":cur_step, "BLEU":matric_bleu_result["bleu"], "sacreBLEU":matric_scarebleu_result["score"], "sacreBLEU_v14":matric_scarebleu_v14_result["score"]})
                logging_losses = []
            model.train()
        
        elif cur_step % args.logging_step == 0:
            wandb.log({"loss":np.mean(logging_losses), "_step":cur_step})
            logging_losses = []
            
        if cur_step > full_step:
            break


if os.path.exists(os.path.join("results.csv")):
    result_df = pd.read_csv("results.csv", index_col = 0)
else:
    result_df = pd.DataFrame(columns=["save_path", "bleu", "scare_bleu", "scare_bleu_v14"])  

model.from_pretrained(os.path.join(save_path,str(best_step)), local_files_only=True)
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

        out = model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False, num_beams=args.num_beam, do_sample=True, max_new_tokens=512)
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
    
    result_str_dict = dict()
    for idx,(r,p) in enumerate(zip(refers, preds)):
        result_str_dict[str(idx)] = {"ref":r, "pred":p}

    json.dump(result_str_dict, open(os.path.join(save_path,"test_result.json"), "w", encoding="utf8"), indent=2)

    result_df.loc[len(result_df.index)] = {"save_path":save_path, "bleu":matric_bleu_result["bleu"], "scare_bleu":matric_scarebleu_result["score"], "scare_bleu_v14":matric_scarebleu_v14_result["score"]}
    result_df.to_csv("results.csv")
    
    wandb.log({"test_BLEU":matric_bleu_result["bleu"], "test_sacreBLEU":matric_scarebleu_result["score"], "test_sacreBLEU_v14":matric_scarebleu_v14_result["score"]})