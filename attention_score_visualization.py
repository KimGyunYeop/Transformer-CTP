from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders
from transformers import PreTrainedTokenizerFast, BartTokenizer, GPT2Tokenizer, GPT2Model, BartModel, BartForConditionalGeneration, AutoModelForPreTraining, AutoTokenizer
from modeling_mc_for_visualization import MixcoderForConditionalGeneration, MixcoderConfig
import torch
from matplotlib import pyplot as plt
import argparse
from datasets import load_dataset
import custom_tokenizer
import custom_datasets
import os
from torch.functional import F
from torchinfo import summary
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument("--data_name", type=str, default="wmt14")
argparser.add_argument("--subset", type=str, default="de-en")
argparser.add_argument("--src_lang", type=str, default="de")
argparser.add_argument("--tgt_lang", type=str, default="en")
argparser.add_argument("--batch_size", type=int, default=16)
argparser.add_argument("--tokenizer_path", type=str, default="tokenizer/wmt14_de-en_BPEtokenizer.json")
argparser.add_argument("--save_path", type=str, default="./results_base/wmt14_en-de/baseline-/1000000")
argparser.add_argument("--gpu", type=int, default=0)

args = argparser.parse_args()

os.makedirs("figs", exist_ok=True)

# args.save_path = "/home/nlplab/hdd1/gyop/research/GenrateFromCurrentPosition/results/wmt14_de-en/-avg_prev_token-share_att-indi_self_q-indi_self_out-share_cross_att-indi_cross_q-indi_cross_out-hidden_cross_att/1000000"
args.save_path = "/home/nlplab/hdd1/gyop/research/GenrateFromCurrentPosition/results/wmt14_de-en/-new_token-share_att-indi_self_q-indi_self_out-share_cross_att-indi_cross_q-indi_cross_out-hidden_cross_att/700000"
# args.save_path = "/home/nlplab/hdd1/gyop/research/GenrateFromCurrentPosition/results/wmt14_de-en/baseline-/1000000"
# "facebook/bart-base", "t5-base", "gpt2", "meta-llama/Meta-Llama-3-8B", "lucadiliello/bart-small"
# args.save_path = "lucadiliello/bart-small"

dataset = load_dataset(args.data_name, args.subset, split="test")
print("before filtering:")
print(dataset)


if "results" not in args.save_path:
    model = AutoModelForPreTraining.from_pretrained(args.save_path)
    tokenizer = AutoTokenizer.from_pretrained(args.save_path)
    
else:
    tokenizer = custom_tokenizer.get_tokenizer(args.tokenizer_path)
    
    if "baseline" in args.save_path:
        model = BartForConditionalGeneration.from_pretrained(args.save_path, local_files_only=True)
        # model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        # tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    else:
        model = MixcoderForConditionalGeneration.from_pretrained(args.save_path, local_files_only=True)
        if model.config.next_token_type == "new_token":
            tokenizer.add_tokens("<next>", special_tokens=True)
            next_token_id = tokenizer.convert_tokens_to_ids("<next>")
        else:
            next_token_id = None

if "gpt2" in args.save_path:
    tokenizer.pad_token = tokenizer.eos_token

print(sum(p.numel() for p in model.parameters()))
print(model.num_parameters(only_trainable=True, exclude_embeddings=True))
print(summary(model))

test_dataset = custom_datasets.WmtDataset(dataset, tokenizer=tokenizer, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=test_dataset.collate_fn)

print(model)
print(len(tokenizer))
input()

for idx, batch in enumerate(test_dataloader):
    if model.config.is_encoder_decoder:
        input_sentence = tokenizer.eos_token + tokenizer.batch_decode(batch["input_ids"][:,:-1])[0]
        print(input_sentence)
        label_sentence = tokenizer.batch_decode(torch.where(batch["labels"] == -100, tokenizer.pad_token_id, batch["labels"]), skip_special_tokens=False)[0]
        print(label_sentence)
        print(batch)
        out = model(**batch, output_attentions=True, output_hidden_states=True)
        
    else:
        batch["input_ids"] = torch.where(batch["labels"] == -100, tokenizer.pad_token_id, batch["labels"])
        input_sentence = tokenizer.eos_token + tokenizer.batch_decode(batch["input_ids"][:,:-1])[0]
        print(input_sentence)
        label_sentence = tokenizer.batch_decode(batch["labels"], skip_special_tokens=False)[0]
        print(label_sentence)
        out = model(input_ids = batch["input_ids"], output_attentions=True, output_hidden_states=True)
    print(out.keys())
    
    if hasattr(out, "decoder_attentions"):
        att = torch.stack(out.decoder_attentions)
        print(torch.stack(out.decoder_attentions).shape)
    else:
        att = torch.stack(out.attentions)
        print(torch.stack(out.attentions).shape)
        
    # att = torch.stack(out.attentions)
    # print(torch.stack(out.attentions).shape)
    # print(torch.mean(torch.mean(att, dim=0), dim=1).unsqueeze(0))

    os.makedirs("figs/"+str(idx), exist_ok=True)
    with open(f"figs/{idx}/data.txt", "w") as f:
        f.write(f"source: {input_sentence}\ntarget: {label_sentence}\n")
        
    plt.matshow(torch.mean(torch.mean(att, dim=0), dim=1).squeeze()[:,:].detach().numpy(), vmin=0, vmax=1)
    plt.savefig(f"figs/{idx}/att.png")
    plt.clf()
    for i in range(att.size(0)):
        # print(torch.mean(att[i,0,:,:], dim=0).unsqueeze(0))
        plt.matshow(torch.mean(att[i,0,:,:,:], dim=0).squeeze()[:,:].detach().numpy(), vmin=0, vmax=1)
        plt.savefig(f"figs/{idx}/att_{i}.png")
        plt.clf()
    
    print(idx)

    if hasattr(out, "decoder_hidden_states"):
        input_embs = out.decoder_hidden_states[0]
        last_hidden_state = out.decoder_hidden_states[-1]
    elif hasattr(out, "hidden_states"):
        input_embs = out.hidden_states[0]
        last_hidden_state = out.hidden_states[-1]
    print(input_embs.shape)
    print(last_hidden_state.shape)
    
    print(batch["labels"].shape)
    print(out["logits"].shape)
    print(tokenizer.convert_ids_to_tokens(batch["input_ids"].squeeze().tolist()))
    print(tokenizer.convert_ids_to_tokens(torch.where(batch["labels"] == -100, tokenizer.pad_token_id, batch["labels"]).squeeze().tolist()))
    print(tokenizer.convert_ids_to_tokens(out["logits"].argmax(-1).squeeze().tolist()))
    print(tokenizer.batch_decode(batch["input_ids"]))
    print(tokenizer.batch_decode(torch.where(batch["labels"] == -100, tokenizer.pad_token_id, batch["labels"]), skip_special_tokens=True))
    print(tokenizer.batch_decode(out["logits"].argmax(-1), skip_special_tokens=True))
    
    # l2d = (input_embs - last_hidden_state).pow(2).sum(2).sqrt().T
    # cs = F.cosine_similarity(input_embs, last_hidden_state, dim=-1).T

    # print(l2d)
    # print(torch.mean(l2d, dim=0))
    # print(cs)
    # print(torch.mean(cs, dim=0))
    
    input()