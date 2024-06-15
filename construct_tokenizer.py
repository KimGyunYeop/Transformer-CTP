from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from datasets import load_dataset
import os
from tqdm import tqdm
import argparse


argparser = argparse.ArgumentParser()
argparser.add_argument("--data_name", type=str, default="wmt14")
argparser.add_argument("--subset", type=str, default="de-en")
argparser.add_argument("--src_lang", type=str, default="en")
argparser.add_argument("--tgt_lang", type=str, default="de")
args = argparser.parse_args()

def to_list_casting(data):
    data[args.src_lang] = wmt14["train"]["translation"][args.src_lang]
    data[args.tgt_lang] = wmt14["train"]["translation"][args.tgt_lang]
    return data

# Load the WMT14 en-de dataset
wmt14 = load_dataset(args.data_name, args.subset)

data = []
for i in tqdm(range(len(wmt14["train"]["translation"]))):
    data.append(wmt14["train"][i]["translation"][args.src_lang])
    data.append(wmt14["train"][i]["translation"][args.tgt_lang])

# wmt14 = wmt14.map(to_list_casting, num_proc=4)
# print(wmt14)

# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# Set up pre-tokenization
# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Initialize a trainer
trainer = trainers.BpeTrainer(vocab_size=37000, min_frequency=2, special_tokens=["<unk>", "<s>", "</s>", "<pad>"], show_progress=True)

tokenizer.train_from_iterator(data, trainer=trainer)

os.makedirs("tokenizer", exist_ok=True)
# Save the trained tokenizer
tokenizer.save(f"tokenizer/{args.data_name}_{args.subset}_BPEtokenizer.json")