from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from datasets import load_dataset
import os
from tqdm import tqdm

data_name = "xsum"

DATA_INFO = {
    "cnn_dailymail": {"data_path":"abisee/cnn_dailymail","subset":"3.0.0","src_lang":"article", "tgt_lang":"highlights"},
    "xsum": {"data_path":"EdinburghNLP/xsum","subset":None,"src_lang":"document", "tgt_lang":"summary"},
}

data_path = DATA_INFO[data_name]["data_path"]
subset = DATA_INFO[data_name]["subset"]
src_lang = DATA_INFO[data_name]["src_lang"]
tgt_lang = DATA_INFO[data_name]["tgt_lang"]

def to_list_casting(data):
    data[src_lang] = dataset["train"][src_lang]
    data[tgt_lang] = dataset["train"][tgt_lang]
    return data

# Load the dataset en-de dataset
dataset = load_dataset(data_path, subset)
print(dataset)

data = []
for i in tqdm(range(len(dataset["train"]))):
    data.append(dataset["train"][i][src_lang])
    data.append(dataset["train"][i][tgt_lang])

# dataset = dataset.map(to_list_casting, num_proc=4)
# print(dataset)

# Initialize a tokenizer
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# Set up pre-tokenization
# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Initialize a trainer
trainer = trainers.BpeTrainer(vocab_size=30000, min_frequency=2, special_tokens=["<unk>", "<s>", "</s>", "<pad>"], show_progress=True)

tokenizer.train_from_iterator(data, trainer=trainer)
data_called = data_name.split("/")[-1]
os.makedirs("tokenizer", exist_ok=True)
# Save the trained tokenizer
tokenizer.save(f"tokenizer/{data_called}_{subset}_BPEtokenizer.json")