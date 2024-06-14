import torch
import argparse
import numpy as np


SETTING_DICT = {
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

#set seed function
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
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

    argparser.add_argument("--data_name", type=str, default="wmt14")
    argparser.add_argument("--subset", type=str, default="de-en")
    argparser.add_argument("--src_lang", type=str, default="en")
    argparser.add_argument("--tgt_lang", type=str, default="de")
    argparser.add_argument("--batch_size", type=int, default=16)
    argparser.add_argument("--tokenizer_path", type=str, default="tokenizer/wmt14_de-en_BPEtokenizer.json")
    argparser.add_argument("--pre_train_path", type=str, default="lucadiliello/bart-small")
    argparser.add_argument("--gpu", type=int, default=0)
    argparser.add_argument("--learning_rate", type=float, default=5e-5)
    argparser.add_argument("--weight_decay", type=float, default=0.01)
    argparser.add_argument("--epoch", type=int, default=10)
    argparser.add_argument("--num_beam", type=int, default=5)
    argparser.add_argument("--full_step", type=int, default=1000010)
    argparser.add_argument("--eval_step", type=int, default=50000)
    argparser.add_argument("--save_path", type=str, default="")
    argparser.add_argument("--baseline", default=False, action="store_true")
    argparser.add_argument("--max_norm", type=float, default=1.0)
    argparser.add_argument("--seed", type=int, default=42)
    argparser.add_argument("--logging_step", type=int, default=1000)

    args = argparser.parse_args()

    return args