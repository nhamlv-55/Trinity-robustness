from maraboupy import MarabouCore, Marabou, MarabouUtils
from TrinityNet import TrinityNet
import torch
import pickle
import tempfile
import numpy as np
import numpy.typing as npt
from typing import List, Dict, Tuple
from settings import MARABOU_BIN
import os
import subprocess
import json
import sys
from parse import parse


ZERO = 10**-6
NUM_THREADS = 30
TIMEOUT = 600

cex_file = sys.argv[1]

#load the dataset
trinity = torch.load("trinity.pth", map_location=torch.device('cpu'))
with open("data_for_Marabou.pkl", "rb") as f:
    prepared_data = pickle.load(f)
print(prepared_data[0].keys())

#get the input_idx, true_label, and adv_label from the filename
items = parse("assignments_{}_{}vs{}.txt", os.path.basename(cex_file))
input_idx, true_label, adv_label = items
input_idx = int(input_idx)
true_label = int(true_label)
adv_label = int(adv_label)
input = prepared_data[input_idx]["h"]

#extract input and output var assignments computed by Marabou
v = []
with open(cex_file, "r") as f:
    data = f.readlines()

    for l in data[1:]:
        idx, val = l.strip().split(",")
        idx = int(idx)
        val = float(val)
        v.append(val)

input_vars: List[float] = v[:512]
logit_vars:List[float] = v[512:512+61]
probe_vars:List[float] = v[512+61:512+61+192]
output_vars: List[float] = v[512:512+61+192]

#run the input through the network
output = trinity(torch.Tensor(input_vars))
logits = output.squeeze()[:61]
probe = output.squeeze()[61:]

#assert that Marabou output and the computed output are the same
assert torch.allclose(output.squeeze(), torch.Tensor(output_vars), atol=10**-5), f"{output.squeeze()[:10]} != {output_vars[:10]}"
print("Pass: The output computed by Marabou is close enough to the output computed by Pytorch")

#assert that the logits at adv_label has higher value than the logits at true_label
assert logit_vars[adv_label] > logit_vars[true_label], f"{logit_vars[adv_label]} is not greater than {logit_vars[true_label]}"
print("Pass: The predicted value for adv_label is greater than the predicted value for true_label")

#assert that the board state stay the same
original_board = trinity(torch.Tensor(input)).squeeze()[61:]
original_board = torch.argmax(original_board.reshape(64, 3), dim = 1).reshape(8,8)

marabou_board = torch.argmax(torch.Tensor(probe_vars).reshape(64, 3), dim = 1).reshape(8,8)

assert torch.equal(original_board, marabou_board), f"{original_board} \n != {marabou_board}"
print("Pass: The board state are the same")