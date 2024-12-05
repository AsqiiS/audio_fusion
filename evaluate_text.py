import os

import math
from functools import partial
from collections import defaultdict
from typing import NamedTuple, Callable, Literal

import torch
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear
from torch.nn.utils.rnn import pad_sequence
from torch.utils._pytree import tree_map
from torch import nn, Tensor, tensor, is_tensor, stack

from ema_pytorch import EMA

from functools import partial, wraps

from loguru import logger

import torchaudio
from typing import Optional

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

from torchdiffeq import odeint

import numpy as np 
import librosa 
import pandas as pd
from random import random  

from torch.utils.data import Dataset, Sampler
from datasets import load_dataset, load_from_disk
from datasets import Dataset as Dataset_

from functools import partial 

from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from torch.nn.utils.rnn import pad_sequence

import einx 
from einops import rearrange, repeat, reduce, einsum, pack
from einops.layers.torch import Rearrange

import random 
import json 
import jaxtyping
from tqdm import tqdm

from main_model import AudioFusion 
from sklearn.model_selection import train_test_split
from utils import cycle 

from dataset import create_dataloader, CustomDataset, collate_fn, convert_pandas_to_custom_format

# dataset 
extract_dir = "./LJSpeech-1.1"
# List the files in the extracted directory
dataset_dir = '/home/askhat.sametov/LJSpeech-1.1/LJSpeech-1.1'
extracted_files = os.listdir('/home/askhat.sametov/LJSpeech-1.1/LJSpeech-1.1')
audio_folder = os.path.join(dataset_dir, 'wavs')

lj = pd.read_csv(os.path.join(dataset_dir, 'metadata.csv'), sep='|', header=None, names=['ID', 'Transcription', 'Normalized_Transcription'])
lj = lj.dropna()

custom_data = convert_pandas_to_custom_format(lj, audio_folder)

train_data, test_data = train_test_split(custom_data, test_size=0.1, random_state=42)
train_dataset = CustomDataset(train_data)
test_dataset = CustomDataset(test_data)

# custom_mel_dataset = CustomDataset(custom_data) 
train_dataloader = create_dataloader(train_dataset, batch_size = 4, shuffle = True)
test_dataloader = create_dataloader(test_dataset, batch_size = 4, shuffle = False) 


# model initialization 
model = AudioFusion( 
    transformer = dict(
        dim = 512, depth = 8, dim_head = 64, heads = 8
    ),
    num_text_tokens = 256,
    modality_default_shape=(450, 100)
).cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# training setup 
iter_dl = cycle(test_dataloader)

state_dict = torch.load('tts_training_500000.pth')

model.load_state_dict(state_dict)

# evaluation of text 
GENERATE_LENGTH = 256 

total_log_likelihood = 0
total_tokens = 0

for batch in test_dataloader:
    batch = [[inp.to(device) for inp in pair] for pair in batch]

    with torch.no_grad():
        loss = model(batch)

        seq_lens = [seq_len[0].size()[0] for seq_len in batch]
        max_len = max(seq_lens)

        total_log_likelihood += loss[1][1].item() * max_len
        total_tokens += max_len


avg_log_likelihood = total_log_likelihood / total_tokens 
perplexity = math.exp(avg_log_likelihood)

print(perplexity)