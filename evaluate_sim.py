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

from sklearn.metrics.pairwise import cosine_similarity
from transformers import WavLMModel, AutoFeatureExtractor



from dataset import create_dataloader, CustomDataset, collate_fn, convert_pandas_to_custom_format
import warnings

warnings.filterwarnings("ignore")

# load vocoder
from huggingface_hub import snapshot_download, hf_hub_download
#from pydub import AudioSegment, silence
from transformers import pipeline
from vocos import Vocos
from dataset import load_vocoder


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


feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-large")
wav_model = WavLMModel.from_pretrained("microsoft/wavlm-large")


# model initialization 
model = AudioFusion( 
    transformer = dict(
        dim = 512, depth = 8, dim_head = 64, heads = 8
    ),
    num_text_tokens = 256,
    modality_default_shape=(450, 100)
).cuda()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_embedding(waveform):
    """
    Extract speaker embedding from an audio file using WavLM
    
    Args:
        audio: waveform of audio
        
    Returns:
        numpy.ndarray: Speaker embedding
    """
    # Load audio file
    # waveform, sample_rate = torchaudio.load(audio_path)
    
    # Ensure mono channel
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample to 16kHz if necessary
    
    resampler = torchaudio.transforms.Resample(orig_freq=24_000, new_freq=16000)
    waveform = resampler(waveform)
    
    # Process audio with WavLM
    inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = wav_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]  # Get the [CLS] token embeddings
    
    return embeddings.cpu().numpy()

# load vocoder 
vocoder_local_path = "../checkpoints/charactr/vocos-mel-24khz"
vocoder = load_vocoder(vocoder_name='vocos', is_local=False, local_path=vocoder_local_path)


iter_dl = cycle(test_dataloader)

state_dict = torch.load('tts_training_500000.pth')

model.load_state_dict(state_dict)

total_score, valid_scores = 0, 0 

similarity_scores = []

for batch in test_dataloader:
    batch = [[inp.to(device) for inp in pair] for pair in batch]

    for audio in batch:
        audio = audio[1]
        gt_wave = vocoder.decode(batch[0][1].unsqueeze(0).permute(0, 2, 1)).cpu()
        
        mel_output = model.generate_modality_only() # 1, 615, 100 
        generated_wave = vocoder.decode(mel_output.permute(0, 2, 1)).cpu()

        gt_embedding = extract_embedding(gt_wave)
        generated_embedding = extract_embedding(generated_wave)

        sim_score = cosine_similarity(gt_embedding, generated_embedding)[0][0]
        similarity_scores.append(sim_score)

        total_score += sim_score 
        valid_scores += 1 
        

average_score = total_score / valid_scores 
print(average_score)
print(similarity_scores)