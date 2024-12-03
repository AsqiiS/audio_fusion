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

# character based tokenizer

def char_tokenize(
    text,
    device = None,
    offset = 0
) -> Tensor:
    return tensor([*map(ord, text)], device = device) + offset


def decode_chars(
    t,
    offset = 0,
) -> str:
    byte_list = (t - offset).clamp(min = 0, max = 127).tolist()
    return ''.join([*map(chr, byte_list)])


import jaxtyping

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]

Float = TorchTyping(jaxtyping.Float)
Int   = TorchTyping(jaxtyping.Int)
Bool  = TorchTyping(jaxtyping.Bool)

# types

Scalar = Float['']

ModalitySample = list[Int[''] | Int['_'] | Float['...'] | tuple[int, Float['...']]]

ModalityTokenTransform = str | Callable | None

RawModalityPositions = list[list[tuple[int, int, int]]]

GetPredFlows = dict[int, list[Callable[[Tensor], Tensor]]]


class LossBreakdown(NamedTuple):
    total: Float['']
    text: Float['']
    flow: list[Float['']]
    velocity: list[Float['']] | None
    direction: list[Float['']] | None

def get_tokens_since_rightmost_id(
    t,
    rightmost_id
) -> Tensor:
    """
    ex. [9] [2] [8] [4] [7]
    2 would return [8] [4] [7]
    """

    mask = t == rightmost_id

    if not mask.any():
        return t[0:0] # return empty tensor if no id found

    reverse_cumsum = mask.flip(dims = (0,)).cumsum(dim = 0).flip(dims = (0,))
    after_right_mask = reverse_cumsum == 0
    return t[after_right_mask]

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def first(it):
    return it[0]

def prepend(arr, el):
    arr.insert(0, el)

def join(arr, delimiter = ''):
    return delimiter.join(arr)

def divisible_by(num, den):
    return (num % den) == 0

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def tree_map_tensor(sample, fn: Callable):
    return tree_map(lambda t: t if not is_tensor(t) else fn(t), sample)


# adds batch into input 
def add_temp_batch_dim(fn: Callable):
    @wraps(fn)
    def inner(t: Tensor, *args, **kwargs) -> Tensor:
        t = rearrange(t, '... -> 1 ...')
        out = fn(t, *args, **kwargs)
        out = rearrange(out, '1 ... -> ...')
        return out
    return inner

# sets to evaluation mode 
def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner


# tensor helpers

def l2norm(t):
    return F.normalize(t, dim = -1)

def softclamp(t, value = 50.):
    return (t / value).tanh() * value

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    noise = gumbel_noise(t) * int(temperature > 0)
    return (t / temperature + noise).argmax(dim = dim, keepdim = keepdim)


# converts modality shape of type string to tuple "3,224,244" -> (3,224,244)
def default_to_modality_shape_fn(maybe_shape_str):
    return tuple([*map(int, maybe_shape_str.split(','))])

# default function for translating modality length to times (noise level, where 0 is highest noise)
# int(b, m) -> float(b, m), b - batch size, m - modality length 
def random_modality_length_to_time_fn(modality_length):
    return torch.rand_like(modality_length.float())

# maps modality lengths into structured time values for noise scheduling 
# Assigns structured noise times (0.5) to processed modalities, and random noise to unprocessed modalities
def default_modality_length_to_time_fn(modality_length):
    total_modalities, device = modality_length.shape[-1], modality_length.device

    num_modalities = (modality_length > 0).sum(dim = -1).float()
    rand_num_modalities = torch.floor(torch.rand_like(num_modalities) * num_modalities)
    seq = torch.arange(total_modalities, device = device)

    prev_decoded_modality = einx.less('m, b -> b m', seq, rand_num_modalities)
    curr_modality_rand_time = torch.rand_like(num_modalities)

    # in paper, they fix previous decoded modalities to 500 / 1000 steps for discrete ddpm, here using flow matching with times 0 - 1 so corresponds to 0.5
    return einx.where('b m, , b -> b m', prev_decoded_modality, 0.5, curr_modality_rand_time)

# def default_modality_length_to_time_fn(num_modalities: Int['b']) -> Float['b m']:
#     batch, device = num_modalities.shape[0], num_modalities.device
#     total_modalities = num_modalities.amax().item()

#     if total_modalities == 0:
#         return torch.empty((batch, 0), device = device, dtype = torch.float)

#     rand_num_modalities = torch.floor(torch.rand_like(num_modalities.float()) * num_modalities)
#     seq = torch.arange(total_modalities, device = device)

#     prev_decoded_modality = einx.less('m, b -> b m', seq, rand_num_modalities)
#     curr_modality_rand_time = torch.rand_like(num_modalities.float())

#     # in paper, they fix previous decoded modalities to 500 / 1000 steps for discrete ddpm, here using flow matching with times 0 - 1 so corresponds to 0.5
#     return einx.where('b m, , b -> b m', prev_decoded_modality, 0.5, curr_modality_rand_time)

# pretty print
# text is int or long, modality is float
def print_modality_sample(modality_sample):
    output = []

    for sample in modality_sample:
        if isinstance(sample, tuple):
            modality_type, sample = sample
            output.append((f'modality:{modality_type}', sample.shape))
        elif sample.dtype in (torch.int, torch.long):
            output.append(('text', sample.shape))
        else:
            output.append(('modality', sample.shape))
    
    logger.info(output)

# masking 
# flex attention mask construction
# https://pytorch.org/blog/flexattention/

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

# offset - starting index of modality in sequence, length - length of modality in sequence 
def modality(offset, length):

    def mask_fn(b, h, q_idx, kv_idx):
        return (q_idx >= offset) & (kv_idx < (offset + length))

    return mask_fn

# modalities: int(b m 3)
def transfusion_attn_mask(modalities):
    modalities = modalities.long()

    def mask_mod(b, h, q_idx, kv_idx):
        mask = causal(b, h, q_idx, kv_idx)

        modality_batch = modalities[b]

        for _, offset, length in modality_batch:
            mask = mask | modality(offset, length)(b, h, q_idx, kv_idx)

        return mask

    return mask_mod

# normalized attention scores 
def softcap_score_mod(softcap):
    def inner(score, b, h, q_idx, kv_idx):
        score = score / softcap
        score = torch.tanh(score)
        score = score * softcap
        return score
    return inner

# losses
# calculates directional cosine similarity-based score, outupt is scalar loss value for each vector pair 
def calc_direction_loss(pred, target):
    return 0.5 * (1. - einsum(l2norm(pred), l2norm(target), '... d, ... d -> ...'))


# converting a raw list of modality offsets and lengths to tensor
# input int(b m 2) -> int(b m 3), modalities padded 
def modality_positions_to_tensor(
    modalities,
    pad_value = 0,
    device = None
):

    modalities: list[Tensor] = [tensor(modality, device = device) for modality in modalities]
    modalities = pad_sequence(modalities, padding_value = pad_value)

    if modalities.ndim == 2:
        modalities = modalities.reshape(*modalities.shape, 3)

    return modalities.long()

# sanitizing modalities tensor, making sure it is ordered
# input int(b m 3) -> tuple(int(b m 3), int(b m)), returns modalities sorted in ascending order by offset  
# last dim - (type, offset, modality length)
def order_modality_positions_by_seq_offset(
    modalities
):

    type, offsets, lengths = modalities.unbind(dim = -1)

    no_modality_mask = lengths <= 0 # there may be uneven number of modalities per batch sample
    offsets_to_sort = offsets.masked_fill(no_modality_mask, 1e10)
    _, sorted_indices = offsets_to_sort.sort(dim = -1)

    # sort by ascending offset

    modalities = einx.get_at('b [mi] ..., b mo -> b mo ...', modalities, sorted_indices)
    return modalities, sorted_indices


# functions for managing modality token mask
# input modalities int(b m 3), offset int(2) -> bool(b t m n)
def modality_positions_to_is_modality_mask(
    seq_len,
    modalities,
    offset = None,
    device = None,
    num_modalities = 1
):

    device = modalities.device

    if exists(offset):
        offset = F.pad(offset, (1, 0))
        modalities = modalities + offset.to(modalities)

    seq = torch.arange(seq_len, device = device)
    type_seq = torch.arange(num_modalities, device = device)

    modality_types = modalities[..., 0]

    left, right = modalities[..., 1:].cumsum(dim = -1).unbind(dim = -1)

    is_instance_for_type = einx.equal('b m, t -> b t m', modality_types, type_seq)

    is_modality_along_seq = (
        einx.greater_equal('i, b m -> b m i', seq, left) &
        einx.less('j, b m -> b m j', seq, right)
    )

    return einx.logical_and('b t m, b m n -> b t m n', is_instance_for_type, is_modality_along_seq)


# deriving relative positions from modality positions
# ex. given a sequence of 10 with an image at offset 3 with length 4 - [t] [t] [t] [i] [i] [i] [i] [t] [t] [t]
# relative positions for rotary will be [0] [1] [2] [3] [3] [3] [3] [4] [5] [6]
# rationale is that each modality will need the same position so there is no distance when conducting bidirectional attention, but should still have a relative distance to other text tokens and modalities
# input int(b m 3) -> int(b n)
def derive_rotary_positions_from_modality_positions(
    seq_len,
    modalities
):

    device = modalities.device

    modality_mask = modality_positions_to_is_modality_mask(seq_len, modalities, offset = torch.tensor([1, -1]))
    is_any_modality = reduce(modality_mask, 'b t m n -> b n', 'any')

    return torch.arange(seq_len, device = device) - is_any_modality.cumsum(dim = -1)


# modality tokens are given as list of tensors, can be then be embedded into the modality tokens for attending alongside text tokens
# aligns modality specific embeddings into a shared sequence with text tokens 
# output of shape (batch, seq_len, dim) 
def embed_modality_tokens(
    seq_len: int,
    dim: int,
    modality_tokens: list[list[Float['...']]],
    modalities: Int['b m 3'],
    modality_id: int,
    channel_first: bool
) -> Float['b n d']:

    batch, device = modalities.shape[0], modalities.device

    shape = (batch, seq_len, dim) if not channel_first else (batch, dim, seq_len)
    output = torch.zeros(shape, device = device)

    for batch_ind, (one_modality, one_modality_token) in enumerate(zip(modalities, modality_tokens)):
        for (modality_type, offset, length), batch_modality_token in zip(one_modality, one_modality_token):

            if modality_id != modality_type or length <= 0:
                continue

            modality_shape = batch_modality_token.shape

            if channel_first:
                mod_dim, *mod_axial_shape = modality_shape
                batch_modality_token = rearrange(batch_modality_token, 'd ... -> d (...)')
            else:
                *mod_axial_shape, mod_dim = modality_shape
                batch_modality_token = rearrange(batch_modality_token, '... d -> (...) d')

            mod_length = math.prod(mod_axial_shape)

            assert length == mod_length, f'received a modality of shape {modality_shape} but sequence length in modalities info is {length}'
            assert dim == mod_dim, f'received modality [{modality_id}] with shape {modality_shape} but expected dimension of {dim}'

            if channel_first:
                output[batch_ind, :, offset:(offset + length)] = batch_modality_token
            else:
                output[batch_ind, offset:(offset + length), :] = batch_modality_token

    return output



# input int(b m 3) -> bool(b i j)
def naive_attn_mask(
    seq_len,
    modalities,
    device = None
):

    _, offsets, length = modalities.unbind(dim = -1)

    seq = torch.arange(seq_len, device = device)

    is_causal = einx.greater_equal('i, j -> i j', seq, seq)

    is_modality = (
        einx.greater_equal('i, b m -> b m i 1', seq, offsets) &
        einx.less('j, b m -> b m 1 j', seq, offsets + length)
    )

    return is_causal | is_modality.any(dim = 1)

# sampling related functions

# min_p for text
# https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

def append_dims(t, ndims):
    return t.reshape(*t.shape, *((1,) * ndims))

def is_empty(t):
    return t.numel() == 0


def pack_one_with_inverse(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, packed_shape, inv_pattern)[0]

    return packed, inverse


def divisible_by(num, den):
    return (num % den) == 0
