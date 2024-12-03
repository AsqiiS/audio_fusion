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

from utils import (
    char_tokenize,
    decode_chars,
    get_tokens_since_rightmost_id,
    exists,
    default,
    identity,
    first,
    prepend,
    join,
    divisible_by,
    cast_tuple,
    tree_map_tensor,
    add_temp_batch_dim,
    eval_decorator,
    l2norm,
    softclamp,
    max_neg_value,
    log,
    gumbel_noise,
    gumbel_sample,
    default_to_modality_shape_fn,
    random_modality_length_to_time_fn,
    default_modality_length_to_time_fn,
    print_modality_sample,
    causal,
    modality,
    transfusion_attn_mask,
    softcap_score_mod,
    calc_direction_loss,
    modality_positions_to_tensor,
    order_modality_positions_by_seq_offset,
    modality_positions_to_is_modality_mask,
    derive_rotary_positions_from_modality_positions,
    embed_modality_tokens,
    naive_attn_mask,
    min_p_filter,
    append_dims,
    is_empty,
    pack_one_with_inverse
)

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


# generates position embeddings 
class MLPAxialPositions(Module): 
    def __init__(
        self,
        *,
        num_dimensions, # 2 for images, 3 for video, etc etc
        dim,
        expand_factor = 2.
    ):
        super().__init__()
        self.num_dimensions = num_dimensions
        dim_hidden = int(dim * expand_factor)

        self.mlp = nn.Sequential(
            nn.Linear(num_dimensions, dim),
            nn.SiLU(),
            nn.Linear(dim, dim_hidden),
            nn.SiLU(),
            nn.Linear(dim_hidden, dim)
        )

        # tensor typing

        self._d = dim

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        modality_shape,
        flatten_dims = False
    ):

        if isinstance(modality_shape, torch.Size):
            modality_shape = tensor(modality_shape)

        modality_shape = modality_shape.to(self.device)

        assert len(modality_shape) == self.num_dimensions
        dimensions = modality_shape.tolist()

        grid = torch.meshgrid([torch.arange(dim_len, device = self.device) for dim_len in dimensions], indexing = 'ij')
        axial_positions = torch.stack(grid, dim = -1)

        pos_emb = self.mlp(axial_positions.float())

        if flatten_dims:
            #pos_emb = rearrange(pos_emb, '... d -> (...) d')
            pos_emb = pos_emb.reshape(-1, pos_emb.shape[-1])

        return pos_emb

# random fourier embedding

class RandomFourierEmbed(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        self.dim = dim
        self.register_buffer('weights', torch.randn(dim // 2))

    def forward(
        self,
        times
    ):

        if times.ndim == 1:
            times = rearrange(times, 'b -> b 1')

        freqs = einx.multiply('... i, j -> ... i j', times, self.weights) * 2 * torch.pi
        fourier_embed, _ = pack((times, freqs.sin(), freqs.cos()), 'b n *')
        return fourier_embed
    

# adaptive layernorm and ada-ln zero rolled into one wrapper
# from DiT paper and sota for time conditioning for now

class AdaptiveWrapper(Module): 
    def __init__(
        self,
        fn,
        dim,
        dim_cond,
        ada_ln_zero_init_bias = -2
    ):
        super().__init__()
        self.fn = fn
        self.dim = dim
        self.dim_cond = dim_cond

        self.layernorm = nn.LayerNorm(dim, elementwise_affine = False)

        # text will be subjected to normal layernorm bias
        # and for output will use layerscale

        self.layernorm_gamma = nn.Parameter(torch.zeros(dim))
        self.layerscale = nn.Parameter(torch.zeros(dim))

        # modalities will get the adaptive layernorm + ada-ln zero

        self.to_film = Linear(dim_cond, dim * 2)
        self.to_ada_ln_zero = Linear(dim_cond, dim)

        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_ada_ln_zero.weight)
        nn.init.constant_(self.to_ada_ln_zero.bias, ada_ln_zero_init_bias)

    def forward_text(
        self,
        x,
        **kwargs
    ):
        x = self.layernorm(x)

        x = x * (self.layernorm_gamma + 1.)

        out = self.fn(x, **kwargs)

        (out, *rest), tree_spec = tree_flatten(out)

        out = out * (self.layerscale + 1.)

        out = tree_unflatten((out, *rest), tree_spec)

        return out

    def forward_modality(
        self,
        x,
        cond,
        **kwargs
    ):
        x = self.layernorm(x)

        gamma, beta = self.to_film(cond).chunk(2, dim = -1)

        modality_tokens = x * (gamma + 1.) + beta

        # attention or feedforwards

        out = self.fn(modality_tokens, **kwargs)

        (out, *rest), tree_spec = tree_flatten(out)

        # take care of conditioning output separately for text vs modality

        modalities_out = out * self.to_ada_ln_zero(cond).sigmoid()

        # take care of function returning cache or value residual

        modalities_out = tree_unflatten((modalities_out, *rest), tree_spec)

        return modalities_out


    def forward(
        self,
        x,
        cond = None,
        is_any_modality = None,
        modality_only = False,
        **kwargs
    ):
        if exists(cond) and cond.ndim == 2:
            cond = rearrange(cond, 'b d -> b 1 d')

        if modality_only:
            return self.forward_modality(x, cond = cond, **kwargs)

        assert not (exists(cond) ^ exists(is_any_modality))

        has_modality = exists(is_any_modality)

        if not has_modality:
            return self.forward_text(x, **kwargs)

        if isinstance(is_any_modality, bool):
            is_any_modality = torch.full((x.shape[:-1]), is_any_modality, device = x.device, dtype = torch.bool)

        is_any_modality = rearrange(is_any_modality, '... -> ... 1')

        x = self.layernorm(x)

        gamma, beta = self.to_film(cond).chunk(2, dim = -1)

        text_tokens = x * (self.layernorm_gamma + 1.)

        modality_tokens = x * (gamma + 1.) + beta

        x = torch.where(is_any_modality, modality_tokens, text_tokens)

        # attention or feedforwards

        out = self.fn(x, **kwargs)

        (out, *rest), tree_spec = tree_flatten(out)

        # take care of conditioning output separately for text vs modality

        text_out = out * (self.layerscale + 1.)

        modalities_out = out * self.to_ada_ln_zero(cond).sigmoid()

        conditioned_out = torch.where(is_any_modality, modalities_out, text_out)

        # take care of function returning cache or value residual

        conditioned_out = tree_unflatten((conditioned_out, *rest), tree_spec)

        return conditioned_out

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return l2norm(x) * self.scale * (self.gamma + 1.) # use unit offset from Ohad Rubin

class GEGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return F.gelu(gates) * x

def FeedForward(
    dim,
    expansion_factor = 4.,
    dropout = 0.
):
    dim_inner = int(dim * expansion_factor * 2 / 3)
    return nn.Sequential(
        Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        Linear(dim_inner, dim)
    )

class RearrangeQKV(nn.Module):
    def __init__(self, heads, dim_head):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head

    def forward(self, x):
        b, n, _ = x.shape 
        x = x.reshape(b, n, 3, self.heads, self.dim_head)
        x = x.permute(2, 0, 3, 1, 4)

        return x 

class RearrangeGates(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads

    def forward(self, x):
        # Assume input x has shape (b, n, heads)
        b, n, h = x.shape
        assert h == self.heads, f"Expected heads dimension to be {self.heads}, but got {h}"
        
        # Permute to shape (b, heads, n, 1)
        x = x.permute(0, 2, 1).unsqueeze(-1)
        return x


class RearrangeOut(nn.Module):
    def forward(self, x):
        # Assume x has shape (b, h, n, d)
        b, h, n, d = x.shape
        # Permute and reshape to (b, n, h * d)
        x = x.permute(0, 2, 1, 3).reshape(b, n, h * d)
        return x


class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        softcap_value = 50.,
        gate_values = True
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        self.heads = heads 
        self.dim_head = dim_head 
        self.gate_values = gate_values 

        self.to_qkv = nn.Sequential(
            Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
            #RearrangeQKV(heads=self.heads, dim_head=self.dim_head)
        )

        self.to_gates = nn.Sequential(
            nn.Linear(dim, heads, bias = False),
            #RearrangeGates(self.heads)
            Rearrange('b n h -> b h n 1', h = heads)
        ) if gate_values else None

        self.softcap_value = softcap_value

        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            #RearrangeOut(),
            Rearrange('b h n d -> b n (h d)'),
            Linear(dim_inner, dim, bias = False)
        )

    def forward(
        self,
        x,
        attn_mask = None, # for manual masking
        rotary_emb = None,
        cache = None,
        causal = False,
        block_mask = None, # only passed in for flex attention
        return_kv_cache = False,
        return_values = False,
        value_residual = None 
    ):
        device, input_is_cuda, is_decoding_with_cache = x.device, x.is_cuda, exists(cache)

        #should_use_flex_attn = self.use_flex_attn and input_is_cuda

        # handle maybe mask
        # if receiving kv cache, assume decoding and turn off all masking

        if is_decoding_with_cache:
            block_mask = attn_mask = None

        #assert not (exists(block_mask) and exists(attn_mask))
        #assert not (not self.use_flex_attn and exists(block_mask)), 'you cannot pass in the `block_mask` if `use_flex_attn` was not set to be `True`'

        # project to queries, keys, values

        q, k, v = self.to_qkv(x)

        # value residual 
        orig_v = v 

        if exists(value_residual):
            v = 0.5 * (v + value_residual)

        # handle cache being passed in

        if exists(cache):
            cached_k, cached_v = cache
            k = torch.cat((cached_k, k), dim = -2)
            v = torch.cat((cached_v, v), dim = -2)

        # maybe kv cache

        if return_kv_cache:
            kv_cache = torch.stack((k, v))

        # rotary embeddings

        if exists(rotary_emb):
            q, k = tuple(apply_rotary_emb(rotary_emb, t, freqs_seq_dim = -2) for t in (q, k))
        
        
        q = q * self.scale
        #sim = einsum(q, k, 'b h i d, b h j d -> b h i j')
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k)

        sim = softclamp(sim, self.softcap_value)

        mask_value = max_neg_value(sim)

        if causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        if exists(attn_mask):
            sim = einx.where('b i j, b h i j, -> b h i j', attn_mask, sim, mask_value)
            # attn_mask_expanded = attn_mask.unsqueeze(1).expand(-1, sim.shape[1], -1, -1) # should be of shape (b, h, i, j)
            # sim = torch.where(attn_mask_expanded, sim, torch.full_like(sim, mask_value))
            

        attn = sim.softmax(dim = -1)

        attn = self.dropout(attn)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        # out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)

        # maybe gate values

        if exists(self.to_gates):
            out = out * self.to_gates(x).sigmoid()

        # combine heads and out

        out = self.to_out(out)

        if return_values: 
            out = (out, orig_v)

        if not return_kv_cache:
            return out

        return out, kv_cache
    
class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_expansion_factor = 4,
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict(),
        unet_skips = True,
        use_flex_attn = False
    ):
        super().__init__()
        self.use_flex_attn = use_flex_attn

        self.dim = dim
        self.dim_head = dim_head

        self.to_time_cond = nn.Sequential(
            RandomFourierEmbed(dim),
            Linear(dim + 1, dim * 4),
            nn.SiLU()
        )

        layers = ModuleList([])

        for ind in range(depth):
            is_latter_half = ind >= (depth / 2)

            skip_proj = Linear(dim * 2, dim, bias = False) if is_latter_half and unet_skips else None

            attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = dropout, **attn_kwargs)

            ff = FeedForward(dim = dim, expansion_factor = ff_expansion_factor, **ff_kwargs)

            attn = AdaptiveWrapper(attn, dim = dim, dim_cond = dim * 4)
            ff = AdaptiveWrapper(ff, dim = dim, dim_cond = dim * 4)

            layers.append(ModuleList([skip_proj, attn, ff]))

        self.layers = layers
        self.norm = RMSNorm(dim)

    def forward(
        self,
        x,
        times = None,
        attn_mask = None,
        modality_positions = None,
        is_any_modality = None,
        rotary_emb = None,
        cache = None,
        decode_length = None,
        modality_only = False,
        causal_mask = False,
        return_kv_cache = False
    ):
        batch, seq_len, device, input_is_cuda = x.shape[0], x.shape[-2], x.device, x.is_cuda

        is_decoding_with_cache = exists(cache)
        needs_masking = not is_decoding_with_cache

        should_use_flex_attn = input_is_cuda and needs_masking and self.use_flex_attn

        assert not (exists(attn_mask) and exists(modality_positions))

        # handle time

        cond = None

        # processes time into embedding 
        if exists(times):
            if times.ndim == 0:
                times = repeat(times, ' -> b', b = batch)

            cond = self.to_time_cond(times)

        # create the specialized mask needed for autoregressive text + bidirectional flow attention

        attn_mask_kwargs = dict()

        # applies causal and modality masking 
        if needs_masking:
            if causal_mask:
                if should_use_flex_attn: # should be false 
                    pass 
                    #block_mask = create_block_mask(causal, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len, device = device)
                    #attn_mask_kwargs.update(block_mask = block_mask)
                else:
                    attn_mask_kwargs.update(causal = True)

            if exists(modality_positions):
                assert not causal_mask

                if should_use_flex_attn: # should be false 
                    pass 
                    #transfusion_mask_fn = transfusion_attn_mask(modality_positions)
                    #block_mask = create_block_mask(transfusion_mask_fn, B = None, H = None, Q_LEN = seq_len, KV_LEN = seq_len, device = device)
                    #attn_mask_kwargs.update(block_mask = block_mask)
                else:
                    attn_mask = naive_attn_mask(seq_len, modality_positions, device = device)
                    attn_mask_kwargs.update(attn_mask = attn_mask)
        
        if not exists(is_any_modality) and exists(modality_positions):
            is_any_modality = modality_positions_to_is_modality_mask(seq_len, modality_positions).any(dim = 1) 
            is_any_modality = reduce(is_any_modality, 'b t n -> b n', 'any')

        # handle kv caching

        if is_decoding_with_cache:
            assert exists(decode_length)

            cache_length = first(cache).shape[-2]

            x = x[..., -decode_length:, :]
            cond = cond[..., -decode_length:, :]

            if is_tensor(is_any_modality):
                is_any_modality = is_any_modality[..., -decode_length:]

        # adaptive layernorm kwargs, which handles text and modality tokens differently

        adaptive_kwargs = dict(
            cond = cond,
            modality_only = modality_only,
            is_any_modality = is_any_modality
        )

        # handle cache

        cache = default(cache, (None,))
        iter_cache = iter(cache)

        # transformer layers as usual, using mask from above

        skips = []
        value_residual = None

        new_cache = []

        depth = len(self.layers)

        for ind, (skip_proj, attn, ff) in enumerate(self.layers):
            layer = ind + 1

            # skip connection

            is_first_half = layer <= (depth // 2)
            is_later_half = not is_first_half

            if is_first_half:
                skips.append(x)

            if is_later_half and exists(skip_proj):
                skip = skips.pop()

                residual = x
                x = torch.cat((x, skip), dim = -1)
                x = skip_proj(x) + residual

            # attention and feedforward
            # x - input embeddings
            # rotary_emb - positional embeddings 
            # is_any_modality - mask for identifying each modality 
            (attn_out, attn_values), kv_cache = attn(
                x,
                rotary_emb = rotary_emb,
                cache = next(iter_cache, None),
                return_kv_cache = True,
                return_values = True,
                value_residual = value_residual,
                **attn_mask_kwargs,
                **adaptive_kwargs
            )

            value_residual = default(value_residual, attn_values)

            new_cache.append(kv_cache)

            x = attn_out + x
            x = ff(x, **adaptive_kwargs) + x

        assert len(skips) == 0

        out = self.norm(x)

        if not return_kv_cache:
            return out

        return out, torch.stack(new_cache)

