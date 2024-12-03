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


from dit import (
    TorchTyping,
    Float,
    Int,
    Bool,
    Scalar,
    ModalitySample,
    ModalityTokenTransform,
    RawModalityPositions,
    GetPredFlows,
    LossBreakdown,
    get_tokens_since_rightmost_id,
    MLPAxialPositions,
    RandomFourierEmbed,
    AdaptiveWrapper,
    RMSNorm,
    GEGLU,
    FeedForward,
    RearrangeQKV,
    RearrangeGates,
    RearrangeOut,
    Attention,
    Transformer
)

class AudioFusion(Module):
    def __init__(
            self, 
            transformer, 
            num_text_tokens, 
            # mel_spec_module,
            # mel_spec_kwargs 
            odeint_kwargs = dict(
                atol = 1e-5,
                rtol = 1e-5, 
                method = 'midpoint'
            ), 
            ignore_index = -1,
            text_loss_weight = 1.0, 
            flow_loss_weight = 1.0, 
            modality_default_shape = None
            ):
        super().__init__()
        
        if isinstance(transformer, dict):
            self.transformer = Transformer(**transformer)

        self.dim = self.transformer.dim 

        # entire "sentence" start and end id

        num_text_special_ids = 2

        self.sos_id, self.eos_id = num_text_tokens, (num_text_tokens + 1)

        # modality meta, start and end tokens - termed [mom] [som] [eom] in this repo

        num_modality_special_ids = 2
        som_eom_tensor = torch.arange(num_modality_special_ids) + num_text_tokens + num_text_special_ids # shift to the very end
        som_eom_tensor = rearrange(som_eom_tensor, '(start_end m) -> start_end m', start_end = 2)

        # modality meta, start and end ids

        self.som_ids, self.eom_ids = som_eom_tensor.tolist()
        
        # char tokenizing for modality meta information

        meta_token_offset = num_text_tokens + num_text_special_ids + num_modality_special_ids

        self.modality_default_shape = modality_default_shape

        self.meta_id = meta_token_offset

        self.char_tokenizer = partial(char_tokenize, offset = meta_token_offset + 1)
        self.decode_chars = partial(decode_chars, offset = meta_token_offset + 1)

        num_meta_tokens = 128 + 1

        dim = self.dim 
        # convert to mel_specs embeddings
        #self.mel_specs = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        # self.mel_specs = MelSpec(mel_spec_type="vocos")
        self.mel_dim = 100 # change later 
        self.mel_to_model_projs = nn.Linear(self.mel_dim, dim)

        # convert from model output to mels 
        self.model_to_mel_preds = nn.Linear(dim, self.mel_dim)

        self.rotary_emb = RotaryEmbedding(self.transformer.dim_head)

        # embeddings and un-embeddings

        effective_num_text_tokens = num_text_tokens + num_text_special_ids + num_modality_special_ids + num_meta_tokens

        self.text_embed = nn.Embedding(effective_num_text_tokens, dim)

        self.to_text_logits = Linear(dim, effective_num_text_tokens, bias = False)

        self.odeint_fn = partial(odeint, **odeint_kwargs)

        self.ignore_index = ignore_index 
        self.text_loss_weight = text_loss_weight
        self.flow_loss_weight = flow_loss_weight

        self.to_modality_shape_fn = default_to_modality_shape_fn
        self.fallback_to_default_shape_if_invalid = False 

        # masks everything below num_text_tokens 
        text_only_mask = torch.arange(effective_num_text_tokens) < num_text_tokens
        self.register_buffer('text_only_logits_mask', text_only_mask, persistent = False)

        
    @property
    def device(self):
        return next(self.parameters()).device
    

    @torch.no_grad()
    @eval_decorator
    def sample(
        self,
        prompt = None,
        max_length = 2048,
        text_temperature = 1.5,
        text_min_p = 0.1,
        cache_kv = False,
        fixed_modality_shape: tuple[int, ...] | None = None,
        init_modality_noise: Float['n d'] | None = None,
        modality_steps = 16,
        return_unprocessed_modalities = False,
        vocoder = None
    ):

        device = self.device
        
        cache = {} if cache_kv else None 

        init_text_seq = tensor([self.sos_id], device = device)
        modality_sample = [init_text_seq, *default(prompt, [])]

        # take care of moving to device

        modality_sample = tree_map_tensor(modality_sample, lambda t: t.to(device))
        modality_sample = tree_map_tensor(modality_sample, lambda t: rearrange(t, '-> 1') if t.ndim == 0 else t)

        *_, last_modality_sample = modality_sample # last element 
        assert last_modality_sample.dtype in (torch.int, torch.long), 'prompt must be text tokens'

        curr_length = 0
        curr_modality_id = None
        modality_shape = None
        num_past_modalities = 0  # starts off with no modalities in output

        text_is_greedy = text_temperature == 0.
        is_decoding_text = True  # starts off with text decoding, and alternates with modalities depending on [som] tokens detected

        def maybe_transition_to_modality_decoding(seq):
            nonlocal modality_shape
            nonlocal is_decoding_text
            nonlocal curr_modality_id

            sampled_token_id = seq[-1]

            if sampled_token_id not in self.som_ids:
                return

            curr_modality_id = self.som_ids.index(sampled_token_id)

            if exists(fixed_modality_shape):
                modality_shape = fixed_modality_shape

            # get the tokens after the modality meta id

            maybe_meta_tensor = get_tokens_since_rightmost_id(seq, self.meta_id)

            default_shape = self.modality_default_shape
            maybe_modality_num_dim = 2

            meta_str_to_modality_shape = self.to_modality_shape_fn

            if maybe_meta_tensor.numel() > 0:
                meta_tensor = maybe_meta_tensor[:-1]
                meta_str = self.decode_chars(meta_tensor)

                if not meta_str.isdigit() or int(meta_str) <= 0:

                    assert exists(default_shape), 'invalid modality meta information detected, please set `modality_default_shape` in order to properly fallback'
                    modality_shape = default_shape
                else:
                    modality_shape = meta_str_to_modality_shape(meta_str)

            modality_shape = default(modality_shape, default_shape)

            if self.fallback_to_default_shape_if_invalid:

                if exists(maybe_modality_num_dim) and len(modality_shape) != maybe_modality_num_dim:
                    logger.warning(f'invalid modality shape {modality_shape} for modality {curr_modality_id}. falling back to default shape {default_shape}')
                    modality_shape = default_shape

            assert exists(modality_shape), f'language model did not produce a proper modality shape for modality type {curr_modality_id} - please set a fallback shape with `modality_default_shape`'
            assert not exists(maybe_modality_num_dim) or maybe_modality_num_dim == len(modality_shape), f'expected modality type {curr_modality_id} to have {maybe_modality_num_dim} dimensions but language model produced a shape of {modality_shape}'

            is_decoding_text = False

        # determine if to transition from start

        maybe_transition_to_modality_decoding(last_modality_sample)

        # cache = None
        # cache = {} if cache_kv else None

        with tqdm(total = max_length) as pbar:

            while curr_length <= max_length:

                if is_decoding_text:
                    pbar.set_description('decoding text')

                    # if cache != None and torch.is_tensor(cache):
                    if cache is not None and torch.is_tensor(cache):
                        cache = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cache.items()}
                        # cache = tree_map_tensor(cache, lambda t: t.to(device) if torch.is_tensor(t) else t)

                    *_, seq = modality_sample

                    logits, new_kv_cache = self.forward(
                        [modality_sample],
                        return_loss = False,
                        cache = cache,
                        decode_length = 1,
                        decoding_text_or_modality = 'text',
                        return_kv_cache = True
                    )

                    logits = logits[0][-1].to(device)
     

                    if text_is_greedy:
                        sampled = logits.argmax(dim = -1, keepdim = True)
                    else:
                        logits = min_p_filter(logits, min_p = text_min_p)

                        probs = (logits / text_temperature).softmax(dim = -1)
                        sampled = torch.multinomial(probs, 1)

                    seq = torch.cat((seq, sampled), dim = -1)
                    modality_sample[-1] = seq

                    pbar.update(1)
                    curr_length += 1

                    if cache_kv:
                        cache = new_kv_cache

                    sampled_token_id = sampled.item()

                    if sampled_token_id == self.eos_id:
                        logger.info(f'detecting an end of string token [{self.eos_id}], terminating sampling early')
                        break

                    maybe_transition_to_modality_decoding(seq)

                else:
                    pbar.set_description(f'decoding audio')
                    modality_length = modality_shape[0]

                    if exists(init_modality_noise): 
                        noise = init_modality_noise[:modality_shape[0], :self.mel_dim] # should be of size seq_len 100 
                    else: 
                        noise = torch.randn((modality_shape[0], self.mel_dim), device=device)

                    assert noise.shape == modality_shape

                    new_kv_cache = None 

                    def ode_step_fn(step_times, denoised):
                        #nonlocal new_kv_cache 
                        nonlocal cache 

                        step_times = rearrange(step_times, ' -> 1 1') # batch size of 1
                        step_times = F.pad(step_times, (num_past_modalities, 0), value = 1.) # past decoded modalities receive a time conditioning of 1.

                        denoised = denoised.reshape(modality_shape) # seq_len, mel_dim 
                        
                        # if cache != None and torch.is_tensor(cache):
                            # cache = cache.to(device)
                        # if cache is not None and isinstance(cache, dict):
                        #     cache = {k: v.to(device) if torch.is_tensor(v) else v for k, v in cache.items()}


                        embeds, new_kv_cache = self.forward(
                            [[*modality_sample, (curr_modality_id, denoised)]],
                            times = step_times, 
                            return_embed = True, 
                            cache = cache, 
                            decode_length = modality_shape[0], 
                            return_kv_cache = True, 
                            decoding_text_or_modality = 'modality'
                        )

                        to_flow_pred = self.model_to_mel_preds
                        flow = to_flow_pred(embeds)
                        
                        return flow[0, -modality_shape[0]]
                        #return flow[0, -modality_length:]
                    
                    times = torch.linspace(0, 1, modality_steps, device=device)
                    trajectory = self.odeint_fn(ode_step_fn, noise, times)

                    sampled_modality = trajectory[-1]

                    sampled_modality = sampled_modality.reshape(modality_shape)
                    modality_sample.append(sampled_modality)

                    eom_id = self.eom_ids[curr_modality_id]
                    modality_sample.append(tensor([eom_id], device=device))

                    if cache_kv: 
                        cache = new_kv_cache 

                    pbar.update(modality_length)
                    curr_length += modality_length 

                    num_past_modalities += 1 
                    curr_modality_id = None 
                    

        if return_unprocessed_modalities:
            return modality_sample 

        processed_modality_sample = []
        for sample in modality_sample:
            if not isinstance(sample, tuple):
                processed_modality_sample.append(sample)
                continue

            modality_id, modality = sample
            processed_modality_sample.append((modality_id, modality))

        return processed_modality_sample

    def forward_text(
        self,
        text: Int['b n'],
        return_loss = True,
        return_embed = False,
        cache: Tensor | None = None,
        return_kv_cache = False
    ) -> (
        Float[''] |
        Float['b n d'] |
        tuple[Float['b n d'], list[Float['...']]]
    ):

        device = self.device
        text = text.to(device)

        if return_loss:
            text, labels = text[:, :-1], text[:, 1:]

        # embed text

        text = text.masked_fill(text == -1, 0)
        tokens = self.text_embed(text)

        # rotary

        seq_len = tokens.shape[-2]
        pos = torch.arange(seq_len, device = device)

        rotary_emb = self.rotary_emb(pos)

        # attention

        embed, kv_cache = self.transformer(
            tokens,
            rotary_emb = rotary_emb,
            causal_mask = True,
            cache = cache,
            return_kv_cache = True
        )

        # text unembedding

        logits = self.to_text_logits(embed)

        if not return_loss:
            if not return_kv_cache:
                return logits

            return logits, kv_cache

        logits = logits.masked_fill(~self.text_only_logits_mask, max_neg_value(logits))

        loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss
    
    @torch.no_grad()
    @eval_decorator
    def generate_text_only(
        self,
        prompt: Int['b n'],
        seq_len: int,
        temperature = 1.5,
        min_p = 0.1,
    ) -> Int['b no']:

        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)

        for _ in tqdm(range(sample_num_times)):
            logits = self.forward_text(out, return_loss = False)
            logits = logits[:, -1]

            logits = min_p_filter(logits, min_p = min_p)

            logits.masked_fill_(~self.text_only_logits_mask, max_neg_value(logits))

            sample = gumbel_sample(logits, temperature = temperature, dim = -1)

            out = torch.cat((out, sample), dim = -1)

        return out[..., prompt_seq_len:]
    
    def forward_modality(
        self, 
        modalities, # float (b ...)
        times = None, 
        velocity_consistency_ema_model = None, 
        velocity_consistency_delta_time = 1e-5, 
        return_loss = True, 
        return_loss_breakdown = False
    ): 
        requires_velocity_consistency = exists(velocity_consistency_ema_model)
        modalities = modalities.to(self.device)

        mel_to_model_fn = self.mel_to_model_projs
        model_to_flow_pred_fn = self.model_to_mel_preds
        

        # modalities should be tensor of shape (b seq_len trasnformer_dim)
        #modalities = mel_to_model_fn(modalities) 
        tokens = modalities 
        shape = modalities.shape 
        batch, device = tokens.shape[0], tokens.device

        # times 
        if not exists(times):
            times = torch.rand((batch), device=device)

        if return_loss: 
            # add velocity_consistency later 
            padded_times = append_dims(times, tokens.ndim - 1)
            noise = torch.rand_like(tokens)

            noised_tokens = padded_times * tokens + (1. - padded_times) * noise 

            flow = tokens - noise 
        else: 
            noised_tokens = tokens 
        
        # from noise mel to model projection mel (b seq_len mel_dim) -> (b seq_len emb_dim)
        noised_tokens = mel_to_model_fn(noised_tokens)

        seq_len = noised_tokens.shape[-2]
        pos = torch.arange(seq_len, device=device)

        rotary_emb = self.rotary_emb(pos)

        embed = self.transformer(noised_tokens, times=times, rotary_emb=rotary_emb, modality_only=True)

        # from model project to mel -> (b seq_len embed_dim) -> (b seq_len emb_dim)
        pred_flow = model_to_flow_pred_fn(embed)
        
        #assert pred_flow.shape == flow.shape, f'Shape mismatch: pred_flow {pred_flow.shape}, flow {flow.shape}'

        if not return_loss: 
            return pred_flow 
        
        # flow loss 
        #flow = rearrange(flow, 'b ... d -> b (...) d')

        flow_loss = F.mse_loss(pred_flow, flow)
        
        # add velocity loss later 

        return flow_loss 
    

    @torch.no_grad()
    @eval_decorator
    def generate_modality_only(
        self,
        batch_size: int = 1,
        modality_type: int | None = None,
        fixed_modality_shape: tuple[int, ...] | None = None,
        modality_steps = 16,
        return_unprocessed_modalities = False
    ) -> Tensor:
        
        device = self.device

        mel_dim = 100 
        to_flow_pred = self.model_to_mel_preds

        # set some default modality shape 
        default_modality_shape = (315, 100) 
        modality_shape = default(fixed_modality_shape, default_modality_shape)

        noise = torch.randn((batch_size, modality_shape[0], modality_shape[1]), device=device) # noise of shape seq_len emb_dim 

        def ode_step_fn(step_times, denoised):
            step_times = repeat(step_times, ' -> b', b = batch_size)
            #denoised = denoised.reshape(batch_size, modality_shape[0], self.dim)

            flow = self.forward_modality(denoised, times=step_times, return_loss=False)

            return flow 
        
        times = torch.linspace(0., 1., modality_steps, device=device)
        trajectory = self.odeint_fn(ode_step_fn, noise, times)

        sampled_modality = trajectory[-1] # has shape (b, seq_len, mel_dim)
        # sampled_modality = sampled_modality.reshape(batch_size, modality_shape[0], modality_shape[1])

        # sampled_modality = to_flow_pred(sampled_modality) 

        return sampled_modality # should be spectrogram (b n d)
    
    def forward(
       self,
       modalities,
       times: Float['b m'] | None = None,
       modality_length_to_times_fn: Callable[[Int['b m']], Float['b m']] | None = None, # allows a researcher to customize the times (noise level) based on the modality lengths in a given sample
       modality_type: int | None = None,
       cache: Tensor | None = None,
       decode_length: int | None = None,
       decoding_text_or_modality: Literal['text', 'modality'] | None = None,
       velocity_consistency_ema_model: EMA | None = None,
       velocity_consistency_delta_time = 1e-3,
       return_only_pred_flows = False,
       return_loss = True,
       return_breakdown = False,
       return_embed = False,
       return_kv_cache = False,
   ): 

        is_decoding = exists(decoding_text_or_modality)
      
        is_text_only = is_tensor(modalities) and modalities.dtype in (torch.int, torch.long)
        is_modality_only = is_tensor(modalities) and modalities.dtype == torch.float


        mel_to_model_fn = self.mel_to_model_projs
        model_to_flow_pred_fn = self.model_to_mel_preds


        # add ema later


        return_loss &= not (return_embed or is_decoding)

        if is_text_only:
            forward_text_kwargs = dict(
                return_loss = return_loss,
                return_embed = return_embed,
                cache = cache,
                return_kv_cache = return_kv_cache
            )


            return self.forward_text(modalities, **forward_text_kwargs)
      
        if is_modality_only:
            assert return_loss


            forward_modality_kwargs = dict(
                modality_type = modality_type,
                velocity_consistency_ema_model = velocity_consistency_ema_model
            )


            return self.forward_modality(modalities, **forward_modality_kwargs)
      
        device = self.device
        tensor_ = partial(tensor, device = device)

        if return_loss:
            modalities = modalities.copy()


            for i, modality in enumerate(modalities):
                modalities[i] = [
                    tensor_([self.sos_id]),
                    *modality,
                    tensor_([self.eos_id])
                ]
        
        modality_positions, modality_tokens = [], []


        text = []


        for batch_modalities in modalities:
            batch_modality_positions = []
            batch_modality_tokens = []
            batch_modality_pos_emb = []
            batch_text = []
            offset = 0


            for modality in batch_modalities:
                # if non-text modality detected and not given as a tuple
                # cast to (int, Tensor) where int is defaulted to type 0 (convenience for one modality)


                if is_tensor(modality) and modality.dtype == torch.float:
                    modality = (0, modality)


                is_text = not isinstance(modality, tuple)
                is_modality = not is_text


                if is_text:
                    modality_tensor = modality
                else:
                    modality_type, modality_tensor = modality


                modality_tensor = modality_tensor.to(device)


                # if is_modality:
                #     if not is_decoding:
                #         modality_tensor = mel_to_model_fn(modality_tensor)
                
                # auto ward against scalars (lone start end tokens)
                if modality_tensor.dtype in (torch.int, torch.long) and modality_tensor.ndim == 0:
                    modality_tensor = rearrange(modality_tensor, '-> 1')


                # handle text
                if is_text:
                    assert modality_tensor.ndim == 1

                    text_length = modality_tensor.shape[0]

                    batch_text.append(modality_tensor)
                    offset += text_length

                    continue

                    # pos embeds ?


                # otherwise handle modality
                # modality_shape_tuple = tuple(modality_tensor.shape[:-1])
                # modality_length = math.prod(modality_shape_tuple)


                # text_tensor = torch.full((modality_length,), -1, device=device) # text is all -1 here, so text labels are not learned on


                modality_shape_tuple = tuple(modality_tensor.shape)
                # modality_length = math.prod(modality_shape_tuple)
                modality_length = modality_shape_tuple[0] # sequence length
                text_tensor = torch.full((modality_shape_tuple[0], ), -1, device=device)

                # only add modality meta information when not returning embedding, which only occurs when sampling modality


                succeed_modality_tokens = precede_modality_tokens = 0


                if not return_embed:
                    # add the [som] and [eom] tokens for the modality type
                    # som_id, eom_id = self.som_ids[modality_type], self.eom_ids[modality_type]
                    som_id, eom_id = self.som_ids[0], self.eom_ids[0]

                    # start by just storing token length of modality
                    modality_shape_str = join([*map(str, modality_shape_tuple)], ',')
                    modality_meta_info = self.char_tokenizer(modality_shape_str, device = device)

                    precede_modality_tokens = len(modality_meta_info) + 2
                    succeed_modality_tokens = 1

                    text_tensor = torch.cat((
                        tensor_([self.meta_id]),
                        modality_meta_info,
                        tensor_([som_id]),
                        text_tensor,
                        tensor_([eom_id])
                    ))

                batch_modality_positions.append((modality_type, offset + precede_modality_tokens, modality_length)) # offset + preceding meta tag length (which includes the modality start token)

                offset += modality_length + precede_modality_tokens + succeed_modality_tokens # +2 due to [som] and [eom] - then account for meta start id and modality shape information (or eventually any meta information about modality)

                #modality_tensor = rearrange(modality_tensor, '... d -> (...) d')


                batch_modality_tokens.append(modality_tensor)
                batch_text.append(text_tensor)

            text.append(torch.cat(batch_text))

            modality_tokens.append(batch_modality_tokens)  
            modality_positions.append(batch_modality_positions)

        if return_loss:
            total_tokens = sum([t.numel() for t in text])
        
        text = pad_sequence(text, padding_value=-1)
    
        text = text.permute(1, 0)

        # split for next token prediction
        if return_loss:
            text, text_labels = text[:, :-1], text[:, 1:]

        # derive modality mask for flow
        batch, seq_len, device = *text.shape, text.device


        assert len(modality_positions) == batch


        modality_positions = modality_positions_to_tensor(modality_positions)
        modality_positions = modality_positions.permute(1, 0, 2)


        if isinstance(modality_positions, list):
            modality_positions = modality_positions_to_tensor(modality_positions, device=device)


        if modality_positions.shape[-1] == 2: # (b m 2) -> (b m 3)
            modality_positions = F.pad(modality_positions, (1, 0), value=0)


        # use dummy padding if empty
        if modality_positions.numel() == 0:
            modality_positions = F.pad(modality_positions, (0, 0, 0, 1))


        # embed the list of modality tokens into a sequence of (b n d) at right offsets and lengths
        if is_tensor(modality_tokens):
            modality_tokens = [modality_tokens]
        
        # embed the modality tokens into one Tensor if not given as one
      
        if isinstance(modality_tokens, list) and isinstance(first(modality_tokens), list): # detect list[list[tensor]]
            #modality_tokens = [embed_modality_tokens(seq_len, dim_latent, modality_tokens, modality_positions, modality_id) for modality_id, dim_latent in enumerate(self.dim_latents)]   
            modality_tokens = [
                embed_modality_tokens(seq_len=seq_len, dim=self.mel_dim, modality_tokens=modality_tokens, modalities=modality_positions, modality_id=0, channel_first=False)
            ]

        # sort the modalities tensor and sanitize, readying for noising of modalities
        modality_positions, sorted_indices = order_modality_positions_by_seq_offset(modality_positions)
        is_modalities = modality_positions_to_is_modality_mask(seq_len, modality_positions, num_modalities = 1, device = device)

        is_any_modality = reduce(is_modalities, 'b t m n -> b n', 'any')
   
        # embed text
        text = text.masked_fill(text == -1, 0)
   
        text_tokens = self.text_embed(text)

        # noise the modality tokens

        if not exists(times):
            modality_length_to_times_fn = default(modality_length_to_times_fn, default_modality_length_to_time_fn)
           
            if exists(modality_length_to_times_fn):
                times = modality_length_to_times_fn(modality_positions[..., -1]).to(device)
  

        is_modalities = is_modalities.to(device)
        times_per_token = einsum(is_modalities.float(), times, 'b t m n, b m -> b t n').to(device)
   

        if return_loss:
            noised_modality_tokens = []
            flows = []
          

            for modality_id, one_modality_tokens in enumerate(modality_tokens):
             
                #one_modality_tokens = one_modality_tokens.to(device)
                noise = torch.randn_like(one_modality_tokens, device=device)
               
                one_times = times_per_token[:, modality_id].to(device)
             
                padded_times = rearrange(one_times, 'b n -> b n 1').to(device)

                one_modality_tokens = one_modality_tokens.to(device)           
                one_noised_modality_tokens = one_modality_tokens * padded_times + noise * (1. - padded_times)

                # the flow is the (data - noise)
                one_flow = one_modality_tokens - noise
        
                # append

                flows.append(one_flow)
                noised_modality_tokens.append(one_noised_modality_tokens)

            modality_tokens = noised_modality_tokens


        # project the modality tokens to model


        #modality_tokens = [fn(one_modality_tokens) for fn, one_modality_tokens in zip(self.latent_to_model_projs, modality_tokens)]
        modality_tokens = [self.mel_to_model_projs(one_modality_tokens.to(device)) for one_modality_tokens in modality_tokens]
        modality_tokens = sum(modality_tokens)


        # intersperse the modalities with the text for the joint transformer + flow system
        is_any_modality = is_any_modality.to(device)
  
       
        tokens = einx.where('b n, b n d, b n d', is_any_modality, modality_tokens, text_tokens).to(device)
        
        # derive rotary positions

        rotary_positions = derive_rotary_positions_from_modality_positions(seq_len, modality_positions)
        rotary_positions = rotary_positions.to(device)

        rotary_emb = self.rotary_emb(rotary_positions).to(device)
        
        rotary_emb = rearrange(rotary_emb, 'b n d -> b 1 n d')
        rotary_emb = rotary_emb.to(device)

        # take care of cache

        is_any_modality_when_decoding = None

        if exists(cache):
            assert exists(decode_length), '`decode_length` must be passed in on forward for modality sampling. think of a cleaner way on some future date'
            assert exists(decoding_text_or_modality)

            if decoding_text_or_modality == 'text':
                decode_length = 1

            is_any_modality_when_decoding = decoding_text_or_modality == 'modality'
            modality_positions = None
            cache = tree_map_tensor(cache, lambda t: t.to(device) if torch.is_tensor(t) else t)


        # times
        times_per_token = times_per_token.to(device)

        times_cond = reduce(times_per_token, 'b t n -> b n', 'sum').to(device)

        modality_positions = modality_positions.to(device)

        # attention
        embed, kv_cache = self.transformer(
            tokens,
            times = times_cond,
            rotary_emb = rotary_emb,
            modality_positions = modality_positions,
            is_any_modality = is_any_modality_when_decoding,
            cache = cache,
            decode_length = decode_length,
            return_kv_cache = True
        )

        # early return for embedding for decoding modality

        if return_embed:
            if not return_kv_cache:
                return embed


            return embed, kv_cache

        # text unembedding

        text_logits = self.to_text_logits(embed)

        if not return_loss:
            if not return_kv_cache:
                return text_logits


            return text_logits, kv_cache
        
        # flow loss

        #pred_flows = [fn(embed) for fn in self.model_to_latent_preds]
        pred_flows = [model_to_flow_pred_fn(embed)]

        # text autoregressive loss
        text_labels = text_labels.masked_fill(is_any_modality, self.ignore_index)


        text_loss = F.cross_entropy(
            rearrange(text_logits, 'b n l -> b l n'),
            text_labels,
            ignore_index = self.ignore_index
        )

        text_loss_weight = (text_labels != self.ignore_index).sum() / total_tokens

        # calculate flow losses

        flow_losses = []

        modality_loss_weights = []


        for flow, pred_flow, is_one_modality in zip(flows, pred_flows, is_modalities.unbind(dim = 1)):
            flow_loss = F.mse_loss(
                pred_flow,
                flow,
                reduction = 'none'
            )

            is_one_modality = reduce(is_one_modality, 'b m n -> b n', 'any')

            flow_loss = flow_loss[is_one_modality].mean()

            modality_loss_weight = is_one_modality.sum() / total_tokens

            modality_loss_weights.append(modality_loss_weight)


            flow_losses.append(flow_loss)


        modality_loss_weights = stack(modality_loss_weights)

        # only the token positions that are not modalities have autoregressive loss
       
        modality_loss_weights = modality_loss_weights.to(device)

        total_loss = (
            text_loss * text_loss_weight * self.text_loss_weight +
            (stack(flow_losses) * modality_loss_weights).sum() * self.flow_loss_weight
        )

        if not return_breakdown:
            return total_loss


        return total_loss, LossBreakdown(total_loss, text_loss, flow_losses)
