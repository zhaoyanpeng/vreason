import numpy as np
import os, sys, time
import torch
from typing import Tuple
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import torch.nn.functional as F

from . import MetaEncHead, ENCODER_HEADS_REGISTRY
from .. import _get_activation_fn, Hard1DEmbedder, Hard2DEmbedder, Soft2DEmbedder

if True: 
    from .. import TransformerEncoder, TransformerEncoderLayer
else:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer

__all__ = ["DalleTokenEncHead"]


@ENCODER_HEADS_REGISTRY.register()
class DalleTokenEncHead(MetaEncHead):
    def __init__(self, cfg, txt_token_vocab, vis_token_vocab=None, **kwargs):
        super().__init__(cfg, None)
        self._emb_size = cfg.embed_dim
        self.txt_offset = cfg.txt_offset
        self.vis_offset = cfg.vis_offset

        self.len_txt_seq = cfg.len_txt_seq
        self.len_vis_seq = cfg.len_vis_seq
        H = W = int(self.len_vis_seq ** 0.5)

        self.txt_token_vocab = txt_token_vocab
        self.vis_token_vocab = vis_token_vocab

        self.num_txt_token = len(txt_token_vocab)
        self.num_vis_token = len(vis_token_vocab)

        self.txt_embed = eval(cfg.txt_embed)(
            self.num_txt_token, self._emb_size,
            (self.len_txt_seq,), offset=self.txt_offset,
            tok_padding_idx=txt_token_vocab.PAD_IDX, use_pos_padding=False,
        )
        self.vis_embed = eval(cfg.vis_embed)(
            self.num_vis_token, self._emb_size, (H,) * 2,
            offset=self.vis_offset, tok_padding_idx=None, use_pos_padding=False,
        )

    def _reset_parameters(self):
        pass

    @property
    def emb_size(self):
        return self._emb_size

    @property
    def is_bart(self):
        return self.vis_offset == 1

    def _encode_positions(self, t=None, v=None):
        if t is not None:
            t = self.txt_embed(t)
        if v is not None:
            v = F.pad(v, (1, 0), value=self.vis_token_vocab.BOS_IDX) if self.is_bart else v
            v = self.vis_embed(v)
        return t, v

    def forward(self, t_seq, v_seq=None, **kwargs):
        t, v = self._encode_positions(t=t_seq, v=v_seq) 
        return t, v, {} 

@ENCODER_HEADS_REGISTRY.register()
class DalleBartEncHead(MetaEncHead):
    """ Standard Transformer Encoder.
    """
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, token_vocab)
        self.encoder = None
        if cfg.num_layer > 0:
            layer_fn = TransformerEncoderLayer(
                cfg.m_dim, cfg.num_head, cfg.f_dim, cfg.t_dropout, 
                activation=cfg.activation, norm_first=cfg.norm_first
            )
            self.encoder = TransformerEncoder(layer_fn, cfg.num_layer)

        self.stability = cfg.stability

        self._reset_parameters()

    def forward(
        self, 
        x: Tensor, *args,
        self_attn_mask: Tensor=None,
        self_key_padding_mask: Tensor=None,
        attn_weight_type: str=None,
        use_cache: bool=False,
        cache: Tensor=None,
        **kwargs
    ):
        if cache is not None and use_cache:
            return cache, None, cache, {}

        if self.encoder is None:
            return x, None, None, {}

        if self.stability > 0.:
            x = x * self.stability + x.detach() * (1 - self.stability)
        
        x = x.transpose(0, 1)

        x = self.encoder(
            x, src_key_padding_mask=self_key_padding_mask,
        ) 
        if isinstance(x, (tuple, list)):
            x, *_ = x

        x = x.transpose(0, 1)

        cache = x if use_cache else None

        return x, None, cache, {} 
