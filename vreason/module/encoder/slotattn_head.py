import numpy as np
import os, sys, time
import torch
from typing import Tuple
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch.nn.functional as F

from . import MetaEncHead, ENCODER_HEADS_REGISTRY
from .. import _get_activation_fn, SoftPositionalEncoder, SlotAttnBlock

__all__ = ["SlotCNNEncHead", "SlotAttnEncHead"]

@ENCODER_HEADS_REGISTRY.register()
class SlotCNNEncHead(MetaEncHead):
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, token_vocab)
        I = 3 # 3-channel images
        D = cfg.m_dim
        K = cfg.kernel_size
        activation = cfg.activation

        self.encoder = nn.Sequential(
            nn.Conv2d(I, D, K, padding="same"),
            _get_activation_fn.get(activation, nn.GELU),
            nn.Conv2d(D, D, K, padding="same"),
            _get_activation_fn.get(activation, nn.GELU),
            nn.Conv2d(D, D, K, padding="same"),
            _get_activation_fn.get(activation, nn.GELU),
            nn.Conv2d(D, D, K, padding="same"),
            _get_activation_fn.get(activation, nn.GELU),
        )
        input_resolution = (
            [cfg.input_resolution] * 2 if isinstance(cfg.input_resolution, int)
            else list(cfg.input_resolution)[:2] 
        )
        self.pos_embed = SoftPositionalEncoder(D, input_resolution)
        self.mlp_ln = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(D, D),
            _get_activation_fn.get(activation, nn.GELU),
            nn.Linear(D, D), 
        )
        self._reset_parameters()

    def _reset_parameters(self):
        pass

    def forward(
        self, 
        x: Tensor, *args,
        self_attn_mask: Tensor=None,
        self_key_padding_mask: Tensor=None,
        attn_weight_type: str=None,
        **kwargs
    ):
        if self.encoder is None:
            return x, None, None

        x = self.encoder(x).permute(0, 2, 3, 1) # (B, C, H, W)
        x = self.pos_embed(x).flatten(1, 2) # (B, C)
        x = self.mlp(self.mlp_ln(x))
        return x, None, None, {}

@ENCODER_HEADS_REGISTRY.register()
class SlotAttnEncHead(SlotCNNEncHead):
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, token_vocab, **kwargs)
        self.slotattn = eval(cfg.block)(
            cfg.m_dim, cfg.num_head, cfg.f_dim, cfg.num_slot, cfg.attn_cls_intra,
            dropout=cfg.t_dropout,
            qk_scale=cfg.qk_scale,
            qkv_bias=cfg.qkv_bias,
            activation=cfg.activation,
            epsilon=cfg.epsilon,
            niter=cfg.niter,
        )

    def forward(
        self, 
        x: Tensor, *args,
        self_attn_mask: Tensor=None,
        self_key_padding_mask: Tensor=None,
        attn_weight_type: str=None,
        **kwargs
    ):
        if self.encoder is None:
            return x, None, None

        x = self.encoder(x).permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        x = self.pos_embed(x).flatten(1, 2) # (B, C)
        x = self.mlp(self.mlp_ln(x))

        x, attns = self.slotattn(None, x)
        return x, attns, None, {}
