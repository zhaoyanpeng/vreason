import numpy as np
import os, sys, time
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer

import torch.nn.functional as F

from . import MetaDecHead, DECODER_HEADS_REGISTRY
from .. import _get_activation_fn, SoftPositionalEncoder

__all__ = ["SlotCNNDecHead"]

@DECODER_HEADS_REGISTRY.register()
class SlotCNNDecHead(MetaDecHead):
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, token_vocab)
        self.I = 3 # 3-channel images
        D = cfg.m_dim
        K = cfg.kernel_size
        activation = cfg.activation

        self.encoder = None
        self.encoder = nn.Sequential(
            nn.ConvTranspose2d(D, D, K, stride=2, padding=2, output_padding=1),
            _get_activation_fn.get(activation, nn.GELU),
            nn.ConvTranspose2d(D, D, K, stride=2, padding=2, output_padding=1),
            _get_activation_fn.get(activation, nn.GELU),
            nn.ConvTranspose2d(D, D, K, stride=2, padding=2, output_padding=1),
            _get_activation_fn.get(activation, nn.GELU),
            nn.ConvTranspose2d(D, D, K, stride=2, padding=2, output_padding=1),
            _get_activation_fn.get(activation, nn.GELU),
            nn.ConvTranspose2d(D, D, K, stride=1, padding=2, output_padding=0),
            _get_activation_fn.get(activation, nn.GELU),
            nn.ConvTranspose2d(D, 4, 3, stride=1, padding=1, output_padding=0),
        )
        self.input_resolution = (
            [cfg.input_resolution] * 2 if isinstance(cfg.input_resolution, int)
            else list(cfg.input_resolution)[:2] 
        )
        self.pos_embed = SoftPositionalEncoder(D, self.input_resolution)
        self.output_resolution = (
            [cfg.output_resolution] * 2 if isinstance(cfg.output_resolution, int)
            else list(cfg.output_resolution)[:2] 
        )
        self._reset_parameters()

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
        
        assert x.dim() == 3, f"expect `x` of shape (B, N, H)"
        B, S, H = x.shape
        x = x.reshape((-1, H)).unsqueeze(1).unsqueeze(1)
        x = x.repeat((1, self.input_resolution[0], self.input_resolution[1], 1))

        x = self.pos_embed(x).permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        x = self.encoder(x)
        if list(x.shape[-2:]) != self.output_resolution: 
            x = x[..., :self.output_resolution[0], :self.output_resolution[1]]
        
        x_out, masks = x.reshape((B, S) + x.shape[1:]).split((self.I, 1), dim=2)
        masks = masks.softmax(1) # (B, S, C, H, W) where C = 3 (self.I) or 1 
        x_avg = (x_out * masks).sum(1)
        return x_avg, masks.squeeze(2), x_out, {}
