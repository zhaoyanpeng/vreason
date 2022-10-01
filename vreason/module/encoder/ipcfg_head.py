import numpy as np
import os, sys, time
import torch
from typing import Tuple
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import torch.nn.functional as F

from . import MetaEncHead, ENCODER_HEADS_REGISTRY, SlotCNNEncHead
from .. import _get_activation_fn, SlotAttnBlock

if True: 
    from .. import TransformerEncoder, TransformerEncoderLayer
else:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer

__all__ = ["BiLSTMLatentEncHead", "SlotAttnLatentEncHead", "BiAttnLatentEncHead"]


class BiLSTMBaseEncHead(MetaEncHead):
    def __init__(self, cfg, txt_token_vocab, vis_token_vocab=None, **kwargs):
        super().__init__(cfg, None)
        self.vis_token_vocab = vis_token_vocab
        self.vis_token_embed = nn.Embedding(len(self.vis_token_vocab), cfg.w_dim)
        self.encoder = nn.LSTM(
            cfg.w_dim, cfg.m_dim,
            bidirectional=True, num_layers=cfg.num_layer, batch_first=True
        )

        modules = [nn.Linear(cfg.m_dim * 2, cfg.z_dim * 2)]
        self.predictor = nn.Sequential(*modules)

        self.z_dim = cfg.z_dim
        self.pooling = cfg.pooling
        self._reset_parameters()

@ENCODER_HEADS_REGISTRY.register()
class BiLSTMLatentEncHead(BiLSTMBaseEncHead):
    def forward(
        self, 
        x: Tensor, *args, # tokens
        self_attn_mask: Tensor=None,
        self_key_padding_mask: Tensor=None,
        attn_weight_type: str=None,
        lengths: Tensor=None,
        sortedx: bool=True,
        **kwargs
    ):
        B, L = x.shape[:2]
        # just in case x is yet to be sorted 
        if lengths is None:
            sortedx = False
            lengths = torch.tensor([L] * B).to(x)

        x = self.vis_token_embed(x)
        x = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=(not sortedx)
        )
        x, _ = self.encoder(x)

        if self.pooling == "max":
            padding_value = float("-inf")
            x, _ = pad_packed_sequence(
                x, batch_first=True, padding_value=padding_value
            )
            x, attn_weights = x.max(1)
        else: # mean
            padding_value = 0.
            x, _ = pad_packed_sequence(
                x, batch_first=True, padding_value=padding_value
            )
            x = x.sum(1) / lengths.unsqueze(-1)
            attn_weights = None

        z = self.predictor(x)
        mean = z[:, : self.z_dim]
        lvar = z[:, self.z_dim :]
        return mean, lvar, None, {} 

@ENCODER_HEADS_REGISTRY.register()
class SlotAttnLatentEncHead(SlotCNNEncHead):
    def __init__(self, cfg, txt_token_vocab, vis_token_vocab=None, **kwargs):
        super().__init__(cfg, None)
        self.slotattn = None
        if cfg.num_slot > 0:
            self.slotattn = eval(cfg.block)(
                cfg.m_dim, cfg.num_head, cfg.f_dim, cfg.num_slot, cfg.attn_cls_intra,
                dropout=cfg.t_dropout,
                qk_scale=cfg.qk_scale,
                qkv_bias=cfg.qkv_bias,
                activation=cfg.activation,
                epsilon=cfg.epsilon,
                niter=cfg.niter,
            )
        self.vis_token_vocab = vis_token_vocab
        self.vis_token_embed = nn.Embedding(len(self.vis_token_vocab), cfg.w_dim)
            
        modules = [nn.Linear(cfg.m_dim, cfg.z_dim * 2)]
        self.predictor = nn.Sequential(*modules)

        self.z_dim = cfg.z_dim
        self.pooling = cfg.pooling
        self._reset_parameters()

    def forward(
        self, 
        x: Tensor, *args,
        self_attn_mask: Tensor=None,
        self_key_padding_mask: Tensor=None,
        attn_weight_type: str=None,
        **kwargs
    ):
        B, L = x.shape[:2]
        x = x.reshape((B,) + (int(L ** 0.5),) * 2) 
        x = self.vis_token_embed(x).permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        x = self.encoder(x).permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        x = self.pos_embed(x).flatten(1, 2) # (B, C)
        x = self.mlp(self.mlp_ln(x))
        
        if self.slotattn is not None:
            x, *_ = self.slotattn(None, x)

        if self.pooling == "max":
            x, attn_weights = x.max(1)
        else: # mean
            x = x.mean(1)
            attn_weights = None

        z = self.predictor(x)
        mean = z[:, : self.z_dim]
        lvar = z[:, self.z_dim :]
        return mean, lvar, None, {} 

class BiAttnBaseEncHead(MetaEncHead):
    def __init__(self, cfg, txt_token_vocab, vis_token_vocab=None, **kwargs):
        super().__init__(cfg, None)
        self.vis_token_vocab = vis_token_vocab
        self.vis_token_embed = nn.Embedding(len(self.vis_token_vocab), cfg.w_dim)
        layer_fn = TransformerEncoderLayer(
            cfg.m_dim, cfg.num_head, cfg.f_dim, cfg.t_dropout, 
            activation=cfg.activation, norm_first=cfg.norm_first
        )
        iln = oln = lambda x: x
        if cfg.norm_first:
            oln = nn.LayerNorm(cfg.m_dim)
        else:
            iln = nn.LayerNorm(cfg.m_dim)
        self.encoder = TransformerEncoder(layer_fn, cfg.num_layer, iln=iln, oln=oln)

        modules = [nn.Linear(cfg.m_dim, cfg.z_dim * 2)]
        self.predictor = nn.Sequential(*modules)

        self.z_dim = cfg.z_dim
        self.pooling = cfg.pooling
        self._reset_parameters()

@ENCODER_HEADS_REGISTRY.register()
class BiAttnLatentEncHead(BiAttnBaseEncHead):
    def forward(
        self, 
        x: Tensor, *args,
        self_attn_mask: Tensor=None,
        self_key_padding_mask: Tensor=None,
        attn_weight_type: str=None,
        **kwargs
    ):
        if self_key_padding_mask is None:
            self_key_padding_mask = torch.zeros_like(x).bool() 

        x = self.vis_token_embed(x)

        x = x.transpose(0, 1)

        x = self.encoder(
            x, src_key_padding_mask=self_key_padding_mask,
        ) 
        if isinstance(x, (tuple, list)):
            x, *_ = x

        x = x.transpose(0, 1)

        if self.pooling == "max":
            padding_value = float("-inf")
            x[self_key_padding_mask] = padding_value
            x, attn_weights = x.max(1)
        else: # mean
            padding_value = 0. 
            x[self_key_padding_mask] = padding_value
            x = x.sum(1) / (~self_key_padding_mask).sum(-1, keepdim=True)
            attn_weights = None

        z = self.predictor(x)
        mean = z[:, : self.z_dim]
        lvar = z[:, self.z_dim :]
        return mean, lvar, None, {} 
