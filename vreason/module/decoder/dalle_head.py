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

__all__ = ["DalleGPTDecHead"]

@DECODER_HEADS_REGISTRY.register()
class DalleGPTDecHead(MetaDecHead):
    def __init__(self, cfg, txt_token_vocab, vis_token_vocab=None, **kwargs):
        super().__init__(cfg, None)
        self.encoder = None
        if cfg.num_layer > 0:
            layer_fn = TransformerEncoderLayer(
                cfg.m_dim, cfg.num_head, cfg.f_dim, cfg.t_dropout, 
                activation=cfg.activation, norm_first=cfg.norm_first
            )
            self.encoder = TransformerEncoder(layer_fn, cfg.num_layer)

            self_attn_mask = (torch.triu(
                torch.ones(cfg.block_size, cfg.block_size, dtype=torch.uint8), 
            diagonal=1) == 1)

            # flexible context length
            contx_size = cfg.contx_size if cfg.contx_size > 0 else 0
            self_attn_mask[:contx_size, :contx_size] = False
            self.contx_size = contx_size

            self.register_buffer("self_attn_mask", self_attn_mask) 

        self.num_head = cfg.num_head

        self.len_txt_seq = cfg.len_txt_seq
        self.len_vis_seq = cfg.len_vis_seq

        self.txt_token_vocab = txt_token_vocab
        self.vis_token_vocab = vis_token_vocab

        self.num_txt_token = len(txt_token_vocab)
        self.num_vis_token = len(vis_token_vocab)

        self.ln_output = nn.LayerNorm(cfg.m_dim) if cfg.norm_first else nn.Identity()
        self.t_predictor = nn.Linear(cfg.m_dim, self.num_txt_token)
        self.v_predictor = nn.Linear(cfg.m_dim, self.num_vis_token)

        self.stability = cfg.stability

        self._reset_parameters()

    def forward( 
        self, 
        t: Tensor,
        v: Tensor,
        t_seq: Tensor=None, 
        v_seq: Tensor=None,
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        infer: bool=False,
        **kwargs,
    ):
        # `x' has been embedded into (B, L, H) by default while 
        # `x_seq' is the original sequence of (B, L).
        if memory is None: # may or may not have inter attention
            memory = memo_attn_mask = memo_key_padding_mask = None 
        if infer and not self.training:
            return self.dispatch_inference(
                t, v, t_seq = t_seq, v_seq = v_seq,
                memory=memory,
                memo_attn_mask=memo_attn_mask,
                memo_key_padding_mask=memo_key_padding_mask,
                **kwargs,
            )

        x = torch.cat([t, v], dim=1)
        x_seq = torch.cat([t_seq, v_seq], dim=1)

        B, L = x.shape[:2]
        assert L == self.len_txt_seq + self.len_vis_seq, f"seq len {L} != t_len {self.len_txt_seq} + v_len {self.len_vis_seq}"
        self_attn_mask = self.self_attn_mask[:L - 1, :L - 1]

        x = x[:, :-1]
        if self.stability > 0.:
            x = x * self.stability + x.detach() * (1 - self.stability)
        
        x = x.transpose(0, 1)
        
        x = self.encoder(
            x, mask=self_attn_mask,
        ) 

        x = x.transpose(0, 1)

        x = self.ln_output(x)

        x_txt = x[:, :self.len_txt_seq - 1]
        x_txt = self.t_predictor(x_txt)

        x_vis = x[:, -self.len_vis_seq:]
        x_vis = self.v_predictor(x_vis)

        t_seq = x_seq[:, 1:self.len_txt_seq].contiguous()
        v_seq = x_seq[:, -self.len_vis_seq:].contiguous()

        return (x_txt, x_vis), (t_seq, v_seq), None, {}
