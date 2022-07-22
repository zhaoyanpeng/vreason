import numpy as np
import os, sys, time
import torch
from typing import Tuple
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch.nn.functional as F

from . import MetaEncHead, ENCODER_HEADS_REGISTRY
from .. import _get_activation_fn, Hard1DEmbedder, Hard2DEmbedder, Soft2DEmbedder

__all__ = ["DalleTokenEncHead"]


@ENCODER_HEADS_REGISTRY.register()
class DalleTokenEncHead(MetaEncHead):
    def __init__(self, cfg, txt_token_vocab, vis_token_vocab=None, **kwargs):
        super().__init__(cfg, None)
        self.mode = cfg.mode

        self._emb_size = cfg.embed_dim
        self.len_txt_seq = cfg.len_txt_seq
        self.len_vis_seq = cfg.len_vis_seq
        H = W = int(self.len_vis_seq ** 0.5)

        self.txt_token_vocab = txt_token_vocab
        self.vis_token_vocab = vis_token_vocab

        self.num_txt_token = len(txt_token_vocab)
        self.num_vis_token = len(vis_token_vocab)

        self.txt_embed = eval(cfg.txt_embed)(
            self.num_txt_token, self._emb_size, (self.len_txt_seq,),
            tok_padding_idx=txt_token_vocab.PAD_IDX, use_pos_padding=False,
        )
        self.vis_embed = eval(cfg.vis_embed)(
            self.num_vis_token, self._emb_size, (H,) * 2,
            tok_padding_idx=None, use_pos_padding=False,
        )

    def _reset_parameters(self):
        pass

    @property
    def emb_size(self):
        return self._emb_size

    def _encode_positions(self, t=None, v=None):
        if t is not None:
            t = self.txt_embed(t)
        if v is not None:
            v = self.vis_embed(v)
        return t, v

    def forward(self, t_seq, v_seq=None, **kwargs):
        return self._encode_positions(t=t_seq, v=v_seq)
