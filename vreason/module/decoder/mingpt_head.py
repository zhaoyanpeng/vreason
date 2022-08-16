import numpy as np
import os, sys, time
import warnings
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import torch.nn.functional as F

from . import Infer, MetaDecHead, DECODER_HEADS_REGISTRY
from .. import MiniTF, MiniTFBlock, MiniTFAttention

try:
    from lib.mingpt import GPT
except Exception as e:
    warnings.warn(
        "No GPT module: {e}", UserWarning
    )

__all__ = ["MiniGPTDecHead", "MiniTFDecHead"]

@DECODER_HEADS_REGISTRY.register()
class MiniGPTDecHead(MetaDecHead, Infer):
    def __init__(self, cfg, txt_token_vocab, vis_token_vocab=None, **kwargs):
        super().__init__(cfg, None)
        config = dict(
            model_type=cfg.model_type, n_layer=cfg.n_layer, n_head=cfg.n_head,
            n_embd=cfg.n_embd, vocab_size=cfg.vocab_size, block_size=cfg.block_size, 
        )
        model_config = GPT.get_default_config()
        model_config.merge_from_dict(config)
        self.encoder = GPT(model_config) 

        self.len_txt_seq = cfg.len_txt_seq
        self.len_vis_seq = cfg.len_vis_seq

    def configure_optimizers(self, cfg):
        return self.encoder.configure_optimizers(cfg)

    def forward( 
        self, 
        t: Tensor,
        v: Tensor,
        t_seq: Tensor=None, 
        v_seq: Tensor=None,
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        force_infer: bool=False,
        infer: bool=False,
        **kwargs,
    ):
        # `x' has been embedded into (B, L, H) by default while 
        # `x_seq' is the original sequence of (B, L).
        if memory is None: # may or may not have inter attention
            memory = memo_attn_mask = memo_key_padding_mask = None 
        if force_infer or (infer and not self.training):
            return self.dispatch_inference(
                t, v, t_seq = t_seq, v_seq = v_seq,
                memory=memory,
                memo_attn_mask=memo_attn_mask,
                memo_key_padding_mask=memo_key_padding_mask,
                **kwargs,
            )
        
        x_seq = torch.cat([t_seq, v_seq], dim=1)

        x = self.encoder(x_seq[:, :-1]) 
        if isinstance(x, (tuple, list)):
            x, *_ = x

        x_txt = x[:, :self.len_txt_seq - 1]
        x_vis = x[:, -self.len_vis_seq:]

        t_seq = x_seq[:, 1:self.len_txt_seq].contiguous()
        v_seq = x_seq[:, -self.len_vis_seq:].contiguous()
        
        if self.len_txt_seq == 1:
            return (x_vis,), (v_seq,), None, {}
        else:
            return (x_txt, x_vis), (t_seq, v_seq), None, {}

    def infer(
        self,
        t: Tensor,
        v: Tensor,
        t_seq: Tensor=None, 
        v_seq: Tensor=None,
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        sampling: bool=True,
        use_cache: bool=True,
        cache: Tensor=None,
        **kwargs,
    ): # one-step inference
        x_seq = torch.cat([t_seq, v_seq], dim=1)
       
        x = self.encoder(x_seq) 
        if isinstance(x, (tuple, list)):
            x, *_ = x

        x_txt = x[:, :self.len_txt_seq - 1]
        x_vis = x[:, self.len_txt_seq - 1:]

        t_seq = x_seq[:, 1:self.len_txt_seq].contiguous()
        v_seq = x_seq[:, self.len_txt_seq: ].contiguous()

        if not sampling: # logits are not for sampling
            x_vis = x_vis[:, :v_seq.shape[1]]

        if self.len_txt_seq == 1:
            return (x_vis,), (v_seq,), None, {}
        else:
            return (x_txt, x_vis), (t_seq, v_seq), None, {}

@DECODER_HEADS_REGISTRY.register()
class MiniTFDecHead(MetaDecHead, Infer):
    def __init__(self, cfg, txt_token_vocab, vis_token_vocab=None, **kwargs):
        super().__init__(cfg, None)
        self.encoder = None
        if cfg.num_layer > 0: 
            layer_fn = lambda ilayer: eval(cfg.block)(
                cfg.m_dim, cfg.num_head, cfg.f_dim, cfg.attn_cls_intra, 
                attn_cls_inter=cfg.attn_cls_inter, 
                ilayer=ilayer,
                dropout=cfg.t_dropout, 
                qk_scale=cfg.qk_scale,
                norm_first=cfg.norm_first,
                activation=cfg.activation,
                attn_dropout=cfg.attn_dropout,
                proj_dropout=cfg.proj_dropout,
                num_head_intra=cfg.num_head_intra,
                num_head_inter=cfg.num_head_inter,
                inter_layers=list(cfg.inter_layers),
            )
            iln = oln = lambda x: x
            if cfg.norm_first:
                oln = nn.LayerNorm(cfg.m_dim)
            else:
                iln = nn.LayerNorm(cfg.m_dim)
            self.encoder = MiniTF(layer_fn, cfg.num_layer, iln=iln, oln=oln) 

            self_attn_mask = (torch.triu(
                torch.ones(cfg.block_size, cfg.block_size, dtype=torch.uint8), 
            diagonal=1) == 1)
            self.register_buffer("self_attn_mask", self_attn_mask) 

        self.num_head = cfg.num_head
        self.stability = cfg.stability

        self.len_txt_seq = cfg.len_txt_seq
        self.len_vis_seq = cfg.len_vis_seq

        self.txt_token_vocab = txt_token_vocab
        self.vis_token_vocab = vis_token_vocab

        self.t_predictor = nn.Linear(cfg.m_dim, len(txt_token_vocab))
        self.v_predictor = nn.Linear(cfg.m_dim, len(vis_token_vocab))

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
        force_infer: bool=False, # infer during training
        infer: bool=False,
        **kwargs,
    ):
        # `x' has been embedded into (B, L, H) by default while 
        # `x_seq' is the original sequence of (B, L).
        if memory is None: # may or may not have inter attention
            memory = memo_attn_mask = memo_key_padding_mask = None 
        if force_infer or (infer and not self.training):
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
        assert L == self.len_txt_seq + self.len_vis_seq, (
            f"seq len ({L}) != t_len ({self.len_txt_seq}) + v_len ({self.len_vis_seq})"
        )
        self_attn_mask = self.self_attn_mask[:L - 1, :L - 1]

        x = x[:, :-1]
        x = self.encoder(
            x, memory=memory, 
            self_attn_mask=self_attn_mask.squeeze(0),
            memo_key_padding_mask=memo_key_padding_mask,
            require_attn_weight=True
        ) 
        if isinstance(x, (tuple, list)):
            x, *_ = x

        x_txt = x[:, :self.len_txt_seq - 1]
        x_txt = self.t_predictor(x_txt)

        x_vis = x[:, -self.len_vis_seq:]
        x_vis = self.v_predictor(x_vis)

        t_seq = x_seq[:, 1:self.len_txt_seq].contiguous()
        v_seq = x_seq[:, -self.len_vis_seq:].contiguous()
        
        if self.len_txt_seq == 1:
            return (x_vis,), (v_seq,), None, {}
        else:
            return (x_txt, x_vis), (t_seq, v_seq), None, {}

    def infer(
        self,
        t: Tensor,
        v: Tensor,
        t_seq: Tensor=None, 
        v_seq: Tensor=None,
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        sampling: bool=True,
        use_cache: bool=True,
        cache: Tensor=None,
        **kwargs,
    ): # one-step inference
        x = torch.cat([t, v], dim=1)
        x_seq = torch.cat([t_seq, v_seq], dim=1)
        
        B, L = x.shape[:2]
        self_attn_mask = self.self_attn_mask[:L, :L]
       
        x = self.encoder(
            x, memory=memory, 
            self_attn_mask=self_attn_mask.squeeze(0),
            memo_key_padding_mask=memo_key_padding_mask,
            require_attn_weight=True
        ) 
        if isinstance(x, (tuple, list)):
            x, *_ = x

        x_txt = x[:, :self.len_txt_seq - 1]
        x_txt = self.t_predictor(x_txt)

        x_vis = x[:, self.len_txt_seq - 1:]
        x_vis = self.v_predictor(x_vis)

        t_seq = x_seq[:, 1:self.len_txt_seq].contiguous()
        v_seq = x_seq[:, self.len_txt_seq: ].contiguous()

        if not sampling: # logits are not for sampling
            x_vis = x_vis[:, :v_seq.shape[1]]

        if self.len_txt_seq == 1:
            return (x_vis,), (v_seq,), None, {}
        else:
            return (x_txt, x_vis), (t_seq, v_seq), None, {}
