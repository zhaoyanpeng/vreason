import numpy as np
import os, sys, time
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import torch.nn.functional as F

from . import MetaDecHead, DECODER_HEADS_REGISTRY
from .. import _get_activation_fn, SoftPositionalEncoder

if True: 
    from .. import TransformerEncoder, TransformerEncoderLayer
    from .. import TransformerDecoder, TransformerDecoderLayer
else:
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    from torch.nn import TransformerDecoder, TransformerDecoderLayer

__all__ = ["Infer", "DalleGPTDecHead", "DalleBartDecHead"]

class Infer: # a standalone class
    def _debug_cache(self, cache):
        t, v, x = cache 
        if self.cache != None:
            to, vo, xo = self.cache
            eq1 = (to == t).all().cpu().item()
            df1 = (to -  t).abs().sum().cpu().item()

            eq2 = (xo == x[:, :-1]).all().cpu().item()
            df2 = (xo -  x[:, :-1]).abs().sum().cpu().item()

            eq3 = (vo == v[:, :-1]).all().cpu().item()
            df3 = (vo -  v[:, :-1]).abs().sum().cpu().item()

            eq22 = torch.allclose(xo, x[:, :-1], atol=1e-6)
            df22 = (xo - x[:, :-1]).abs().max().cpu().item()
            print("infer", eq1, df1, eq2, df2, eq22, df22)
        self.cache = cache

    def dispatch_inference(
        self,
        t: Tensor,
        v: Tensor,
        t_seq: Tensor=None, 
        v_seq: Tensor=None,
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        infer_type: str="default",
        **kwargs,
    ):
        # TODO for cache debug
        if not hasattr(self, "cache"):
            setattr(self, "cache", None)
        # TODO conditional switch
        inference_fn = self.infer
        return inference_fn(
            t, v, t_seq = t_seq, v_seq = v_seq,
            memory=memory,
            memo_attn_mask=memo_attn_mask,
            memo_key_padding_mask=memo_key_padding_mask,
            **kwargs,
        )

@DECODER_HEADS_REGISTRY.register()
class DalleGPTDecHead(MetaDecHead, Infer):
    def __init__(self, cfg, txt_token_vocab, vis_token_vocab=None, **kwargs):
        super().__init__(cfg, None)
        self.encoder = None
        if cfg.num_layer > 0:
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
        if self.stability > 0.:
            x = x * self.stability + x.detach() * (1 - self.stability)
        
        x = x.transpose(0, 1)
        
        x = self.encoder(
            x, mask=self_attn_mask,
        ) 
        if isinstance(x, (tuple, list)):
            x, *_ = x

        x = x.transpose(0, 1)

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
        use_cache: bool=False,
        cache: Tensor=None,
        **kwargs,
    ): # one-step inference
        x = torch.cat([t, v], dim=1)
        x_seq = torch.cat([t_seq, v_seq], dim=1)
        
        B, L = x.shape[:2]
        self_attn_mask = self.self_attn_mask[:L, :L]
       
        x = x.transpose(0, 1)
        
        x = self.encoder(
            x, mask=self_attn_mask, caches=cache, use_cache=use_cache
        ) 
        if isinstance(x, (tuple, list)):
            x, cache = x

        x = x.transpose(0, 1)

        #self._debug_cache((t, v, x))

        x_txt = x[:, :self.len_txt_seq - 1]
        x_txt = self.t_predictor(x_txt)

        x_vis = x[:, self.len_txt_seq - 1:]
        x_vis = self.v_predictor(x_vis)

        t_seq = x_seq[:, 1:self.len_txt_seq].contiguous()
        v_seq = x_seq[:, self.len_txt_seq: ].contiguous()

        if not sampling: # logits are not for sampling
            x_vis = x_vis[:, :v_seq.shape[1]]

        if self.len_txt_seq == 1:
            return (x_vis,), (v_seq,), cache, {}
        else:
            return (x_txt, x_vis), (t_seq, v_seq), cache, {}

@DECODER_HEADS_REGISTRY.register()
class DalleBartDecHead(MetaDecHead, Infer):
    def __init__(self, cfg, txt_token_vocab, vis_token_vocab=None, **kwargs):
        super().__init__(cfg, None)
        self.encoder = None
        if cfg.num_layer > 0:
            layer_fn = TransformerDecoderLayer(
                cfg.m_dim, cfg.num_head, cfg.f_dim, cfg.t_dropout, 
                activation=cfg.activation, norm_first=cfg.norm_first
            )
            iln = oln = lambda x: x
            if cfg.norm_first:
                oln = nn.LayerNorm(cfg.m_dim)
            else:
                iln = nn.LayerNorm(cfg.m_dim)
            self.encoder = TransformerDecoder(layer_fn, cfg.num_layer, iln=iln, oln=oln)

            self_attn_mask = (torch.triu(
                torch.ones(cfg.block_size, cfg.block_size, dtype=torch.uint8), 
            diagonal=1) == 1)
            self.register_buffer("self_attn_mask", self_attn_mask) 

        self.num_head = cfg.num_head
        self.stability = cfg.stability

        self.len_vis_seq = cfg.len_vis_seq
        self.vis_token_vocab = vis_token_vocab

        self.predictor = nn.Linear(cfg.m_dim, len(vis_token_vocab))

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

        x = v 
        x_seq = v_seq 

        B, L = x.shape[:2]
        self_attn_mask = self.self_attn_mask[:L - 1, :L - 1]
        
        x = x[:, :-1]
        if self.stability > 0.:
            x = x * self.stability + x.detach() * (1 - self.stability)

        memory = memory.transpose(0, 1)

        x = x.transpose(0, 1)
        
        x = self.encoder(
            x, 
            memory=memory, 
            tgt_mask=self_attn_mask,
            memory_key_padding_mask=memo_key_padding_mask,
        ) 
        if isinstance(x, (tuple, list)):
            x, *_ = x

        x = x.transpose(0, 1)

        x = self.predictor(x) 
        return (x,), (x_seq,), None, {}

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
        use_cache: bool=False,
        cache: Tensor=None,
        **kwargs,
    ): # one-step inference
        x = v 
        x_seq = v_seq 

        B, L = x.shape[:2]
        self_attn_mask = self.self_attn_mask[:L, :L]
        
        memory = memory.transpose(0, 1)
        
        x = x.transpose(0, 1)
         
        x = self.encoder(
            x, 
            memory=memory, 
            tgt_mask=self_attn_mask,
            memory_key_padding_mask=memo_key_padding_mask,
            caches=cache, use_cache=use_cache
        ) 
        if isinstance(x, (tuple, list)):
            x, cache = x

        x = x.transpose(0, 1)
        
        #self._debug_cache((memory, v, x))

        x = self.predictor(x) 

        if not sampling: # logits are not for sampling
            x = x[:, :x_seq.shape[1]]

        return (x,), (x_seq,), cache, {}
