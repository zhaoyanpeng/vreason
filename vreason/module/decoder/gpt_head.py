import numpy as np
import os, sys, time
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from fvcore.common.registry import Registry

from .decoder_head import MetaDecHead

class GPTDecHead(MetaDecHead):
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, token_vocab)
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

        self.predictor = nn.Sequential(
            nn.Linear(cfg.m_dim, len(self.token_vocab))
        ) 

        self.max_dec_len = cfg.max_dec_len
        self.beg_dec_len = cfg.beg_dec_len

        self._reset_parameters()

    def forward( 
        self, 
        x: Tensor,
        x_seq: Tensor=None, 
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
                x, x_seq = x_seq,
                memory=memory,
                memo_attn_mask=memo_attn_mask,
                memo_key_padding_mask=memo_key_padding_mask,
                **kwargs,
            )

        B, L = x.shape[:2]
        self_attn_mask = self.self_attn_mask[:L - 1, :L - 1]
        
        x = x[:, :-1]
        o_seqs = x_seq[:, 1:]
        
        x = x.transpose(0, 1)
        
        x = self.encoder(
            x, mask=self_attn_mask,
        ) 

        x = x.transpose(0, 1)

        x = self.predictor(x) 

        dec_len = L - self.contx_size
        x = x[:, -dec_len:].contiguous()
        o_seqs = o_seqs[:, -dec_len:]

        return x, o_seqs.contiguous(), None, {}

    def dispatch_inference(
        self,
        x: Tensor,
        x_seq: Tensor=None, 
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        infer_type: str="ll",
        **kwargs,
    ):
        if infer_type == "ll": # likelihood
            return self.ll_inference(
                x, x_seq = x_seq,
                memory=memory,
                memo_attn_mask=memo_attn_mask,
                memo_key_padding_mask=memo_key_padding_mask,
                **kwargs,
            )
        elif infer_type == "greedy":
            return self.greedy_inference(
                x, x_seq = x_seq,
                memory=memory,
                memo_attn_mask=memo_attn_mask,
                memo_key_padding_mask=memo_key_padding_mask,
                **kwargs,
            )

    def ll_inference( 
        self, 
        x: Tensor,
        x_seq: Tensor=None, 
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        tgt_t_pos: Tensor=None, # (L, H) additive
        tgt_v_pos: Tensor=None, # (B, H) concatenate
        token_embed: Tensor=None, # from the embedder 
        tgt_v_indice: Tensor=None, # target-panel id
        negative_seq: Tensor=None, # 
        **kwargs,
    ):
        B, K, L = negative_seq.shape

        ctx_x = x[:, :self.contx_size]
        ctx_x_seq = x_seq[:, :self.contx_size]

        x = x[:, self.contx_size:]
        x_seq = x_seq[:, self.contx_size:]

        bos, eos = x_seq[:, :1], x_seq[:, -1:]
        bos = bos.unsqueeze(1).expand(-1, K, -1)
        eos = eos.unsqueeze(1).expand(-1, K, -1)

        neg_seq = torch.cat([bos, negative_seq, eos], dim=-1).view(-1, L + 2) 
        shape = neg_seq.shape + (-1,)
        neg_emb = token_embed[neg_seq.reshape(-1)].view(*shape)
        neg_emb = neg_emb + tgt_t_pos[:shape[1]].unsqueeze(0)
        v_pos = tgt_v_pos.unsqueeze(1).unsqueeze(1).expand(-1, K, shape[1], -1).reshape(*shape)
        neg_emb = torch.cat([neg_emb, v_pos], dim=-1) 
        shape = (B, K) + neg_emb.shape[1:]
        neg_emb = neg_emb.reshape(*shape)
        
        tgt_emb = torch.cat([x.unsqueeze(1), neg_emb], dim=1)
        neg_seq = neg_seq.view(B, K, L + 2)
        tgt_seq = torch.cat([x_seq.unsqueeze(1), neg_seq], dim=1)

        logits, targets = [], []

        for i in range(K + 1):
            x = tgt_emb[:, i]
            x_seq = tgt_seq[:, i]

            x = torch.cat([ctx_x, x], dim=1)
            x_seq = torch.cat([ctx_x_seq, x_seq], dim=1)

            B, L = x.shape[:2]
            self_attn_mask = self.self_attn_mask[:L - 1, :L - 1]

            x = x[:, :-1]
            o_seq = x_seq[:, 1:]

            x = x.transpose(0, 1)
            
            x = self.encoder(
                x, mask=self_attn_mask,
            ) 

            x = x.transpose(0, 1)

            x = self.predictor(x) 
            
            # target chunk
            dec_len = L - self.contx_size
            x = x[:, -dec_len:].contiguous()
            o_seq = o_seq[:, -dec_len:]

            logits.append(x)
            targets.append(o_seq)
        
        all_logits = torch.stack(logits, dim=1)
        all_targets = torch.stack(targets, dim=1)
        return all_logits, all_targets, None, {}
            
    def greedy_inference( 
        self, 
        x: Tensor,
        x_seq: Tensor=None, 
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        tgt_t_pos: Tensor=None, # additive
        tgt_v_pos: Tensor=None, # concatenate
        token_embed: Tensor=None, # from the embedder 
        beg_dec_len: int=0, # TODO hack
        **kwargs,
    ):
        ctx_x = x[:, :self.contx_size]
        ctx_x_seq = x_seq[:, :self.contx_size]

        x = x[:, self.contx_size:]
        x_seq = x_seq[:, self.contx_size:]

        # to generate sequences
        o_seqs = x_seq[:, 1 :]
        i_seqs = x_seq[:, :-1]
        B, L = x_seq.shape[:2]

        device = x_seq.device
        beg_len = beg_dec_len 
        vocab_size, H = token_embed.shape[:2]

        logits = list() 
        if beg_len > 0:
            all_ctx = i_seqs[:, :beg_len + 1]
            logit = torch.zeros((B, beg_len, vocab_size), device=device).fill_(float("-inf"))
            logit = logit.scatter(2, all_ctx[:, 1:].unsqueeze(-1), 0)
            logits.append(logit)
        else:
            all_ctx = i_seqs[:, :1]

        def _encode_positions(seq):
            shape = seq.shape + (-1,)
            seq_emb = token_embed[seq.reshape(-1)].view(*shape)
            seq_emb = seq_emb + tgt_t_pos[:shape[1]].unsqueeze(0)
            v_pos = tgt_v_pos.unsqueeze(1).expand(-1, shape[1], -1)
            seq_emb = torch.cat([seq_emb, v_pos], dim=-1) 
            return seq_emb

        for i in range(beg_len, self.max_dec_len):
            x = _encode_positions(all_ctx)

            x = torch.cat([ctx_x, x], dim=1)

            B, L = x.shape[:2]
            self_attn_mask = self.self_attn_mask[:L, :L]

            x = x.transpose(0, 1)

            x = self.encoder(
                x, mask=self_attn_mask,
            ) 

            x = x.transpose(0, 1)

            logit = self.predictor(x[:, -1:])
            logits.append(logit)

            new_ctx = logit.argmax(dim=-1)
            all_ctx = torch.cat((all_ctx, new_ctx), dim=1)

        all_logits = torch.cat(logits, dim=1)
        return all_logits, o_seqs.contiguous(), None, {}


class NeoGPTDecHead(MetaDecHead):
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, token_vocab)
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

        self.predictor = nn.Sequential(
            nn.Linear(cfg.m_dim, len(self.token_vocab))
        ) 

        self._reset_parameters()

    def forward( 
        self, 
        x: Tensor,
        x_seq: Tensor=None, 
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
                x, x_seq = x_seq,
                memory=memory,
                memo_attn_mask=memo_attn_mask,
                memo_key_padding_mask=memo_key_padding_mask,
                **kwargs,
            )

        B, L = x.shape[:2]
        self_attn_mask = self.self_attn_mask[:L - 1, :L - 1]

        x = x[:, :-1]
        o_seqs = x_seq[:, 1:]
        
        x = x.transpose(0, 1)
        
        x = self.encoder(
            x, mask=self_attn_mask,
        ) 

        x = x.transpose(0, 1)

        x = self.predictor(x) 

        dec_len = L - self.contx_size
        x = x[:, -dec_len:].contiguous()
        o_seqs = o_seqs[:, -dec_len:]

        return x, o_seqs.contiguous(), None, {}

    def dispatch_inference(
        self,
        x: Tensor,
        x_seq: Tensor=None, 
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        **kwargs,
    ):
        if self.infer_mode == "ll": # likelihood
            return self.ll_inference(
                x, x_seq = x_seq,
                memory=memory,
                memo_attn_mask=memo_attn_mask,
                memo_key_padding_mask=memo_key_padding_mask,
                **kwargs,
            )
        elif self.infer_mode == "greedy":
            return self.greedy_inference(
                x, x_seq = x_seq,
                memory=memory,
                memo_attn_mask=memo_attn_mask,
                memo_key_padding_mask=memo_key_padding_mask,
                **kwargs,
            )
        else:
            raise ValueError(f"unknown inference mode `{self.infer_mode}'")

    def ll_inference( 
        self, 
        x: Tensor,
        x_seq: Tensor=None, 
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        neg_seq: Tensor=None, #
        neg_emb: Tensor=None, #
        **kwargs,
    ):
        B, K, L = neg_seq.shape

        # slice out context and target
        ctx_x = x[:, :self.contx_size]
        ctx_x_seq = x_seq[:, :self.contx_size]

        x = x[:, self.contx_size:]
        x_seq = x_seq[:, self.contx_size:]
        
        # prepend target to the negative 
        tgt_emb = torch.cat([x.unsqueeze(1), neg_emb], dim=1)
        tgt_seq = torch.cat([x_seq.unsqueeze(1), neg_seq], dim=1)

        logits, targets = [], []

        for i in range(K + 1):
            x = tgt_emb[:, i]
            x_seq = tgt_seq[:, i]

            x = torch.cat([ctx_x, x], dim=1)
            x_seq = torch.cat([ctx_x_seq, x_seq], dim=1)

            B, L = x.shape[:2]
            self_attn_mask = self.self_attn_mask[:L - 1, :L - 1]

            x = x[:, :-1]
            o_seq = x_seq[:, 1:]

            x = x.transpose(0, 1)
            
            x = self.encoder(
                x, mask=self_attn_mask,
            ) 

            x = x.transpose(0, 1)

            x = self.predictor(x) 
            
            # target chunk
            dec_len = L - self.contx_size
            x = x[:, -dec_len:].contiguous()
            o_seq = o_seq[:, -dec_len:]

            logits.append(x)
            targets.append(o_seq)
        
        all_logits = torch.stack(logits, dim=1)
        all_targets = torch.stack(targets, dim=1)
        return all_logits, all_targets, None, {}

    def greedy_inference( 
        self, 
        x: Tensor,
        x_seq: Tensor=None, 
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        tgt_t_pos: Tensor=None, # additive
        tgt_v_pos: Tensor=None, # concatenate
        token_embed: Tensor=None, # from the embedder 
        **kwargs,
    ):
        # slice out context and target
        ctx_x = x[:, :self.contx_size]
        ctx_x_seq = x_seq[:, :self.contx_size]

        x = x[:, self.contx_size:]
        x_seq = x_seq[:, self.contx_size:]

        # to generate sequences
        o_seqs = x_seq[:, 1 :]
        i_seqs = x_seq[:, :-1]
        B, L = x_seq.shape[:2]

        device = x_seq.device
        beg_len = self.beg_dec_len 
        vocab_size, H = token_embed.shape[:2]

        logits = list() 
        if beg_len > 0:
            all_ctx = i_seqs[:, :beg_len + 1]
            logit = torch.zeros((B, beg_len, vocab_size), device=device).fill_(float("-inf"))
            logit = logit.scatter(2, all_ctx[:, 1:].unsqueeze(-1), 0)
            logits.append(logit)
        else:
            all_ctx = i_seqs[:, :1]

        def _encode_positions(seq):
            shape = seq.shape + (-1,)
            seq_emb = token_embed[seq.reshape(-1)].view(*shape)
            seq_emb = seq_emb + tgt_t_pos[:shape[1]].unsqueeze(0)
            v_pos = tgt_v_pos.unsqueeze(1).expand(-1, shape[1], -1)
            seq_emb = torch.cat([seq_emb, v_pos], dim=-1) 
            return seq_emb

        for i in range(beg_len, self.max_dec_len):
            x = _encode_positions(all_ctx)

            x = torch.cat([ctx_x, x], dim=1)

            B, L = x.shape[:2]
            self_attn_mask = self.self_attn_mask[:L, :L]

            x = x.transpose(0, 1)

            x = self.encoder(
                x, mask=self_attn_mask,
            ) 

            x = x.transpose(0, 1)

            logit = self.predictor(x[:, -1:])
            logits.append(logit)

            new_ctx = logit.argmax(dim=-1)
            all_ctx = torch.cat((all_ctx, new_ctx), dim=1)

        all_logits = torch.cat(logits, dim=1)
        return all_logits, o_seqs.contiguous(), None, {}
