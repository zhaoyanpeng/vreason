import numpy as np
import os, sys, time
import torch
from typing import Tuple
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import torch.nn.functional as F

from fvcore.common.registry import Registry

from .. import PartiallyFixedEmbedding

ENCODER_HEADS_REGISTRY = Registry("ENCODER_HEADS")
ENCODER_HEADS_REGISTRY.__doc__ = """
Registry for encoder heads.
"""

def build_encoder_head(cfg, vocab, **kwargs):
    return ENCODER_HEADS_REGISTRY.get(cfg.name)(cfg, vocab, **kwargs)

class MetaEncHead(nn.Module):
    def __init__(self, cfg, token_vocab):
        super(MetaEncHead, self).__init__()
        self.token_vocab = token_vocab
    def _reset_parameters(self):
        pass
    @property
    def emb_size(self):
        return 0 
    @property
    def emb_weight(self):
        return 0 
    @property
    def output_size(self):
        return 0 

@ENCODER_HEADS_REGISTRY.register()
class DummyEncHead(MetaEncHead):
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, token_vocab)
        pass
    def forward(self, *args, **kwargs):
        return None, None, None, {} 

@ENCODER_HEADS_REGISTRY.register()
class TokenEncHead(MetaEncHead):
    def __init__(self, cfg, token_vocab, emb_size=None, **kwargs):
        super().__init__(cfg, token_vocab)
        assert cfg.mode in {"gpt", "seq2seq"}, f"GPT (gpt) or Encoder-Decoder (seq2seq) mode"
        self.mode = cfg.mode

        # contains only special tokens e.g.,
        # ["<MASK>", "<IMG|BOS>", "<IMG|EOS>", "<TOK|BOS>", "<TOK|EOS>", "<SEP>", "<ROW|SEP>", "<COL|SEP>"]
        self.token_embed = nn.Embedding(
            len(self.token_vocab), emb_size, padding_idx=self.token_vocab.PAD_IDX
        )
        special_indice = torch.tensor(self.token_vocab(["<MASK>", "<IMG|BOS>", "<IMG|EOS>"]))
        self.register_buffer("mask_bos_eos_indice", special_indice)

        # positional embeddings, image-level position and visual-token-level position
        # token-level: {0, 1, 2, ..., 100} visual tokens per image.
        # image-level: {0, 1, 2, ..., 8} images, could be None.
        num_token_pos, num_image_pos = cfg.num_token_pos, cfg.num_image_pos
        token_pos_dim, image_pos_dim = cfg.token_pos_dim, cfg.image_pos_dim
        self.t_pos_embed = nn.Parameter(Tensor(num_token_pos, token_pos_dim))
        self.v_pos_embed = nn.Parameter(Tensor(num_image_pos, image_pos_dim))

        self._emb_size = emb_size + image_pos_dim
        self.t_pos_start = cfg.t_pos_start
        self.v_pos_start = cfg.v_pos_start
        
        self.aug_prob = cfg.aug_prob
        self.num_image = cfg.num_image
        weight = [self.aug_prob / (self.num_image - 1)] * (self.num_image - 1)
        weight = torch.tensor(weight + [1 - self.aug_prob], dtype=torch.float)
        self.register_buffer("sample_weight", weight)

        self._reset_parameters()

    def _reset_parameters(self):
        for weight in [self.token_embed.weight, self.t_pos_embed, self.v_pos_embed]:
            if weight is not None:
                nn.init.xavier_uniform_(weight) 

    def _encode_positions(self, x, sequence):
        device = x.device
        B, V_n, T_n, H = x.shape

        shape = (B, V_n, -1, -1)
        mask, bos, eos = self.token_embed(self.mask_bos_eos_indice).chunk(3, dim=0)
        bos = bos.unsqueeze(0).unsqueeze(0).expand(*shape)
        eos = eos.unsqueeze(0).unsqueeze(0).expand(*shape)
        
        x = torch.cat([bos, x, eos], dim=-2) # <IMG|BOS> ... <IMG|EOS>

        t_pos_embed = self.t_pos_embed[self.t_pos_start : self.t_pos_start + T_n + 2]
        t_pos = t_pos_embed.unsqueeze(0).unsqueeze(0)
        
        x = x + t_pos # add intra-image position

        v_pos_embed = self.v_pos_embed[self.v_pos_start : self.v_pos_start + V_n]
        shape = (B, -1, T_n + 2, -1)
        v_pos = v_pos_embed.unsqueeze(0).unsqueeze(-2).expand(*shape)

        x = torch.cat([x, v_pos], dim=-1) # cat inter-image position
        
        batch_indice = torch.arange(B, device=device)
        if not self.training:
            target_indice = torch.tensor([self.num_image - 1] * B, device=device)
        else: # randomly sample an image at training time
            target_indice = torch.multinomial(
                self.sample_weight, B, replacement=True
            )

        target_emb = x[batch_indice, target_indice]
        target_seq = sequence[batch_indice, target_indice]
        bos, eos = self.mask_bos_eos_indice[1:].unsqueeze(0).expand(B, -1).chunk(2, dim=1)
        target_seq = torch.cat([bos, target_seq, eos], dim=1)

        mask = mask.expand(B, -1)
        mask_pos = v_pos_embed[target_indice]
        mask = torch.cat([mask, mask_pos], dim=-1)

        samples = [] 
        H = x.shape[-1]
        for b, k in enumerate(target_indice):
            l = x[b, :k].reshape(-1, H)
            r = x[b, k + 1:].reshape(-1, H)
            m = mask[b : b + 1].reshape(-1, H)
            sample = torch.cat([l, m, r], dim=0)
            samples.append(sample)
        x = torch.stack(samples, dim=0)
        extra = {"tgt_v_indice": target_indice, "tgt_t_pos": t_pos_embed, "tgt_v_pos": mask_pos}

        if self.mode == "gpt":
            bos, eos = self.mask_bos_eos_indice[1:].unsqueeze(0).unsqueeze(0).expand(B, V_n, -1).chunk(2, dim=-1)
            pad_seq = torch.cat([bos, sequence, eos], dim=-1)
            samples = [] # insert mask
            for b, k in enumerate(target_indice):
                l = pad_seq[b, :k].reshape(-1)
                r = pad_seq[b, k + 1:].reshape(-1)
                m = self.mask_bos_eos_indice[0 : 1]
                sample = torch.cat([l, m, r], dim=0)
                samples.append(sample)
            pad_seq = torch.stack(samples, dim=0)
            # concat for GPT, x will be ignored.
            target_seq = torch.cat([pad_seq, target_seq], dim=1)
            target_emb = torch.cat([x, target_emb], dim=1) 

        return x, target_emb, target_seq, extra 

    @property
    def emb_size(self):
        return self._emb_size

    def forward(self, x, sequence, **kwargs):
        return self._encode_positions(x, sequence)

@ENCODER_HEADS_REGISTRY.register()
class NeoTokenEncHead(MetaEncHead):
    def __init__(self, cfg, token_vocab, emb_size=None, fixed_weight=None, **kwargs):
        super().__init__(cfg, token_vocab)
        assert cfg.mode in {"gpt", "seq2seq"}, f"GPT (gpt) or Encoder-Decoder (seq2seq) mode"
        self.mode = cfg.mode

        # fixed weight followed by special tokens e.g.,
        # ["<MASK>", "<IMG|BOS>", "<IMG|EOS>", "<TOK|BOS>", "<TOK|EOS>", "<SEP>", "<ROW|SEP>", "<COL|SEP>"]
        self.token_embed = PartiallyFixedEmbedding(token_vocab, fixed_weight)

        special_indice = torch.tensor(self.token_vocab(["<MASK>", "<IMG|BOS>", "<IMG|EOS>"]))
        self.register_buffer("mask_bos_eos_indice", special_indice)

        # positional embeddings, image-level position and visual-token-level position
        # token-level: {0, 1, 2, ..., 100} visual tokens per image.
        # image-level: {0, 1, 2, ..., 8} images, could be None.
        num_token_pos, num_image_pos = cfg.num_token_pos, cfg.num_image_pos
        token_pos_dim, image_pos_dim = cfg.token_pos_dim, cfg.image_pos_dim
        self.t_pos_embed = nn.Parameter(Tensor(num_token_pos, token_pos_dim))
        self.v_pos_embed = nn.Parameter(Tensor(num_image_pos, image_pos_dim))

        self._emb_size = emb_size + image_pos_dim
        self.t_pos_start = cfg.t_pos_start
        self.v_pos_start = cfg.v_pos_start
        
        self.aug_prob = cfg.aug_prob
        self.num_image = cfg.num_image
        weight = [self.aug_prob / (self.num_image - 1)] * (self.num_image - 1)
        weight = torch.tensor(weight + [1 - self.aug_prob], dtype=torch.float)
        self.register_buffer("sample_weight", weight)

        self._reset_parameters()

    def _reset_parameters(self):
        for weight in [self.t_pos_embed, self.v_pos_embed]:
            if weight is not None:
                nn.init.xavier_uniform_(weight) 

    @property
    def emb_size(self):
        return self._emb_size

    @property
    def emb_weight(self):
        return self.token_embed.full_weight()

    def _encode_positions(self, sequence, negative=None):
        device = sequence.device
        B, V_n, T_n = sequence.shape
        
        ## sample targets
        shape = (B, V_n, -1, -1)
        batch_indice = torch.arange(B, device=device)
        if not self.training:
            target_indice = torch.tensor([self.num_image - 1] * B, device=device)
        else: # randomly sample an image at training time
            target_indice = torch.multinomial(
                self.sample_weight, B, replacement=True
            )
        
        ## arange sequences
        bos, eos = (
            self.mask_bos_eos_indice[1:]
                .unsqueeze(0) # B
                .unsqueeze(0) # V_n
                .expand(B, V_n, -1)
                .chunk(2, dim=-1)
        )
        pad_seq = torch.cat([bos, sequence, eos], dim=-1)
        seq_emb = self.token_embed(pad_seq)

        ## encode positions
        t_pos_embed = self.t_pos_embed[self.t_pos_start : self.t_pos_start + T_n + 2]
        t_pos = t_pos_embed.unsqueeze(0).unsqueeze(0) # (B, V_n, T_n + 2, H)
        # add intra-image position
        seq_emb = seq_emb + t_pos

        v_pos_embed = self.v_pos_embed[self.v_pos_start : self.v_pos_start + V_n]
        shape = (B, -1, T_n + 2, -1) # (B, V_n, T_n + 2, H)
        v_pos = v_pos_embed.unsqueeze(0).unsqueeze(-2).expand(*shape)
        # cat inter-image position
        seq_emb = torch.cat([seq_emb, v_pos], dim=-1)
        
        ## slice out targets
        target_seq = pad_seq[batch_indice, target_indice]
        target_emb = seq_emb[batch_indice, target_indice]

        ## insert mask into context
        # encode mask
        mask_emb = self.token_embed(self.mask_bos_eos_indice[0 : 1])
        mask_emb = mask_emb.expand(B, -1)
        mask_pos = v_pos_embed[target_indice]
        mask_emb = torch.cat([mask_emb, mask_pos], dim=-1)
        # make context 
        seqs, embs = [], []
        H = seq_emb.shape[-1]
        for b, k in enumerate(target_indice):
            # seq
            l = pad_seq[b, :k].reshape(-1)
            r = pad_seq[b, k + 1:].reshape(-1)
            m = self.mask_bos_eos_indice[0 : 1]
            seq = torch.cat([l, m, r], dim=0)
            seqs.append(seq)
            # emb
            l = seq_emb[b, :k].reshape(-1, H)
            r = seq_emb[b, k + 1:].reshape(-1, H)
            m = mask_emb[b : b + 1].reshape(-1, H)
            emb = torch.cat([l, m, r], dim=0)
            embs.append(emb)
        ctx_seq = torch.stack(seqs, dim=0)
        ctx_emb = torch.stack(embs, dim=0)

        if self.mode == "gpt":
            tgt_seq = torch.cat([ctx_seq, target_seq], dim=1)
            tgt_emb = torch.cat([ctx_emb, target_emb], dim=1)
        elif self.mode == "seq2seq":
            tgt_seq = target_seq
            tgt_emb = target_emb

        ## encode negative
        neg_emb = neg_seq = None
        if negative != None:
            N_n = negative.shape[1]
            bos = bos[:, :N_n] # FIXME N_n <= V_n
            eos = eos[:, :N_n] # FIXME N_n <= V_n
            neg_seq = torch.cat([bos, negative, eos], dim=-1)
            neg_emb = self.token_embed(neg_seq) # (B, N_n, T_n + 2, H)
            # intra-image
            neg_emb = neg_emb + t_pos
            # inter-image
            neg_pos = v_pos_embed[target_indice]
            neg_pos = neg_pos.unsqueeze(1).unsqueeze(1).expand(B, N_n, T_n + 2, -1)
            neg_emb = torch.cat([neg_emb, neg_pos], dim=-1)

        extra = {
            "token_embed": self.emb_weight,
            "tgt_v_indice": target_indice, 
            "tgt_t_pos": t_pos_embed, 
            "tgt_v_pos": mask_pos,
        } # needed for decoding

        return (ctx_emb, ctx_seq), (tgt_emb, tgt_seq), (neg_emb, neg_seq), extra

    def forward(self, sequence, negative=None, **kwargs):
        return self._encode_positions(sequence, negative)

@ENCODER_HEADS_REGISTRY.register()
class TorchTFEncHead(MetaEncHead):
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
        
        x = x.transpose(0, 1)

        x = self.encoder(
            x, src_key_padding_mask=self_key_padding_mask,
        ) 

        x = x.transpose(0, 1)

        return x, None, None, {} 
