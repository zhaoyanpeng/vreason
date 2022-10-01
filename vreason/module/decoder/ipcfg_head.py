import numpy as np
import os, sys, time
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import torch.nn.functional as F

from . import MetaDecHead, DECODER_HEADS_REGISTRY
from .. import InsideAlg2D

__all__ = ["IPCFGDecHead"]

class ResLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.linear(x) + x

class IPCFGDecHead(MetaDecHead):
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, None)
        self.s_dim = cfg.s_dim
        self.z_dim = cfg.z_dim

        assert self.z_dim >= 0, f"Use the latent sequence embedding?"

        self._num_rnd_consumed = 0

    def _reset_parameters(self):
        skip_enc_emb = hasattr(self, "enc_emb") and not isinstance(self.enc_emb, nn.Embedding)
        for name, p in self.named_parameters():
            if skip_enc_emb and name.startswith("enc_emb"):
                continue
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
                self._num_rnd_consumed += torch.numel(p)

    @property
    def num_rnd_consumed(self):
        return self._num_rnd_consumed

    def _count_rnd_consumed(self):
        for k, p in self.named_parameters():
            self._num_rnd_consumed += torch.numel(p)

    @staticmethod
    def kl(mean, lvar):
        return -0.5 * (lvar - torch.pow(mean, 2) - torch.exp(lvar) + 1)

@DECODER_HEADS_REGISTRY.register()
class NaiveIPCFGDecHead(IPCFGDecHead, InsideAlg2D):
    def __init__(self, cfg, txt_token_vocab, vis_token_vocab=None, **kwargs):
        super().__init__(cfg, None)
        h_dim = cfg.h_dim
        w_dim = cfg.w_dim
        z_dim = cfg.z_dim
        s_dim = cfg.s_dim

        self.T = cfg.T
        self.NT = cfg.NT
        self.NT_T = cfg.NT + cfg.T
        self.vis_token_vocab = vis_token_vocab
        V = len(self.vis_token_vocab)
        self.V = V
        
        self.term_emb = nn.Parameter(torch.randn(self.T, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))
        
        rule_dim = s_dim + z_dim
        rule_modules = (
            nn.Linear(rule_dim, s_dim),
            ResLayer(s_dim, s_dim),
            ResLayer(s_dim, s_dim),
            nn.Linear(s_dim, self.NT_T ** 2 * 2)
        ) # horizon and vertical
        self.rule_mlp = nn.Sequential(*rule_modules)
        
        root_dim = s_dim + z_dim
        root_modules = (
            nn.Linear(root_dim, s_dim),
            ResLayer(s_dim, s_dim),
            ResLayer(s_dim, s_dim),
            nn.Linear(s_dim, self.NT)
        )
        self.root_mlp = nn.Sequential(*root_modules)
        
        term_dim = s_dim + z_dim
        term_modules = (
            nn.Linear(term_dim, s_dim),
            ResLayer(s_dim, s_dim),
            nn.Linear(s_dim, V)
        )
        self.term_mlp = nn.Sequential(*term_modules)

        self._count_rnd_consumed()
        self._reset_parameters()

    def parameterize(self, x, mean, lvar, use_mean=False, **kwargs):
        B, HW = x.shape[:2]
        H = W = int(np.sqrt(HW))
                
        if self.z_dim > 0:
            z = mean
            if not use_mean:
                z = torch.zeros_like(mean).normal_(0, 1)
                z = (0.5 * lvar).exp() * z + mean
            kl = self.kl(mean, lvar).sum(1) 
            self.z = z
        else:
            kl = torch.zeros((B, 1)).to(x)
            self.z = None
        
        def rules(x):
            nonterm_emb = self.nonterm_emb.unsqueeze(0).expand(
                B, self.NT, self.s_dim
            )
            if self.z_dim > 0:
                z = self.z.unsqueeze(1).expand(-1, self.NT, -1)
                nonterm_emb = torch.cat([nonterm_emb, z], dim=-1)
            
            def estimate(mlp, branch):
                rule_prob = F.log_softmax(mlp(nonterm_emb), -1)
                rule_prob = rule_prob.view(*((B, self.NT, 2) + (self.NT_T,) * branch))
                return rule_prob
                
            rule_prob = estimate(self.rule_mlp, 2)
            return rule_prob
        
        def roots(x):
            root_emb = self.root_emb.expand(B, -1)
            if self.z_dim > 0:
                root_emb = torch.cat([root_emb, self.z], dim=-1)
            mlp = self.root_mlp
            root_prob = F.log_softmax(mlp(root_emb), -1)
            return root_prob
        
        def terms(x):
            term_emb = self.term_emb.unsqueeze(0).unsqueeze(1).expand(
                B, H * W, -1, -1
            )
            if self.z_dim > 0:
                z = self.z.unsqueeze(1).unsqueeze(2).expand(
                    -1, H * W, self.T, -1
                )
                term_emb = torch.cat([term_emb, z], dim=-1)
            mlp = self.term_mlp
            term_prob = term_prob_old = F.log_softmax(mlp(term_emb), -1)
            
            x = x.reshape(B, -1)
            indices = x.unsqueeze(-1).expand(-1, -1, self.T).unsqueeze(-1)
            term_prob = torch.gather(term_prob, -1, indices).squeeze(-1)
            return term_prob
        
        self.pcfgs = (rules(x), roots(x), terms(x))
        return self.pcfgs, kl, {}
    
    def forward(
        self, 
        t: Tensor,
        v: Tensor,
        mean: Tensor=None,
        lvar: Tensor=None,
        t_seq: Tensor=None, 
        v_seq: Tensor=None,
        memory: Tensor=None,
        memo_attn_mask: Tensor=None,
        memo_key_padding_mask: Tensor=None,
        require_marginal: bool=False,
        parallel: bool=True, 
        verbose: bool=False,
        infer: bool=False,
        **kwargs
    ):
        if self.z_dim > 0:
            assert mean is not None and lvar is not None, f"need latent z but it is None"

        x = v_seq
        pcfgs, kl, *_ = self.parameterize(x, mean, lvar, use_mean=infer)
        
        rule_prob, root_prob, term_prob = self.pcfgs
        
        if verbose:
            if isinstance(rule_prob, tuple):
                for x in rule_prob:
                    print(x.shape, torch.isnan(x).any())
            print(
                root_prob.shape, torch.isnan(root_prob).any(),
                term_prob.shape, torch.isnan(term_prob).any()
            )

        outs = self.partition(
            infer=infer, require_marginal=require_marginal, verbose=verbose, parallel=parallel, **kwargs
        ) # ll, argmax, marginal, {}
        outs = (outs[0], kl, x) + outs[1:]
        return outs
