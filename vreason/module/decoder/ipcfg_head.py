import numpy as np
import os, sys, time, math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

import torch.nn.functional as F

from . import MetaDecHead, DECODER_HEADS_REGISTRY
from .. import MAXVAL, InsideAlg1D, InsideAlg2D, InsideAlg2DLA

__all__ = ["IPCFG2DLADecHead", "IPCFG2DDecHead", "IPCFG1DDecHead"]

def mask_prob_schedule(
    H: int,
    W: int,
    beta: float = 1.,
    rate: float = 2.,
    contrast: bool = False,
):
    total = H * W
    c = 1 / total
    def schedule(h, w):
        count = h * w
        inv = max(1 / count, c)
        ratio = min(h, w) / max(h, w) if contrast else 0
        p = math.exp(-count / total - ratio) ** rate * (inv - c)
        return p * beta
    return schedule

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

class BaseIPCFGDecHead(MetaDecHead):
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, None)
        self.s_dim = cfg.s_dim
        self.z_dim = cfg.z_dim
        self.n_set = cfg.n_set
        
        ########
        self.nt_la = cfg.nt_la #nt_la # nonterminal w/ latent annotation

        T, NT = cfg.T, cfg.NT
        sli_np_h = slice(0, NT)
        sli_np_v = slice(NT, NT + NT)
        sli_tp_h = slice(NT + NT, NT + NT + T)
        sli_tp_v = slice(NT + NT + T, NT + NT + T + T)
        self.zeros = [
            (sli_np_h, sli_tp_v), (sli_tp_v, sli_np_h),
            (sli_np_v, sli_tp_h), (sli_tp_h, sli_np_v),
            (sli_tp_v, sli_tp_h), (sli_tp_h, sli_tp_v),
        ] if self.nt_la else []
                
        NT_, T_, NT_T_ = (NT, T, NT + T)
        NT_, T_, NT_T_ = (
            NT_ * 2, T_ * 2, NT_T_ * 2
        ) if self.nt_la else (NT_, T_, NT_T_)
        self.NT_, self.T_, self.NT_T_ = (NT_, T_, NT_T_)
        ########
        
        self.drop_1d_fn = mask_prob_schedule(
             1, 16, beta=cfg.beta_1d, rate=cfg.rate_1d, contrast=True,
        ) if cfg.drop_1d else None
        self.drop_2d_fn = mask_prob_schedule(
            16, 16, beta=cfg.beta_2d, rate=cfg.rate_2d, contrast=True,
        ) if cfg.drop_2d else None

        self.mini_1d_ll = cfg.mini_1d_ll # minimize 1d (row- col-wise) log-likelihood 
        self.mini_1d_2d = cfg.mini_1d_2d # minimize both 1d (row- col-wise) and 2d ll 

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


class ParamIPCFGDecHead(BaseIPCFGDecHead):
    def __init__(self, cfg, token_vocab, vis_token_vocab=None, **kwargs):
        super().__init__(cfg, None)
        h_dim = cfg.h_dim
        w_dim = cfg.w_dim
        z_dim = cfg.z_dim
        s_dim = cfg.s_dim

        self.T = cfg.T
        self.NT = cfg.NT
        self.NT_T = cfg.NT + cfg.T
        self.grid_size = cfg.grid_size
        self.vis_token_vocab = vis_token_vocab
        V = len(self.vis_token_vocab)
        self.V = V
        
        self.term_emb = nn.Parameter(torch.randn(self.T_, s_dim))
        self.nonterm_emb = nn.Parameter(torch.randn(self.NT, s_dim))
        self.root_emb = nn.Parameter(torch.randn(1, s_dim))
        
        rule_dim = s_dim + z_dim
        #rule_modules = (
        #    nn.Linear(rule_dim, self.NT_T ** 2 * 2),
        #) # horizon and vertical
        rule_modules = (
            nn.Linear(rule_dim, s_dim),
            ResLayer(s_dim, s_dim),
            ResLayer(s_dim, s_dim),
            nn.Linear(s_dim, self.NT_T_ ** 2 * self.n_set),
        ) # horizon and vertical
        self.rule_mlp = nn.Sequential(*rule_modules)
        
        root_dim = s_dim + z_dim
        root_modules = (
            nn.Linear(root_dim, s_dim),
            ResLayer(s_dim, s_dim),
            ResLayer(s_dim, s_dim),
            nn.Linear(s_dim, self.NT_)
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
        H, W = (int(np.sqrt(HW)),) * 2 if self.grid_size is None else self.grid_size
                
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
                if not self.nt_la:
                    rule_prob = F.log_softmax(mlp(nonterm_emb), -1)
                    rule_prob = rule_prob.view(*((B, self.NT, self.n_set) + (self.NT_T_,) * branch))
                else:
                    logits = mlp(nonterm_emb).view(*(
                        (B, self.NT, self.n_set) + (self.NT_T_,) * branch
                    ))
                    for a, b in self.zeros:
                        logits[..., a, b] = -MAXVAL
                    rule_prob = F.log_softmax(logits.view(*(B, self.NT, self.n_set, -1)), -1)
                    rule_prob = rule_prob.view(*((B, self.NT, self.n_set) + (self.NT_T_,) * branch))
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
                    -1, H * W, self.T_, -1
                )
                term_emb = torch.cat([term_emb, z], dim=-1)
            mlp = self.term_mlp
            term_prob = term_prob_old = F.log_softmax(mlp(term_emb), -1)
            
            x = x.reshape(B, -1)
            indices = x.unsqueeze(-1).expand(-1, -1, self.T_).unsqueeze(-1)
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
        
        if verbose:
            rule_prob, root_prob, term_prob = self.pcfgs
            if isinstance(rule_prob, tuple):
                for x in rule_prob:
                    print(x.shape, torch.isnan(x).any())
            print(
                root_prob.shape, torch.isnan(root_prob).any(),
                term_prob.shape, torch.isnan(term_prob).any()
            )

            print(
                (root_prob.exp() == 0).sum(),
                (rule_prob.exp() == 0).sum(),
                (term_prob.exp() == 0).sum()
            )
            
            if self.nt_la: 
                X_H_H, X_V_V, X_H_V, X_V_H, lr_rule_prob, ab_rule_prob = self._slice_rule_prob(rule_prob, self.NT, self.T)
                rule_p = (X_H_H, X_V_V, X_H_V, X_V_H) + lr_rule_prob[1:] + ab_rule_prob[1:]

                all_p = []
                for a, b in self.zeros:
                    p = rule_prob[..., a, b].exp().sum((-1, -2))
                    all_p.append(p)
                all_p = torch.stack(all_p, -1)
                #print(all_p.sum(-1), all_p.shape)
                #print((all_p == 0).all())

                all_p = all_p.sum(-1)
                print(torch.allclose(torch.zeros_like(all_p), all_p))

                all_p = []
                for x in rule_p:
                    p = x.exp().sum((-1, -2))
                    all_p.append(p)
                all_p = torch.stack(all_p, -1)
                #print(all_p.sum(-1), all_p.shape)
                #print((all_p == 1).all())

                all_p = all_p.sum(-1)
                print(torch.allclose(torch.ones_like(all_p), all_p))


        drop_1d_fn = self.drop_1d_fn if self.training else None
        drop_2d_fn = self.drop_2d_fn if self.training else None

        require_1d_ll = self.mini_1d_ll or self.mini_1d_2d

        outs = self.partition(
            infer=infer, parallel=parallel, require_marginal=require_marginal, require_1d_ll=require_1d_ll,
            shape=self.grid_size, drop_1d_fn=drop_1d_fn, drop_2d_fn=drop_2d_fn, verbose=verbose, **kwargs
        ) # ll, argmax, marginal, {}
        outs = (outs[0], kl, x) + outs[1:]
        return outs


@DECODER_HEADS_REGISTRY.register()
class IPCFG1DDecHead(ParamIPCFGDecHead, InsideAlg1D):
    def __init__(self, cfg, txt_token_vocab, vis_token_vocab=None, **kwargs):
        super().__init__(cfg, None, vis_token_vocab)

@DECODER_HEADS_REGISTRY.register()
class IPCFG2DDecHead(ParamIPCFGDecHead, InsideAlg2D):
    def __init__(self, cfg, txt_token_vocab, vis_token_vocab=None, **kwargs):
        super().__init__(cfg, None, vis_token_vocab)

@DECODER_HEADS_REGISTRY.register()
class IPCFG2DLADecHead(ParamIPCFGDecHead, InsideAlg2DLA):
    def __init__(self, cfg, txt_token_vocab, vis_token_vocab=None, **kwargs):
        super().__init__(cfg, None, vis_token_vocab)
        assert self.nt_la, f"self.nt_la ({self.nt_la}) must be true for nonterminal w/ annotations."

