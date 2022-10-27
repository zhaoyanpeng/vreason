import torch
import numpy as np
from torch import nn, Tensor

from vreason.util import stripe

MAXVAL = 1e9

__all__ = ["InsideAlg1D", "InsideAlg2D", "InsideAlg2DLA", "MAXVAL"]


class InsideAlg:
    def partition(self, infer=False, require_marginal=False, parallel=False, **kwargs):
        return self._dp(
            infer=infer, require_marginal=require_marginal, parallel=parallel, **kwargs
        )
    
    @torch.enable_grad()
    def argmax(self, infer=True, require_marginal=False, parallel=False, **kwargs):
        return self._dp(
            infer=infer, require_marginal=require_marginal, parallel=parallel, **kwargs
        )
    
    @torch.enable_grad()
    def _dp(self, parallel=False, **kwargs):
        if parallel:
            return self._dp_parallel(**kwargs)
        else:
            return self._dp_serial(**kwargs)
        
    def _compute_marginal(self, **kwargs):
        raise NotImplementedError

    def _extract_parse(self, **kwargs):
        raise NotImplementedError

    def _dp_parallel(self, **kwargs):
        raise NotImplementedError

    def _dp_serial(self, **kwargs):
        raise NotImplementedError
        

class InsideAlg1D(InsideAlg): # standalone class, for sequential inputs

    @staticmethod
    def _slice_rule_prob(rule_prob, NT, T):
        NT_T = NT + T
        sli_np = slice(0, NT)
        sli_tp = slice(NT, NT_T)
        
        X_Y_Z = rule_prob[:, :, sli_np, sli_np]
        X_Y_z = rule_prob[:, :, sli_np, sli_tp]
        X_y_Z = rule_prob[:, :, sli_tp, sli_np]
        X_y_z = rule_prob[:, :, sli_tp, sli_tp]
        return X_Y_Z, X_Y_z, X_y_Z, X_y_z

    @staticmethod
    def _inside_1d_serial(type_area, beta_area, rule_prob, term_prob, drop_1d_fn=None, verbose=False):
        B, W, _, NT = beta_area.shape
        Y_Z, Y_z, y_Z, y_z = rule_prob
        for w in range(2, W + 1):
            for x in range(W + 1 - w):
                beta_all = []
                for sx in range(1, w):
                    if sx > 1 and w - sx > 1:
                        sl = beta_area[:, x, x + sx]
                        sr = beta_area[:, x + sx, x + w]
                        rules = Y_Z
                    elif sx > 1 and w - sx == 1:
                        sl = beta_area[:, x, x + sx]
                        sr = term_prob[:, x + sx]
                        rules = Y_z
                    elif sx == 1 and w - sx > 1:
                        sl = term_prob[:, x]
                        sr = beta_area[:, x + sx, x + w]
                        rules = y_Z
                    elif sx == 1 and w - sx == 1:
                        sl = term_prob[:, x]
                        sr = term_prob[:, x + sx]
                        rules = y_z
                    else:
                        raise ValueError(f"unknown span types")
                    
                    beta = ( # bl, br, bslr
                        sl.unsqueeze(1).unsqueeze(-1) +
                        sr.unsqueeze(1).unsqueeze(1) +
                        rules
                    ).logsumexp((2, 3))

                    beta_all.append(beta)
                    
                beta = torch.stack(beta_all, dim=1).logsumexp(1)
                beta_area[:, x, x + w] = type_area[:, x, x + w] + beta
        return beta_area

    @staticmethod
    def _drop_beta(h, w, beta, drop_prob_fn, device):
        p_drop = drop_prob_fn(h, w) 
        if p_drop > 0:
            p_mask = torch.full(beta.shape, p_drop, device=device).bernoulli().bool()
#            print((h, w), p_drop, p_mask.sum().cpu().numpy(), p_mask.numel())
            beta = beta.masked_fill(p_mask, 0.) 
        return beta

    @staticmethod
    def _inside_1d_parallel(type_area, beta_area, rule_prob, term_prob, drop_1d_fn=None, verbose=False):
        device = beta_area.device
        B, W, _, NT = beta_area.shape
        Y_Z, Y_z, y_Z, y_z = rule_prob
        
        def xyz(sl, sr, rule_prob):   
            beta = ( # bkr, bkr, bslr
                sl.unsqueeze(2).unsqueeze(-1) + # (B, k, 1, r, 1)
                sr.unsqueeze(2).unsqueeze(-2) + # (B, k, 1, 1, r)
                rule_prob.unsqueeze(1) # (B, 1, S, r, r)
            ).logsumexp((3, 4))
            return beta
        
        # special case: span length is 2
        sl = term_prob[:, :W - 1]
        sr = term_prob[:, 1:]
        beta = xyz(sl, sr, y_z)
        # drop some betas / areas
        beta = beta if drop_1d_fn is None else (
            InsideAlg1D._drop_beta(1, 2, beta, drop_1d_fn, device)
        )
        indice = torch.arange(W - 1).to(device)
        beta_area[:, indice, indice + 2] = beta[:, indice] + type_area[:, indice, indice + 2]
        
        for w in range(3, W + 1):
            k = w - 1
            indice = torch.arange(W - k).to(device)
            
            y_term = term_prob[:, :W - k] # (B, k, r)
            z_term = term_prob[:, w - 1:] # (B, k, r)
            
            Y = stripe(beta_area, W - k, w - 1, (0, 1), 1) # (B, k, s, r)
            Z = stripe(beta_area, W - k, w - 1, (1, w), 0) # (B, k, s, r)
                
            # case 1: y_Z
            sl = y_term
            sr = Z[..., 0, :]
            beta1 = xyz(sl, sr, y_Z).unsqueeze(-1)

            # case 2: Y_z
            sl = Y[..., -1, :] # (B, k, r)
            sr = z_term     # (B, k, r)
            beta2 = xyz(sl, sr, Y_z).unsqueeze(-1) # (B, k, r)
            
            # case 3: Y_Z
            if k > 2:
                sl = Y[..., 1:-1, :].reshape(B, -1, NT)
                sr = Z[..., 1:-1, :].reshape(B, -1, NT)
                beta3 = xyz(sl, sr, Y_Z).view(
                    B, -1, k - 2, NT
                ).logsumexp(-2).unsqueeze(-1)
            else:
                beta3 = beta1[..., :0] # empty tensor
            
            beta = torch.cat((beta1, beta2, beta3), dim=-1).logsumexp(-1)
            # drop some betas / areas
            beta = beta if drop_1d_fn is None else (
                InsideAlg1D._drop_beta(1, w, beta, drop_1d_fn, device)
            )
            beta_area[:, indice, indice + w] = beta[:, indice] + type_area[:, indice, indice + w]
        return beta_area
    
    @staticmethod
    @torch.no_grad()
    def _viterbi_1d_manual(best_area, best_type, beta_area, rule_prob, term_prob, verbose=False):
        device = beta_area.device
        B, W, _, NT = beta_area.shape
        Y_Z, Y_z, y_Z, y_z = rule_prob
        
        def xyz(sl, sr, rule_prob):   
            beta = ( # bkr, bkr, bslr
                sl.unsqueeze(2).unsqueeze(-1) + # (B, k, 1, r, 1)
                sr.unsqueeze(2).unsqueeze(-2) + # (B, k, 1, 1, r)
                rule_prob.unsqueeze(1) # (B, 1, S, r, r)
            )
            b, k, s, l, r = beta.shape
            beta, best_lr = beta.view((b, k, s, -1)).max(-1)
            best_l = best_lr.div(r, rounding_mode='floor') # // r # best left symbol
            best_r = best_lr  % r # best right symbol
            best_lr = (best_l << 16) + best_r
            return beta, best_lr
        
        # special case: span length is 2
        sl = term_prob[:, :W - 1]
        sr = term_prob[:, 1:]
        beta, kind = xyz(sl, sr, y_z)
        indice = torch.arange(W - 1).to(device)
        beta_area[:, indice, indice + 2] = beta[:, indice]
        best_type[:, indice, indice + 2] = kind[:, indice]
        best_area[:, indice, indice + 2] = 1
        
        for w in range(3, W + 1):
            k = w - 1
            indice = torch.arange(W - k).to(device)
            
            y_term = term_prob[:, :W - k] # (B, k, r)
            z_term = term_prob[:, w - 1:] # (B, k, r)
            
            Y = stripe(beta_area, W - k, w - 1, (0, 1), 1) # (B, k, s, r)
            Z = stripe(beta_area, W - k, w - 1, (1, w), 0) # (B, k, s, r)
                
            # case 1: y_Z
            sl = y_term
            sr = Z[..., 0, :]
            beta1, type1 = xyz(sl, sr, y_Z)
            beta1 = beta1.unsqueeze(-1)
            type1 = type1.unsqueeze(-1)
            # split point
            best1 = torch.empty_like(beta1).fill_(1).long()

            # case 2: Y_z
            sl = Y[..., -1, :] # (B, k, r)
            sr = z_term     # (B, k, r)
            beta2, type2 = xyz(sl, sr, Y_z)
            beta2 = beta2.unsqueeze(-1)
            type2 = type2.unsqueeze(-1)
            # split point
            best2 = torch.empty_like(beta2).fill_(k).long()
            
            # case 3: Y_Z
            if k > 2:
                sl = Y[..., 1:-1, :].reshape(B, -1, NT)
                sr = Z[..., 1:-1, :].reshape(B, -1, NT)
                beta3, type3 = xyz(sl, sr, Y_Z)
                beta3, best3 = beta3.view(
                    B, -1, k - 2, NT
                ).max(-2)
                type3 = type3.view(B, -1, k - 2, NT)
                type3 = type3.gather(
                    -2, best3.unsqueeze(-2)
                ).squeeze(-2)
                beta3 = beta3.unsqueeze(-1)
                type3 = type3.unsqueeze(-1)
                # split point
                best3 = best3.unsqueeze(-1) + 2
            else: # empty tensor
                beta3 = type3 = beta1[..., :0]
                type3 = best3 = type3.long()
            
            beta, k0 = torch.cat((beta1, beta2, beta3), dim=-1).max(-1)
            kind = torch.cat( # k0 \in {0, 1, 2} -> real best indice
                (type1, type2, type3), dim=-1
            ).gather(-1, k0.unsqueeze(-1)).squeeze(-1)
            best = torch.cat( # k0 \in {0, 1, 2} -> real best indice
                (best1, best2, best3), dim=-1
            ).gather(-1, k0.unsqueeze(-1)).squeeze(-1)
            beta_area[:, indice, indice + w] = beta[:, indice]
            best_type[:, indice, indice + w] = kind[:, indice]
            best_area[:, indice, indice + w] = best[:, indice]
        return beta_area, best_area, best_type    
    
    @staticmethod
    @torch.enable_grad()
    def _viterbi_1d_auto(type_area, beta_area, rule_prob, term_prob, verbose=False):
        device = beta_area.device
        B, W, _, NT = beta_area.shape
        Y_Z, Y_z, y_Z, y_z = rule_prob
        
        def xyz(sl, sr, rule_prob):   
            beta = ( # bkr, bkr, bslr
                sl.unsqueeze(2).unsqueeze(-1) + # (B, k, 1, r, 1)
                sr.unsqueeze(2).unsqueeze(-2) + # (B, k, 1, 1, r)
                rule_prob.unsqueeze(1) # (B, 1, S, r, r)
            ).flatten(-2, -1).max(-1)[0] #.amax((3, 4))
            return beta
        
        # special case: span length is 2
        sl = term_prob[:, :W - 1]
        sr = term_prob[:, 1:]
        beta = xyz(sl, sr, y_z)
        indice = torch.arange(W - 1).to(device)
        beta_area[:, indice, indice + 2] = beta[:, indice] + type_area[:, indice, indice + 2]
        
        for w in range(3, W + 1):
            k = w - 1
            indice = torch.arange(W - k).to(device)
            
            y_term = term_prob[:, :W - k] # (B, k, r)
            z_term = term_prob[:, w - 1:] # (B, k, r)
            
            Y = stripe(beta_area, W - k, w - 1, (0, 1), 1) # (B, k, s, r)
            Z = stripe(beta_area, W - k, w - 1, (1, w), 0) # (B, k, s, r)
                
            # case 1: y_Z
            sl = y_term
            sr = Z[..., 0, :]
            beta1 = xyz(sl, sr, y_Z).unsqueeze(-1)

            # case 2: Y_z
            sl = Y[..., -1, :] # (B, k, r)
            sr = z_term     # (B, k, r)
            beta2 = xyz(sl, sr, Y_z).unsqueeze(-1) # (B, k, r)
            
            # case 3: Y_Z
            if k > 2:
                sl = Y[..., 1:-1, :].reshape(B, -1, NT)
                sr = Z[..., 1:-1, :].reshape(B, -1, NT)
                beta3 = xyz(sl, sr, Y_Z).view(
                    B, -1, k - 2, NT
                ).max(-2)[0].unsqueeze(-1)
            else:
                beta3 = beta1[..., :0] # empty tensor
            
            beta = torch.cat((beta1, beta2, beta3), dim=-1).max(-1)[0]
            beta_area[:, indice, indice + w] = beta[:, indice] + type_area[:, indice, indice + w]
        return beta_area        

    @torch.no_grad()
    def _mbr_1d_manual(self, best_area, beta_area, span_area, verbose=False):
        device = span_area.device
        B, W, *_ = span_area.shape
        # span of length 1
        indice = torch.arange(W).to(device)
        beta_area[:, indice, indice + 1] = span_area[:, indice, indice + 1]
        
        for w in range(2, W + 1):
            k = w - 1
            indice = torch.arange(W - k).to(device)
            Y = stripe(beta_area, W - k, w - 1, (0, 1), 1) # (B, k, s)
            Z = stripe(beta_area, W - k, w - 1, (1, w), 0) # (B, k, s)
            X, k = (Y + Z).max(2) # (B, k, w - 1)
            X = X + span_area[:, indice, indice + w]
            beta_area[:, indice, indice + w] = X
            best_area[:, indice, indice + w] = k + 1
        return beta_area, best_area

    @torch.enable_grad()
    def _mbr_1d_auto(self, type_area, beta_area, span_area, verbose=False):
        device = span_area.device
        B, W, *_ = span_area.shape
        # span of length 1
        indice = torch.arange(W).to(device)
        beta_area[:, indice, indice + 1] = (
            span_area[:, indice, indice + 1] + type_area[:, indice, indice + 1]
        )
        
        for w in range(2, W + 1):
            k = w - 1
            indice = torch.arange(W - k).to(device)
            Y = stripe(beta_area, W - k, w - 1, (0, 1), 1) # (B, k, s)
            Z = stripe(beta_area, W - k, w - 1, (1, w), 0) # (B, k, s)
            X, *_ = (Y + Z).max(2) # (B, k, w - 1); note that max() produces sub-gradients
            X = X + span_area[:, indice, indice + w]
            beta_area[:, indice, indice + w] = X + type_area[:, indice, indice + w]
        return beta_area
    
    @torch.enable_grad()
    def _viterbi_1d_proxy_auto(self, infer=True, require_marginal=False, verbose=False, **kwargs):
        rule_prob, root_prob, term_prob = self.pcfgs
        
        device = rule_prob.device
        NT = rule_prob.shape[1]
        B, L, T = term_prob.shape[:3]
        H = W = L
        
        term_prob = term_prob.view(B, W, -1)
        
        lr_Y_Z, *_ = lr_rule_prob = self._slice_rule_prob(rule_prob[:, :, 0], NT, T) # left-right rules
        
        beta_area = torch.zeros( # the real location of inside scores
            (B, W, W + 1, NT), device=device
        ).fill_(-MAXVAL)
        
        type_area = torch.zeros( # to record grad-able operations
            (B, W, W + 1, NT), device=device
        ).requires_grad_(infer or require_marginal)
        
        #####
        sub_type_area = type_area # (B, W, W + 1, NT)
        sub_beta_area = beta_area.clone() # (B, W, W + 1, NT)
        new_beta_area = self._viterbi_1d_auto(
            sub_type_area, sub_beta_area, lr_rule_prob, term_prob, verbose=False,
        )
        beta_area = new_beta_area #+ sub_type_area
        #####
        
        final = beta_area[:, 0, W] + root_prob
        # final = beta_area[:, 0, H, 0, W]
        ll = final.max(-1)[0]
    
        def _extract_parse(ll, areas, exclude_trivial=False, **kwargs):
            marginals = torch.autograd.grad(
                ll.sum(), areas, create_graph=True, only_inputs=True, allow_unused=False
            )[0]
            if marginals.dim() == 4: # only unlabled eval is supported
                marginals = marginals.sum(-1)
            best_area = marginals.nonzero() # (B, X1, X2)
            best_beta = marginals[
                best_area[:, 0], best_area[:, 1], best_area[:, 2]
            ]
            if exclude_trivial:
                w1 = (best_area[:, 2] - best_area[:, 1]) == 1
                h1w1 = w1
                best_area = best_area[~h1w1]
                best_beta = best_beta[~h1w1]

    #         print(torch.cat((best_area, best_beta.unsqueeze(-1)), dim=-1))

            best_area = best_area[:, 1:] # trim the batch dim
            best_area = best_area.view(B, -1, best_area.shape[-1])
            parses = [best_area[i].cpu().tolist() for i in range(B)]        
            return parses 
        
        argmax = (
            None if not infer else _extract_parse(ll, type_area, **kwargs)
        )
        
        if not require_marginal:
            return ll, argmax, None, {"argmax": argmax, "marginal": None}
        
        marginal = self._compute_marginal(ll, type_area, **kwargs)
        return ll, argmax, marginal, {"argmax": argmax, "marginal": marginal}
    
    @torch.no_grad()
    def _viterbi_1d_proxy_manual(self, infer=True, require_marginal=False, verbose=False, **kwargs):
        rule_prob, root_prob, term_prob = self.pcfgs
        
        device = rule_prob.device
        NT = rule_prob.shape[1]
        B, L, T = term_prob.shape[:3]
        H = W = L
        
        term_prob = term_prob.view(B, W, -1)
        
        lr_Y_Z, *_ = lr_rule_prob = self._slice_rule_prob(rule_prob[:, :, 0], NT, T) # left-right rules
        
        beta_area = torch.zeros( # the real location of inside scores
            (B, W, W + 1, NT), device=device
        ).fill_(-MAXVAL)
        # TODO best_area and best_type can be merged by using bitwise shift
        # For example, 10 bits for split points, 10 for best left symbols, and 10 for best right symbols
        best_area = torch.zeros( # split point for every best symbol
            (B, W, W + 1, NT), device=device
        ).long()
        
        best_type = torch.zeros( # left-right / bottom-up best symbol
            (B, W, W + 1, NT), device=device
        ).long()
        
        #####
        sub_best_area = best_area # (B, W, W + 1, NT)
        sub_best_type = best_type
        sub_beta_area = beta_area.clone() # (B, W, W + 1, NT)
        new_beta_area, new_best_area, new_best_type = self._viterbi_1d_manual(
            sub_best_area, sub_best_type, sub_beta_area, lr_rule_prob, term_prob, verbose=(y == 0),
        )
        beta_area = new_beta_area #+ sub_type_area
        best_type = new_best_type
        best_area = new_best_area
        #####
        final = beta_area[:, 0, W] + root_prob
        # final = beta_area[:, 0, H, 0, W]
        ll, s = final.max(-1) # s: best type
        
        def _extract_parse(ll, areas, types, exclude_trivial=False, **kwargs):

            def backtrack(best, kind, x, w, s):
                if w == 1: # s may be a pre-terminal and will be skipped by this condition, or exceptions will be raised (s >= nt).
                    return [] if exclude_trivial else [(x, x + w)]
                k = best[x, x + w, s].cpu().item()
                z = kind[x, x + w, s].cpu().item()
                l = z >> 16
                r = z & ((1 << 16) - 1)
                if k > 0: # left-right composition
                    area1 = backtrack(best, kind, x, k, l)
                    area2 = backtrack(best, kind, x + k, w - k, r)
                return area1 + area2 + [(x, x + w)]

            parses = [
                backtrack(
                    areas[i], types[i], 0, W, s[i].cpu().item()
                ) for i in range(B)
            ]
            return parses
        
        argmax = (
            None if not infer else _extract_parse(ll, best_area, best_type, **kwargs)
        )
        
        if not require_marginal:
            return ll, argmax, None, {"argmax": argmax, "marginal": None}
        
        marginal = self._compute_marginal(ll, type_area, **kwargs)
        return ll, argmax, marginal, {"argmax": argmax, "marginal": marginal}
    
    @torch.enable_grad()
    def _mbr_1d_proxy_auto(self, areas, exclude_trivial=False, verbose=False, **kwargs):
        device = areas.device
        B, W, *_ = areas.shape # (B, W, W + 1)
        beta_area = torch.zeros_like(areas).fill_(-MAXVAL)
        type_area = torch.zeros_like(areas).requires_grad_(True)
        
        #####
        sub_span_area = areas
        sub_type_area = type_area # (B, W, W + 1)
        sub_beta_area = beta_area.clone() # (B, W, W + 1)
        new_beta_area = self._mbr_1d_auto(
            sub_type_area, sub_beta_area, sub_span_area, verbose=verbose
        )
        beta_area = new_beta_area #+ sub_type_area
        #####
      
        def _extract_parse(ll, areas, **kwargs):
            marginals = torch.autograd.grad(
                ll.sum(), areas, create_graph=True, only_inputs=True, allow_unused=False
            )[0]
            best_area = marginals.nonzero() # (B, X1, X2)
            best_beta = marginals[
                best_area[:, 0], best_area[:, 1], best_area[:, 2]
            ]
            if exclude_trivial:
                w1 = (best_area[:, 2] - best_area[:, 1]) == 1
                h1w1 = w1
                best_area = best_area[~h1w1]
                best_beta = best_beta[~h1w1]

    #         print(torch.cat((best_area, best_beta.unsqueeze(-1)), dim=-1))

            best_area = best_area[:, 1:] # trim the batch dim
            best_area = best_area.view(B, -1, best_area.shape[-1])
            parses = [best_area[i].cpu().tolist() for i in range(B)]        
            return parses
        ll = beta_area[:, 0, W]
        return _extract_parse(ll, type_area, **kwargs)
    
    @torch.no_grad()
    def _mbr_1d_proxy_manual(self, areas, exclude_trivial=False, verbose=False, **kwargs):
        device = areas.device
        B, W, *_ = areas.shape # (B, H, H + 1, W, W + 1)
        beta_area = torch.zeros_like(areas).fill_(-MAXVAL)
        best_area = torch.zeros_like(areas).long()
        
        #####
        sub_span_area = areas
        sub_best_area = best_area # (B, W, W + 1)
        sub_beta_area = beta_area.clone() # (B, W, W + 1)
        new_beta_area, new_best_area = self._mbr_1d_manual(
            sub_best_area, sub_beta_area, sub_span_area, verbose=verbose
        )
        beta_area = new_beta_area #+ sub_type_area
        best_area = new_best_area
        #####
        
        def _extract_parse(best_area, **kwargs):
            def backtrack(best, x, w):
                if w == 1:
                    return [] if exclude_trivial else [(x, x + w)]
                k = best[x, x + w].cpu().item()
                area1 = backtrack(best, x, k)
                area2 = backtrack(best, x + k, w - k)
                return area1 + area2 + [(x, x + w)]

            parses = [
                backtrack(best_area[i], 0, W) for i in range(B)
            ]
            return parses
        return _extract_parse(best_area, **kwargs)
    
    def _compute_marginal(self, ll, areas, marginal_as_dict=False, **kwargs):
        marginals = torch.autograd.grad(
            ll.sum(), areas, create_graph=True, only_inputs=True, allow_unused=False
        )[0]
        if not marginal_as_dict:
            return marginals
        # linearization
        scores = dict()
        device = marginals.device
        B, W, *_ = marginals.shape
        for w in range(2, W + 1):
            x = torch.arange(W + 1 - w).to(device)
            scores[(w,)] = marginals[:, x, x + w]
        return scores

    def _extract_parse(self, ll, areas, mbr=True, auto_infer=False, **kwargs):
        marginals = torch.autograd.grad(
            ll.sum(), areas, create_graph=True, only_inputs=True, allow_unused=False
        )[0]
        if marginals.dim() == 4: # only unlabled eval is supported
            marginals = marginals.sum(-1)
        if mbr:
            mbr_fn = (
                self._mbr_1d_proxy_auto
                if auto_infer else
                self._mbr_1d_proxy_manual
            )
            return mbr_fn(marginals.detach(), **kwargs)
        # will need max operator in _dp() but have not been implemented
        raise NotImplementedError
    
    def _dp_parallel(
        self, infer=False, require_marginal=False, verbose=False,
        drop_1d_fn=None, **kwargs
    ):
        if infer and not kwargs.get("mbr", False):
            return self._viterbi_1d_proxy_auto(
                require_marginal=require_marginal, verbose=verbose, **kwargs
            ) if kwargs.get("auto_infer", True) else self._viterbi_1d_proxy_manual(
                require_marginal=require_marginal, verbose=verbose, **kwargs
            ) # argmax version of _dp_parallel(...)
        
        return self._dp_serial(
            infer=infer, require_marginal=require_marginal, verbose=verbose, 
            drop_1d_fn=drop_1d_fn, parallel=True, **kwargs,
        ) # added a new parameter: parallel
    
    def _dp_serial(
        self, infer=False, require_marginal=False, verbose=False,
        drop_1d_fn=None, parallel=False, **kwargs
    ):
        rule_prob, root_prob, term_prob = self.pcfgs
        
        device = rule_prob.device
        NT = rule_prob.shape[1]
        B, L, T = term_prob.shape[:3]
        H = W = L
        
        term_prob = term_prob.view(B, W, -1)
        
        lr_Y_Z, *_ = lr_rule_prob = self._slice_rule_prob(rule_prob[:, :, 0], NT, T) # left-right rules
        
        beta_area = torch.zeros( # the real location of inside scores
            (B, W, W + 1, NT), device=device
        ).fill_(-MAXVAL)
        
        type_area = torch.zeros( # to record grad-able operations
            (B, W, W + 1, NT), device=device
        ).requires_grad_(infer or require_marginal)
        
        #####
        sub_type_area = type_area # (B, W, W + 1, NT)
        sub_beta_area = beta_area.clone() # (B, W, W + 1, NT)
        new_beta_area = self._inside_1d_parallel(
            sub_type_area, sub_beta_area, lr_rule_prob, term_prob, drop_1d_fn=drop_1d_fn, verbose=False,
        ) if parallel else self._inside_1d_serial(
            sub_type_area, sub_beta_area, lr_rule_prob, term_prob, drop_1d_fn=drop_1d_fn, verbose=False
        )
        beta_area = new_beta_area #+ sub_type_area
        #####
        
        final = beta_area[:, 0, W] + root_prob
        # final = beta_area[:, 0, H, 0, W]
        ll = final.logsumexp(-1)
        
        argmax = (
            None if not infer else self._extract_parse(ll, type_area, **kwargs)
        )
        
        if not require_marginal:
            return ll, argmax, None, {"argmax": argmax, "marginal": None}
        
        marginal = self._compute_marginal(ll, type_area, **kwargs)
        return ll, argmax, marginal, {"argmax": argmax, "marginal": marginal}

class InsideAlg2D(InsideAlg1D): # standalone class, for two-dimensional inputs

    def _compute_marginal(self, ll, areas, marginal_as_dict=False, **kwargs):
        marginals = torch.autograd.grad(
            ll.sum(), areas, create_graph=True, only_inputs=True, allow_unused=False
        )[0]
        if not marginal_as_dict:
            return marginals
        # linearization
        scores = dict()
        device = marginals.device
        B, H, _, W, *_ = marginals.shape
        for h in range(2, H + 1):
            for w in range(2, W + 1):
                y = torch.arange(H + 1 - h).repeat_interleave(W + 1 - w).to(device)
                x = torch.arange(W + 1 - w).repeat(H + 1 - h).to(device)
                scores[(h, w)] = marginals[:, y, y + h, x, x + w]
        return scores

    def _extract_parse(self, ll, areas, mbr=True, auto_infer=False, **kwargs):
        marginals = torch.autograd.grad(
            ll.sum(), areas, create_graph=True, only_inputs=True, allow_unused=False
        )[0]
        if marginals.dim() == 6: # only unlabled eval is supported
            marginals = marginals.sum(-1)
        if mbr:
            mbr_fn = self._mbr_2d_auto if auto_infer else self._mbr_2d_manual
            return mbr_fn(marginals.detach(), **kwargs)
        # will need max operator in _dp() but have not been implemented
        raise NotImplementedError

    @torch.enable_grad()
    def _mbr_2d_auto(self, areas, exclude_trivial=False, verbose=False, **kwargs):
        device = areas.device
        B, H, _, W, *_ = areas.shape # (B, H, H + 1, W, W + 1)
        beta_area = torch.zeros_like(areas).fill_(-MAXVAL)
        type_area = torch.zeros_like(areas).requires_grad_(True)
        
        for y in range(H): # area 1 x w
            if W < 2: continue
            sub_span_area = areas[:, y, y + 1]
            sub_type_area = type_area[:, y, y + 1] # (B, W, W + 1)
            sub_beta_area = beta_area[:, y, y + 1].clone() # (B, W, W + 1)
            new_beta_area = self._mbr_1d_auto(
                sub_type_area, sub_beta_area, sub_span_area, verbose=verbose
            )
            beta_area[:, y, y + 1] = new_beta_area #+ sub_type_area
        
        for x in range(W): # area h x 1
            if H < 2: continue
            sub_span_area = areas[..., x, x + 1]
            sub_type_area = type_area[..., x, x + 1]
            sub_beta_area = beta_area[..., x, x + 1]
            new_beta_area = self._mbr_1d_auto(
                sub_type_area, sub_beta_area, sub_span_area, verbose=verbose
            )
            beta_area[..., x, x + 1] = new_beta_area #+ sub_type_area
            
        # area >= 2 x 2
        for h in range(2, H + 1):
            for w in range(2, W + 1):
                if verbose:
                    print(f"h {h} w {w}")
                
                y = torch.arange(H + 1 - h).to(device)
                x = torch.arange(W + 1 - w).to(device)
                   
                # case 1: left-right composition
                sub_beta_area = beta_area[:, y, y + h].clone().permute(0, 2, 3, 1)
                sl = stripe(sub_beta_area, W + 1 - w, w - 1, (0, 1), 1) # (B, kx, s, y)
                sr = stripe(sub_beta_area, W + 1 - w, w - 1, (1, w), 0) # (B, kx, s, y)
#                 beta1, k1 = (sl + sr).permute(0, 3, 1, 2).max(-1) # (B, y, kx, w - 1)
                beta1 = (sl + sr).permute(0, 3, 1, 2)
                
                # case 2: above-below composition
                sub_beta_area = beta_area[..., x, x + w].clone()
                sa = stripe(sub_beta_area, H + 1 - h, h - 1, (0, 1), 1) # (B, ky, s, x)
                sb = stripe(sub_beta_area, H + 1 - h, h - 1, (1, h), 0) # (B, ky, s, x)
#                 beta2, k2 = (sa + sb).permute(0, 1, 3, 2).max(-1) # (B, ky, x, h - 1)
                beta2 = (sa + sb).permute(0, 1, 3, 2)
    
                # record the best
                beta, *_ = torch.cat((beta1, beta2), dim=-1).max(-1) # (B, y, x, w - 1 + h - 1)
                
                y = y.repeat_interleave(W + 1 - w)
                x = x.repeat(H + 1 - h)
                beta_area[:, y, y + h, x, x + w] = (
                    beta[:, y, x] + areas[:, y, y + h, x, x + w] + type_area[:, y, y + h, x, x + w]
                )
        
        ll = beta_area[:, 0, H, 0, W]
        
        marginals = torch.autograd.grad(
            ll.sum(), type_area, create_graph=True, only_inputs=True, allow_unused=False
        )[0]
        
        marginals = marginals.permute(0, 1, 3, 2, 4) # (B, Y1, X1, Y2, X2)
        best_area = marginals.nonzero()
        best_beta = marginals[
            best_area[:, 0], best_area[:, 1], best_area[:, 2], best_area[:, 3], best_area[:, 4]
        ]
        if exclude_trivial:
            h1 = (best_area[:, 3] - best_area[:, 1]) == 1
            w1 = (best_area[:, 4] - best_area[:, 2]) == 1
            h1w1 = h1 & w1
            best_area = best_area[~h1w1]
            best_beta = best_beta[~h1w1]
        
#         print(torch.cat((best_area, best_beta.unsqueeze(-1)), dim=-1))
        
        best_area = best_area[:, 1:] # trim the batch dim
        best_area = best_area.view(B, -1, best_area.shape[-1])
        parses = [best_area[i].cpu().tolist() for i in range(B)]        
        return parses

    @torch.no_grad()
    def _mbr_2d_manual(self, areas, exclude_trivial=False, verbose=False, **kwargs):
        # TODO make it grad-able and extract the best parse via back-propagation
        device = areas.device
        B, H, _, W, *_ = areas.shape # (B, H, H + 1, W, W + 1)
        beta_area = torch.zeros_like(areas).fill_(-MAXVAL)
        best_area = torch.zeros_like(areas).long()
        
        for y in range(H): # area 1 x w
            if W < 2: continue
            sub_span_area = areas[:, y, y + 1]
            sub_best_area = best_area[:, y, y + 1] # (B, W, W + 1)
            sub_beta_area = beta_area[:, y, y + 1].clone() # (B, W, W + 1)
            new_beta_area, new_best_area = self._mbr_1d_manual(
                sub_best_area, sub_beta_area, sub_span_area, verbose=verbose
            )
            beta_area[:, y, y + 1] = new_beta_area #+ sub_type_area
            best_area[:, y, y + 1] = new_best_area
        
        for x in range(W): # area h x 1
            if H < 2: continue
            sub_span_area = areas[..., x, x + 1]
            sub_best_area = best_area[..., x, x + 1]
            sub_beta_area = beta_area[..., x, x + 1].clone()
            new_beta_area, new_best_area = self._mbr_1d_manual(
                sub_best_area, sub_beta_area, sub_span_area, verbose=verbose
            )
            beta_area[..., x, x + 1] = new_beta_area #+ sub_type_area
            best_area[..., x, x + 1] = new_best_area * -1  
        
        # area >= 2 x 2
        for h in range(2, H + 1):
            for w in range(2, W + 1):
                if verbose:
                    print(f"h {h} w {w}")
                
                y = torch.arange(H + 1 - h).to(device)
                x = torch.arange(W + 1 - w).to(device)
                   
                # case 1: left-right composition
                sub_beta_area = beta_area[:, y, y + h].clone().permute(0, 2, 3, 1)
                sl = stripe(sub_beta_area, W + 1 - w, w - 1, (0, 1), 1) # (B, kx, s, y)
                sr = stripe(sub_beta_area, W + 1 - w, w - 1, (1, w), 0) # (B, kx, s, y)
#                 beta1, k1 = (sl + sr).permute(0, 3, 1, 2).max(-1) # (B, y, kx, w - 1)
                beta1 = (sl + sr).permute(0, 3, 1, 2)
                
                # case 2: above-below composition
                sub_beta_area = beta_area[..., x, x + w].clone()
                sa = stripe(sub_beta_area, H + 1 - h, h - 1, (0, 1), 1) # (B, ky, s, x)
                sb = stripe(sub_beta_area, H + 1 - h, h - 1, (1, h), 0) # (B, ky, s, x)
#                 beta2, k2 = (sa + sb).permute(0, 1, 3, 2).max(-1) # (B, ky, x, h - 1)
                beta2 = (sa + sb).permute(0, 1, 3, 2)
    
                # oh, gotta record which case has been selected; the trick is to use signed split!
    
                # record the best
                topk = w - 1
                beta, k0 = torch.cat((beta1, beta2), dim=-1).max(-1) # (B, y, x, w - 1 + h - 1)
                """
                k1 = (k0 < topk).long() + (k0 >= topk) * -1 # which case: horion (1) & vertical (-1)
                k2 = (k0 % topk).long() + 1 # which split; what if k0 is K (>1) times topk
                best = k1 * k2 # positive split for left-right composition and negative for above-below
                """
                vi = k0 >= topk # vertical
                k0 = k0 + 1
                k0[vi] = (k0[vi] - topk) * -1
                best = k0
                
                y = y.repeat_interleave(W + 1 - w)
                x = x.repeat(H + 1 - h)
                beta_area[:, y, y + h, x, x + w] = beta[:, y, x] + areas[:, y, y + h, x, x + w]
                best_area[:, y, y + h, x, x + w] = best[:, y, x]
        
        def backtrack(best, y, x, h, w):
            if h == 1 and w == 1:
                return [] if exclude_trivial else [(y, x, y + h, x + w)]
            k = best[y, y + h, x, x + w].cpu().item()
            if k > 0: # left-right composition
                area1 = backtrack(best, y, x, h, k)
                area2 = backtrack(best, y, x + k, h, w - k)
            else:
                k = -k # above-below composition
                area1 = backtrack(best, y, x, k, w)
                area2 = backtrack(best, y + k, x, h - k, w)
            return area1 + area2 + [(y, x, y + h, x + w)]
        
        parses = [
            backtrack(best_area[i], 0, 0, H, W) for i in range(B)
        ]
        return parses
    
    @torch.enable_grad()
    def _viterbi_2d_auto(self, infer=True, require_marginal=False, verbose=False, shape=None, **kwargs):
        rule_prob, root_prob, term_prob = self.pcfgs
        
        device = rule_prob.device
        NT = rule_prob.shape[1]
        B, L, T = term_prob.shape[:3]
        H, W = (int(np.sqrt(L)),) * 2 if shape is None else shape[:2]
        
        term_prob = term_prob.view(B, H, W, -1)
        
        lr_Y_Z, *_ = lr_rule_prob = self._slice_rule_prob(rule_prob[:, :, 0], NT, T) # left-right rules
        ab_Y_Z, *_ = ab_rule_prob = self._slice_rule_prob(rule_prob[:, :, 1], NT, T) # above-below rules
        
        beta_area = torch.zeros( # the real location of inside scores
            (B, H, H + 1, W, W + 1, NT), device=device
        ).fill_(-MAXVAL)
        
        type_area = torch.zeros( # to record grad-able operations
            (B, H, H + 1, W, W + 1, NT), device=device
        ).requires_grad_(infer or require_marginal)
        
        for y in range(H): # area 1 x w
            if W < 2: continue
            sub_type_area = type_area[:, y, y + 1] # (B, W, W + 1, NT)
            sub_beta_area = beta_area[:, y, y + 1].clone() # (B, W, W + 1, NT)
            new_beta_area = self._viterbi_1d_auto(
                sub_type_area, sub_beta_area, lr_rule_prob, term_prob[:, y], verbose=False,
            )
            beta_area[:, y, y + 1] = new_beta_area #+ sub_type_area
        
        for x in range(W): # area h x 1
            if H < 2: continue
            sub_type_area = type_area[..., x, x + 1, :]
            sub_beta_area = beta_area[..., x, x + 1, :].clone()
            new_beta_area = self._viterbi_1d_auto(
                sub_type_area, sub_beta_area, ab_rule_prob, term_prob[..., x, :],
            )
            beta_area[..., x, x + 1, :] = new_beta_area #+ sub_type_area
        
        def xyz(sl, sr, rule_prob):
            beta = ( # bkr, bkr, bslr
                sl.unsqueeze(1).unsqueeze(-2) + # (B, 1, k, s, r, 1, x)
                sr.unsqueeze(1).unsqueeze(-3) + # (B, 1, k, s, 1, r, x)
                rule_prob.unsqueeze(2).unsqueeze(2).unsqueeze(-1) # (B, S, 1, 1, r, r, 1)
            ).flatten(-4, -2).max(-2)[0] #.amax((-2, -3, -4)) # (B, S, k, x)
            return beta
            
        # area >= 2 x 2
        for h in range(2, H + 1):
            for w in range(2, W + 1):
                if verbose:
                    print(f"h {h} w {w}")

                y = torch.arange(H + 1 - h).to(device)
                x = torch.arange(W + 1 - w).to(device)
                
                # case 1: left-right composition
                sub_beta_area = beta_area[:, y, y + h].clone().permute(0, 2, 3, 4, 1)
                sl = stripe(sub_beta_area, W + 1 - w, w - 1, (0, 1), 1) # (B, kx, s, r, y)
                sr = stripe(sub_beta_area, W + 1 - w, w - 1, (1, w), 0) # (B, kx, s, r, y)
                beta1 = xyz(sl, sr, lr_Y_Z).permute(0, 3, 2, 1) # (B, S, kx, y) -> (B, y, kx, S)

                # case 2: above-below composition
                sub_beta_area = beta_area[..., x, x + w, :].clone().permute(0, 1, 2, 4, 3)
                sa = stripe(sub_beta_area, H + 1 - h, h - 1, (0, 1), 1) # (B, ky, s, r, x)
                sb = stripe(sub_beta_area, H + 1 - h, h - 1, (1, h), 0) # (B, ky, s, r, x)
                beta2 = xyz(sa, sb, ab_Y_Z).permute(0, 2, 3, 1) # (B, S, ky, x) -> (B, ky, x, S)
                
                # summarize betas
                beta = torch.stack((beta1, beta2), dim=-1).max(-1)[0] # (B, S, y, x)

                y = y.repeat_interleave(W + 1 - w)
                x = x.repeat(H + 1 - h)
                beta_area[:, y, y + h, x, x + w] = beta[:, y, x] + type_area[:, y, y + h, x, x + w]
                        
        final = beta_area[:, 0, H, 0, W] + root_prob
        # final = beta_area[:, 0, H, 0, W]
        ll = final.max(-1)[0]
    
        def _extract_parse(ll, areas, exclude_trivial=False, **kwargs):
            marginals = torch.autograd.grad(
                ll.sum(), areas, create_graph=True, only_inputs=True, allow_unused=False
            )[0]
            if marginals.dim() == 6: # only unlabled eval is supported
                marginals = marginals.sum(-1)

            marginals = marginals.permute(0, 1, 3, 2, 4) # (B, Y1, X1, Y2, X2)
            best_area = marginals.nonzero()
            best_beta = marginals[
                best_area[:, 0], best_area[:, 1], best_area[:, 2], best_area[:, 3], best_area[:, 4]
            ]
            if exclude_trivial:
                h1 = (best_area[:, 3] - best_area[:, 1]) == 1
                w1 = (best_area[:, 4] - best_area[:, 2]) == 1
                h1w1 = h1 & w1
                best_area = best_area[~h1w1]
                best_beta = best_beta[~h1w1]

    #         print(torch.cat((best_area, best_beta.unsqueeze(-1)), dim=-1))

            best_area = best_area[:, 1:] # trim the batch dim
            best_area = best_area.view(B, -1, best_area.shape[-1])
            parses = [best_area[i].cpu().tolist() for i in range(B)]        
            return parses 
        
        argmax = (
            None if not infer else _extract_parse(ll, type_area, **kwargs)
        )
        
        if not require_marginal:
            return ll, argmax, None, {"argmax": argmax, "marginal": None}
        
        marginal = self._compute_marginal(ll, type_area, **kwargs)
        return ll, argmax, marginal, {"argmax": argmax, "marginal": marginal}
    
    @torch.no_grad()
    def _viterbi_2d_manual(self, infer=True, require_marginal=False, verbose=False, shape=None, **kwargs):
        rule_prob, root_prob, term_prob = self.pcfgs
        
        device = rule_prob.device
        NT = rule_prob.shape[1]
        B, L, T = term_prob.shape[:3]
        H, W = (int(np.sqrt(L)),) * 2 if shape is None else shape[:2]
        
        term_prob = term_prob.view(B, H, W, -1)
        
        lr_Y_Z, *_ = lr_rule_prob = self._slice_rule_prob(rule_prob[:, :, 0], NT, T) # left-right rules
        ab_Y_Z, *_ = ab_rule_prob = self._slice_rule_prob(rule_prob[:, :, 1], NT, T) # above-below rules
        
        beta_area = torch.zeros( # the real location of inside scores
            (B, H, H + 1, W, W + 1, NT), device=device
        ).fill_(-MAXVAL)
        # TODO best_area and best_type can be merged by using bitwise shift
        # For example, 10 bits for split points, 10 for best left symbols, and 10 for best right symbols
        best_area = torch.zeros( # split point for every best symbol
            (B, H, H + 1, W, W + 1, NT), device=device
        ).long()
        
        best_type = torch.zeros( # left-right / bottom-up best symbol
            (B, H, H + 1, W, W + 1, NT), device=device
        ).long()
        
        for y in range(H): # area 1 x w
            if W < 2: continue
            sub_best_area = best_area[:, y, y + 1] # (B, W, W + 1, NT)
            sub_best_type = best_type[:, y, y + 1]
            sub_beta_area = beta_area[:, y, y + 1].clone() # (B, W, W + 1, NT)
            new_beta_area, new_best_area, new_best_type = self._viterbi_1d_manual(
                sub_best_area, sub_best_type, sub_beta_area, lr_rule_prob, term_prob[:, y], verbose=(y == 0),
            )
            beta_area[:, y, y + 1] = new_beta_area #+ sub_type_area
            best_type[:, y, y + 1] = new_best_type
            best_area[:, y, y + 1] = new_best_area
        
        for x in range(W): # area h x 1
            if H < 2: continue
            sub_best_area = best_area[..., x, x + 1, :]
            sub_best_type = best_type[..., x, x + 1, :]
            sub_beta_area = beta_area[..., x, x + 1, :].clone()
            new_beta_area, new_best_area, new_best_type = self._viterbi_1d_manual(
                sub_best_area, sub_best_type, sub_beta_area, ab_rule_prob, term_prob[..., x, :], verbose=False,
            )
            beta_area[..., x, x + 1, :] = new_beta_area #+ sub_type_area
            best_type[..., x, x + 1, :] = new_best_type
            best_area[..., x, x + 1, :] = new_best_area * -1
            
        def xyz(sl, sr, rule_prob):
            beta = ( # bkr, bkr, bslr
                sl.unsqueeze(1).unsqueeze(-2) + # (B, 1, k, s, r, 1, x)
                sr.unsqueeze(1).unsqueeze(-3) + # (B, 1, k, s, 1, r, x)
                rule_prob.unsqueeze(2).unsqueeze(2).unsqueeze(-1) # (B, S, 1, 1, r, r, 1)
            ) # (B, s, y, k, r, r, x)
            b, s, y, k, r, _, x = beta.shape
            beta, best_lr = beta.view((b, s, y, -1, x)).max(-2)
            best_s = best_lr.div((r * r), rounding_mode='floor') # // (r * r) # splits of all y * x areas along y
            best_lr = best_lr % (r * r) # areas >= 2 x 2 so must be r * r
            best_l = best_lr.div(r, rounding_mode='floor') # // r
            best_r = best_lr  % r
            best_lr = (best_l << 16) + best_r
            return beta, best_s, best_lr      
            
        # area >= 2 x 2
        for h in range(2, H + 1):
            for w in range(2, W + 1):
                if verbose:
                    print(f"h {h} w {w}")

                y = torch.arange(H + 1 - h).to(device)
                x = torch.arange(W + 1 - w).to(device)
                
                # case 1: left-right composition
                sub_beta_area = beta_area[:, y, y + h].clone().permute(0, 2, 3, 4, 1)
                sl = stripe(sub_beta_area, W + 1 - w, w - 1, (0, 1), 1) # (B, kx, s, r, y)
                sr = stripe(sub_beta_area, W + 1 - w, w - 1, (1, w), 0) # (B, kx, s, r, y)
                beta1, best1, type1 = xyz(sl, sr, lr_Y_Z) # among w - 1
                beta1 = beta1.permute(0, 3, 2, 1) # (B, S, kx, y) -> (B, y, kx, S)
                type1 = type1.permute(0, 3, 2, 1)
                best1 = best1.permute(0, 3, 2, 1)

                # case 2: above-below composition
                sub_beta_area = beta_area[..., x, x + w, :].clone().permute(0, 1, 2, 4, 3)
                sa = stripe(sub_beta_area, H + 1 - h, h - 1, (0, 1), 1) # (B, ky, s, r, x)
                sb = stripe(sub_beta_area, H + 1 - h, h - 1, (1, h), 0) # (B, ky, s, r, x)
                beta2, best2, type2 = xyz(sa, sb, ab_Y_Z) # among h - 1
                beta2 = beta2.permute(0, 2, 3, 1) # (B, S, ky, x) -> (B, ky, x, S)
                type2 = type2.permute(0, 2, 3, 1)
                best2 = best2.permute(0, 2, 3, 1)
                
                # summarize betas
                beta, k0 = torch.stack((beta1, beta2), dim=-1).max(-1) # (B, S, y, x)
                kind = torch.stack( # k0 \in {0, 1} -> real best indice
                    (type1, type2), dim=-1
                ).gather(-1, k0.unsqueeze(-1)).squeeze(-1)
                best = torch.stack( # k0 \in {0, 1} -> real best indice
                    (best1, best2), dim=-1
                ).gather(-1, k0.unsqueeze(-1)).squeeze(-1)
                k1 = (k0 == 0).long() + (k0 != 0) * -1
                best = (best + 1) * k1
                
                y = y.repeat_interleave(W + 1 - w)
                x = x.repeat(H + 1 - h)
                beta_area[:, y, y + h, x, x + w] = beta[:, y, x]
                best_type[:, y, y + h, x, x + w] = kind[:, y, x]
                best_area[:, y, y + h, x, x + w] = best[:, y, x]
                        
        final = beta_area[:, 0, H, 0, W] + root_prob
        # final = beta_area[:, 0, H, 0, W]
        ll, s = final.max(-1) # s: best type
        
        def _extract_parse(ll, areas, types, exclude_trivial=False, **kwargs):

            def backtrack(best, kind, y, x, h, w, s):
                if h == 1 and w == 1: # s may be a pre-terminal and will be skipped by this condition, or exceptions will be raised (s >= nt).
                    return [] if exclude_trivial else [(y, x, y + h, x + w)]
                k = best[y, y + h, x, x + w, s].cpu().item()
                z = kind[y, y + h, x, x + w, s].cpu().item()
                l = z >> 16
                r = z & ((1 << 16) - 1)
                if k > 0: # left-right composition
                    area1 = backtrack(best, kind, y, x, h, k, l)
                    area2 = backtrack(best, kind, y, x + k, h, w - k, r)
                else:
                    k = -k # above-below composition
                    area1 = backtrack(best, kind, y, x, k, w, l)
                    area2 = backtrack(best, kind, y + k, x, h - k, w, r)
                return area1 + area2 + [(y, x, y + h, x + w)]

            parses = [
                backtrack(
                    areas[i], types[i], 0, 0, H, W, s[i].cpu().item()
                ) for i in range(B)
            ]
            return parses
        
        argmax = (
            None if not infer else _extract_parse(ll, best_area, best_type, **kwargs)
        )
        
        if not require_marginal:
            return ll, argmax, None, {"argmax": argmax, "marginal": None}
        
        marginal = self._compute_marginal(ll, type_area, **kwargs)
        return ll, argmax, marginal, {"argmax": argmax, "marginal": marginal}

    def _ll_1d(self, beta_area, root_prob, H, W):
        row_ll = 0
        for y in range(H): # area 1 x w
            sub_beta_area = beta_area[:, y, y + 1, 0, W] # (B, NT)
            sub_ll = (sub_beta_area + root_prob).logsumexp(-1) 
            row_ll += sub_ll

        col_ll = 0 
        for x in range(W): # area h x 1
            sub_beta_area = beta_area[:, 0, H, x, x + 1] # (B, NT)
            sub_ll = (sub_beta_area + root_prob).logsumexp(-1)
            col_ll += sub_ll
        
        return (row_ll + col_ll) / (H + W)

    def _dp_parallel(
        self, infer=False, require_marginal=False, verbose=False, shape=None,
        drop_1d_fn=None, drop_2d_fn=None, require_1d_ll=False, **kwargs,
    ):
        if infer and not kwargs.get("mbr", False):
            return self._viterbi_2d_auto(
                require_marginal=require_marginal, verbose=verbose, shape=shape, **kwargs
            ) if kwargs.get("auto_infer", True) else self._viterbi_2d_manual(
                require_marginal=require_marginal, verbose=verbose, shape=shape, **kwargs
            ) # argmax version of _dp_parallel(...)
        
        rule_prob, root_prob, term_prob = self.pcfgs
        
        device = rule_prob.device
        NT = rule_prob.shape[1]
        B, L, T = term_prob.shape[:3]
        H, W = (int(np.sqrt(L)),) * 2 if shape is None else shape[:2]
        
        term_prob = term_prob.view(B, H, W, -1)
        
        lr_Y_Z, *_ = lr_rule_prob = self._slice_rule_prob(rule_prob[:, :, 0], NT, T) # left-right rules
        ab_Y_Z, *_ = ab_rule_prob = self._slice_rule_prob(rule_prob[:, :, 1], NT, T) # above-below rules
        
        beta_area = torch.zeros( # the real location of inside scores
            (B, H, H + 1, W, W + 1, NT), device=device
        ).fill_(-MAXVAL)
        
        type_area = torch.zeros( # to record grad-able operations
            (B, H, H + 1, W, W + 1, NT), device=device
        ).requires_grad_(infer or require_marginal)

        for y in range(H): # area 1 x w
            sub_type_area = type_area[:, y, y + 1] # (B, W, W + 1, NT)
            sub_beta_area = beta_area[:, y, y + 1].clone() # (B, W, W + 1, NT)
            new_beta_area = self._inside_1d_parallel(
                sub_type_area, sub_beta_area, lr_rule_prob, term_prob[:, y], drop_1d_fn=drop_1d_fn, verbose=False,
            )
            beta_area[:, y, y + 1] = new_beta_area #+ sub_type_area
        
        for x in range(W): # area h x 1
            sub_type_area = type_area[..., x, x + 1, :]
            sub_beta_area = beta_area[..., x, x + 1, :].clone()
            new_beta_area = self._inside_1d_parallel(
                sub_type_area, sub_beta_area, ab_rule_prob, term_prob[..., x, :], drop_1d_fn=drop_1d_fn,
            )
            beta_area[..., x, x + 1, :] = new_beta_area #+ sub_type_area

        def xyz(sl, sr, rule_prob):
            beta = ( # bkr, bkr, bslr
                sl.unsqueeze(1).unsqueeze(-2) + # (B, 1, k, s, r, 1, x)
                sr.unsqueeze(1).unsqueeze(-3) + # (B, 1, k, s, 1, r, x)
                rule_prob.unsqueeze(2).unsqueeze(2).unsqueeze(-1) # (B, S, 1, 1, r, r, 1)
            ).logsumexp((-2, -3, -4)) # (B, S, k, x)
            return beta
            
        # area >= 2 x 2
        for h in range(2, H + 1):
            for w in range(2, W + 1):
                if verbose:
                    print(f"h {h} w {w}")
                        
                ##################################
                # enumerate all possible ways of #
                #  composing a given rectangle   #
                ##################################

                y = torch.arange(H + 1 - h).to(device)
                x = torch.arange(W + 1 - w).to(device)
                
                # case 1: left-right composition
                # --------
                # |  |   |
                # |  |   |
                # |  |   |
                # |  |   |
                # --------
                sub_beta_area = beta_area[:, y, y + h].clone().permute(0, 2, 3, 4, 1)
                sl = stripe(sub_beta_area, W + 1 - w, w - 1, (0, 1), 1) # (B, kx, s, r, y)
                sr = stripe(sub_beta_area, W + 1 - w, w - 1, (1, w), 0) # (B, kx, s, r, y)
                beta1 = xyz(sl, sr, lr_Y_Z).permute(0, 3, 2, 1) # (B, S, kx, y) -> (B, y, kx, S)

                # case 2: above-below composition
                # --------
                # |      |
                # |      |
                # |~~~~~~|
                # |      |
                # --------
                sub_beta_area = beta_area[..., x, x + w, :].clone().permute(0, 1, 2, 4, 3)
                sa = stripe(sub_beta_area, H + 1 - h, h - 1, (0, 1), 1) # (B, ky, s, r, x)
                sb = stripe(sub_beta_area, H + 1 - h, h - 1, (1, h), 0) # (B, ky, s, r, x)
                beta2 = xyz(sa, sb, ab_Y_Z).permute(0, 2, 3, 1) # (B, S, ky, x) -> (B, ky, x, S)
                
                # summarize betas
                beta = torch.stack((beta1, beta2), dim=-1).logsumexp(-1) # (B, S, y, x)
                
                # drop some betas / areas
                beta = beta if drop_2d_fn is None else (
                    self._drop_beta(h, w, beta, drop_2d_fn, device)
                )

                y = y.repeat_interleave(W + 1 - w)
                x = x.repeat(H + 1 - h)
                beta_area[:, y, y + h, x, x + w] = beta[:, y, x] + type_area[:, y, y + h, x, x + w]
                        
        final = beta_area[:, 0, H, 0, W] + root_prob
        # final = beta_area[:, 0, H, 0, W]
        ll = final.logsumexp(-1)

        # compute ll along columns or rows
        ll_1d = self._ll_1d(beta_area, root_prob, H, W) if require_1d_ll else torch.tensor(0).to(ll)
        
        argmax = (
            None if not infer else self._extract_parse(ll, type_area, **kwargs)
        )
        
        if not require_marginal:
            return ll, argmax, None, {"argmax": argmax, "marginal": None, "ll1d": ll_1d}
        
        marginal = self._compute_marginal(ll, type_area, **kwargs)
        return ll, argmax, marginal, {"argmax": argmax, "marginal": marginal, "ll1d": ll_1d}
        
    def _dp_serial(
        self, infer=False, require_marginal=False, verbose=False, shape=None,
        drop_1d_fn=None, drop_2d_fn=None, require_1d_ll=False, **kwargs
    ):
        rule_prob, root_prob, term_prob = self.pcfgs
        
        device = rule_prob.device
        NT = rule_prob.shape[1]
        B, L, T = term_prob.shape[:3]
        H, W = (int(np.sqrt(L)),) * 2 if shape is None else shape[:2]
        
        term_prob = term_prob.view(B, H, W, -1)
        
        lr_Y_Z, *_ = lr_rule_prob = self._slice_rule_prob(rule_prob[:, :, 0], NT, T) # left-right rules
        ab_Y_Z, *_ = ab_rule_prob = self._slice_rule_prob(rule_prob[:, :, 1], NT, T) # above-below rules
        
        beta_area = torch.zeros( # the real location of inside scores
            (B, H, H + 1, W, W + 1, NT), device=device
        ).fill_(-MAXVAL)
        
        type_area = torch.zeros( # to record grad-able operations
            (B, H, H + 1, W, W + 1, NT), device=device
        ).requires_grad_(infer or require_marginal)
        
        for y in range(H): # area 1 x w
            sub_type_area = type_area[:, y, y + 1] # (B, W, W + 1, NT)
            sub_beta_area = beta_area[:, y, y + 1].clone() # (B, W, W + 1, NT)
            new_beta_area = self._inside_1d_serial(
                sub_type_area, sub_beta_area, lr_rule_prob, term_prob[:, y], verbose=False
            )
            beta_area[:, y, y + 1] = new_beta_area #+ sub_type_area
        
        for x in range(W): # area h x 1
            sub_type_area = type_area[..., x, x + 1, :]
            sub_beta_area = beta_area[..., x, x + 1, :].clone()
            new_beta_area = self._inside_1d_serial(
                sub_type_area, sub_beta_area, ab_rule_prob, term_prob[..., x, :]
            )
            beta_area[..., x, x + 1, :] = new_beta_area #+ sub_type_area

        # area >= 2 x 2
        for h in range(2, H + 1):
            for w in range(2, W + 1):
                if verbose:
                    print(f"h {h} w {w}")
                for y in range(H + 1 - h):
                    for x in range(W + 1 - w):
                        
                        ##################################
                        # enumerate all possible ways of #
                        #  composing a given rectangle   #
                        ##################################
                        
                        # case 1: left-right composition
                        # --------
                        # |  |   |
                        # |  |   |
                        # |  |   |
                        # |  |   |
                        # --------
                        beta_all = [] #0.
                        for sx in range(1, w):
                            sl = beta_area[:, y, y + h, x, x + sx]
                            sr = beta_area[:, y, y + h, x + sx, x + w]
                            
                            beta = ( # bl, br, bslr
                                sl.unsqueeze(1).unsqueeze(-1) +
                                sr.unsqueeze(1).unsqueeze(1) +
                                lr_Y_Z
                            ).logsumexp((2, 3))
                            beta_all.append(beta)

                        # case 2: above-below composition
                        # --------
                        # |      |
                        # |      |
                        # |~~~~~~|
                        # |      |
                        # --------
                        for sy in range(1, h):
                            sa = beta_area[:, y, y + sy, x, x + w]
                            sb = beta_area[:, y + sy, y + h, x, x + w]
                            
                            beta = ( # bl, br, bslr
                                sa.unsqueeze(1).unsqueeze(-1) +
                                sb.unsqueeze(1).unsqueeze(1) +
                                ab_Y_Z
                            ).logsumexp((2, 3))
                            beta_all.append(beta)
                        
                        beta = torch.stack(beta_all, dim=1).logsumexp(1)
                        beta_area[:, y, y + h, x, x + w] = beta + type_area[:, y, y + h, x, x + w]
                        
        final = beta_area[:, 0, H, 0, W] + root_prob
        # final = beta_area[:, 0, H, 0, W]
        ll = final.logsumexp(-1)

        # compute ll along columns or rows
        ll_1d = self._ll_1d(beta_area, root_prob, H, W) if require_1d_ll else torch.tensor(0).to(ll)
        
        argmax = (
            None if not infer else self._extract_parse(ll, type_area, **kwargs)
        )
        
        if not require_marginal:
            return ll, argmax, None, {"argmax": argmax, "marginal": None, "ll1d": ll_1d}
        
        marginal = self._compute_marginal(ll, type_area, **kwargs)
        return ll, argmax, marginal, {"argmax": argmax, "marginal": marginal, "ll1d": ll_1d}

class InsideAlg2DLA(InsideAlg2D): # standalone class, for two-dimensional inputs
    
    @staticmethod
    def _slice_rule_prob(rule_prob, NT, T):
        sli_np_h = slice(0, NT)
        sli_np_v = slice(NT, NT + NT)
        sli_tp_h = slice(NT + NT, NT + NT + T)
        sli_tp_v = slice(NT + NT + T, NT + NT + T + T)

        X_H_H = rule_prob[..., sli_np_h, sli_np_h]
        X_V_V = rule_prob[..., sli_np_v, sli_np_v]
        X_H_V = rule_prob[..., sli_np_h, sli_np_v]
        X_V_H = rule_prob[..., sli_np_v, sli_np_h]

        X_H_h = rule_prob[..., sli_np_h, sli_tp_h]
        X_h_H = rule_prob[..., sli_tp_h, sli_np_h]
        X_V_v = rule_prob[..., sli_np_v, sli_tp_v]
        X_v_V = rule_prob[..., sli_tp_v, sli_np_v]

        X_h_h = rule_prob[..., sli_tp_h, sli_tp_h]
        X_v_v = rule_prob[..., sli_tp_v, sli_tp_v]

        return X_H_H, X_V_V, X_H_V, X_V_H, (X_H_H, X_H_h, X_h_H, X_h_h), (X_V_V, X_V_v, X_v_V, X_v_v)

    def _compute_marginal(self, ll, areas, marginal_as_dict=False, **kwargs):
        marginals = torch.autograd.grad(
            ll.sum(), areas, create_graph=True, only_inputs=True, allow_unused=False
        )[0]
        if not marginal_as_dict:
            return marginals
        # linearization
        scores = dict()
        device = marginals.device
        B, _, H, _, W, *_ = marginals.shape
        for h in range(2, H + 1):
            for w in range(2, W + 1):
                y = torch.arange(H + 1 - h).repeat_interleave(W + 1 - w).to(device)
                x = torch.arange(W + 1 - w).repeat(H + 1 - h).to(device)
                scores[(h, w)] = marginals[:, :, y, y + h, x, x + w]
        return scores

    def _extract_parse(self, ll, areas, mbr=True, auto_infer=False, **kwargs):
        marginals = torch.autograd.grad(
            ll.sum(), areas, create_graph=True, only_inputs=True, allow_unused=False
        )[0]
        if marginals.dim() == 7: # only unlabled eval (non-subtype nonterminal) is supported
            marginals = marginals.sum((-1, 1))
        if mbr:
            mbr_fn = self._mbr_2d_auto if auto_infer else self._mbr_2d_manual
            return mbr_fn(marginals.detach(), **kwargs)
        # will need max operator in _dp() but have not been implemented
        raise NotImplementedError
        
    @torch.enable_grad()
    def _viterbi_2d_auto(self, infer=True, require_marginal=False, verbose=False, shape=None, **kwargs):
        rule_prob, root_prob, term_prob = self.pcfgs
        
        device = rule_prob.device
        NT = rule_prob.shape[1]
        B, L, T = term_prob.shape[:3]
        H, W = (int(np.sqrt(L)),) * 2 if shape is None else shape[:2]
        
        T = T // 2 # w/ latent annotation
        
        root_prob = root_prob.view(B, 2, -1)
        term_prob = term_prob.view(B, H, W, 2, -1)
        
        lr_H_H, lr_V_V, lr_H_V, lr_V_H, lr_rule_prob, _ = self._slice_rule_prob(rule_prob[:, :, 0], NT, T) # left-right rules
        ab_H_H, ab_V_V, ab_H_V, ab_V_H, _, ab_rule_prob = self._slice_rule_prob(rule_prob[:, :, 1], NT, T) # above-below rules
        
        beta_area = torch.zeros( # the real location of inside scores
            (B, 2, H, H + 1, W, W + 1, NT), device=device
        ).fill_(-MAXVAL)
        
        type_area = torch.zeros( # to record grad-able operations
            (B, 2, H, H + 1, W, W + 1, NT), device=device
        ).requires_grad_(infer or require_marginal)
          
        for y in range(H): # area 1 x w
            if W < 2: continue
            sub_type_area = type_area[:, 0, y, y + 1] # (B, W, W + 1, NT)
            sub_beta_area = beta_area[:, 0, y, y + 1].clone() # (B, W, W + 1, NT)
            new_beta_area = self._viterbi_1d_auto(
                sub_type_area, sub_beta_area, lr_rule_prob, term_prob[:, y, :, 0], verbose=False
            )
            beta_area[:, 0, y, y + 1] = new_beta_area #+ sub_type_area
        
        for x in range(W): # area h x 1
            if H < 2: continue
            sub_type_area = type_area[:, 1, ..., x, x + 1, :]
            sub_beta_area = beta_area[:, 1, ..., x, x + 1, :].clone()
            new_beta_area = self._viterbi_1d_auto(
                sub_type_area, sub_beta_area, ab_rule_prob, term_prob[..., x, 1, :]
            )
            beta_area[:, 1, ..., x, x + 1, :] = new_beta_area #+ sub_type_area
        
        def xyz(sl, sr, rule_prob):
            beta = ( # bkr, bkr, bslr
                sl.unsqueeze(1).unsqueeze(-2) + # (B, 1, k, s, r, 1, x)
                sr.unsqueeze(1).unsqueeze(-3) + # (B, 1, k, s, 1, r, x)
                rule_prob.unsqueeze(2).unsqueeze(2).unsqueeze(-1) # (B, S, 1, 1, r, r, 1)
            ).flatten(-4, -2).max(-2)[0]   #.amax((-2, -3, -4)) # (B, S, k, x)
            return beta
            
        # area >= 2 x 2
        ff = 0
        for h in range(2, H + 1):
            for w in range(2, W + 1):
                if verbose:
                    print(f"h {h} w {w}")

                y = torch.arange(H + 1 - h).to(device)
                x = torch.arange(W + 1 - w).to(device)
                
                # case 1: left-right composition
                sub_beta_area = beta_area[:, :, y, y + h].clone().view(
                    B * 2, H + 1 - h, W, W + 1, NT
                ).permute(0, 2, 3, 4, 1)
                sl = stripe(sub_beta_area, W + 1 - w, w - 1, (0, 1), 1).view(
                    B, 2, W + 1 - w, w - 1, NT, H + 1 - h
                ) # (B, kx, s, r, y)
                sr = stripe(sub_beta_area, W + 1 - w, w - 1, (1, w), 0).view(
                    B, 2, W + 1 - w, w - 1, NT, H + 1 - h
                ) # (B, kx, s, r, y)
                betas = [
                    xyz(sl[:, 0], sr[:, 0], lr_H_H),
                    xyz(sl[:, 0], sr[:, 1], lr_H_V),
                    xyz(sl[:, 1], sr[:, 0], lr_V_H),
                    xyz(sl[:, 1], sr[:, 1], lr_V_V),
                ] # (B, S, kx, y) -> (B, y, kx, S)
                beta1 = torch.stack([beta.permute(0, 3, 2, 1) for beta in betas], -1).max(-1)[0]

                # case 2: above-below composition
                sub_beta_area = beta_area[..., x, x + w, :].clone().view(
                    B * 2, H, H + 1, W + 1 - w, NT
                ).permute(0, 1, 2, 4, 3)
                sa = stripe(sub_beta_area, H + 1 - h, h - 1, (0, 1), 1).view(
                    B, 2, H + 1 - h, h - 1, NT, W + 1 - w
                ) # (B, ky, s, r, x)
                sb = stripe(sub_beta_area, H + 1 - h, h - 1, (1, h), 0).view(
                    B, 2, H + 1 - h, h - 1, NT, W + 1 - w
                ) # (B, ky, s, r, x)
                betas = [
                    xyz(sa[:, 0], sb[:, 0], ab_H_H),
                    xyz(sa[:, 0], sb[:, 1], ab_H_V),
                    xyz(sa[:, 1], sb[:, 0], ab_V_H),
                    xyz(sa[:, 1], sb[:, 1], ab_V_V),
                ] # (B, S, ky, x) -> (B, ky, x, S)
                beta2 = torch.stack([beta.permute(0, 2, 3, 1) for beta in betas], -1).max(-1)[0]
                
                # summarize betas
                y = y.repeat_interleave(W + 1 - w)
                x = x.repeat(H + 1 - h)
                beta_area[:, 0, y, y + h, x, x + w] = beta1[:, y, x] + type_area[:, 0, y, y + h, x, x + w]
                beta_area[:, 1, y, y + h, x, x + w] = beta2[:, y, x] + type_area[:, 1, y, y + h, x, x + w] 
                        
        final = beta_area[:, :, 0, H, 0, W] + root_prob
        # final = beta_area[:, 0, H, 0, W]
        ll = final.flatten(-2).max(-1)[0] #.amax((-1, -2))
    
        def _extract_parse(ll, areas, exclude_trivial=False, **kwargs):
            marginals = torch.autograd.grad(
                ll.sum(), areas, create_graph=True, only_inputs=True, allow_unused=False
            )[0]
            if marginals.dim() == 7: # only unlabled eval is supported (should contains 1. & 0.)
                marginals = marginals.sum(-1)

            marginals = marginals.permute(0, 1, 2, 4, 3, 5) #(0, 1, 4, 2, 5, 3) # (B, 2, Y1, X1, Y2, X2)
            best_area = marginals.nonzero()
            best_beta = marginals[
                best_area[:, 0], best_area[:, 1], best_area[:, 2], best_area[:, 3], best_area[:, 4], best_area[:, 5]
            ]
            if exclude_trivial:
                h1 = (best_area[:, 4] - best_area[:, 2]) == 1
                w1 = (best_area[:, 5] - best_area[:, 3]) == 1
                h1w1 = h1 & w1
                best_area = best_area[~h1w1]
                best_beta = best_beta[~h1w1]

#             print(torch.cat((best_area, best_beta.unsqueeze(-1)), dim=-1))

            best_area = best_area[:, 2:] # trim the batch dim and the direction dim
            best_area = best_area.view(B, -1, best_area.shape[-1])
            parses = [best_area[i].cpu().tolist() for i in range(B)]        
            return parses 
        
        argmax = (
            None if not infer else _extract_parse(ll, type_area, **kwargs)
        )
        
        if not require_marginal:
            return ll, argmax, None, {"argmax": argmax, "marginal": None}
        
        marginal = self._compute_marginal(ll, type_area, **kwargs)
        return ll, argmax, marginal, {"argmax": argmax, "marginal": marginal}   
        
    @torch.no_grad()
    def _viterbi_2d_manual(self, infer=True, require_marginal=False, verbose=False, shape=None, **kwargs):
        rule_prob, root_prob, term_prob = self.pcfgs
        
        device = rule_prob.device
        NT = rule_prob.shape[1]
        B, L, T = term_prob.shape[:3]
        H, W = (int(np.sqrt(L)),) * 2 if shape is None else shape[:2]
        
        T = T // 2 # w/ latent annotation
        
        root_prob = root_prob.view(B, 2, -1)
        term_prob = term_prob.view(B, H, W, 2, -1)
        
        lr_H_H, lr_V_V, lr_H_V, lr_V_H, lr_rule_prob, _ = self._slice_rule_prob(rule_prob[:, :, 0], NT, T) # left-right rules
        ab_H_H, ab_V_V, ab_H_V, ab_V_H, _, ab_rule_prob = self._slice_rule_prob(rule_prob[:, :, 1], NT, T) # above-below rules
        
        beta_area = torch.zeros( # the real location of inside scores
            (B, 2, H, H + 1, W, W + 1, NT), device=device
        ).fill_(-MAXVAL)
        # TODO best_area and best_type can be merged by using bitwise shift
        # For example, 10 bits for split points, 10 for best left symbols, and 10 for best right symbols
        best_area = torch.zeros( # split point for every best symbol
            (B, 2, H, H + 1, W, W + 1, NT), device=device
        ).long()
        
        best_type = torch.zeros( # left-right / bottom-up best symbol
            (B, 2, H, H + 1, W, W + 1, NT), device=device
        ).long()
        
        for y in range(H): # area 1 x w
            if W < 2: continue
            sub_best_area = best_area[:, 0, y, y + 1] # (B, W, W + 1, NT)
            sub_best_type = best_type[:, 0, y, y + 1]
            sub_beta_area = beta_area[:, 0, y, y + 1].clone() # (B, W, W + 1, NT)
            new_beta_area, new_best_area, new_best_type = self._viterbi_1d_manual(
                sub_best_area, sub_best_type, sub_beta_area, lr_rule_prob, term_prob[:, y, :, 0], verbose=(y == 0),
            )
            beta_area[:, 0, y, y + 1] = new_beta_area #+ sub_type_area
            best_type[:, 0, y, y + 1] = new_best_type
            best_area[:, 0, y, y + 1] = new_best_area
        
        for x in range(W): # area h x 1
            if H < 2: continue
            sub_best_area = best_area[:, 1, ..., x, x + 1, :]
            sub_best_type = best_type[:, 1, ..., x, x + 1, :]
            sub_beta_area = beta_area[:, 1, ..., x, x + 1, :].clone()
            new_beta_area, new_best_area, new_best_type = self._viterbi_1d_manual(
                sub_best_area, sub_best_type, sub_beta_area, ab_rule_prob, term_prob[..., x, 1, :], verbose=False,
            )
            beta_area[:, 1, ..., x, x + 1, :] = new_beta_area #+ sub_type_area
            best_type[:, 1, ..., x, x + 1, :] = new_best_type
            best_area[:, 1, ..., x, x + 1, :] = new_best_area + (1 << 18) + (1 << 16) #* -1
            
        def xyz(sl, sr, rule_prob):
            beta = ( # bkr, bkr, bslr
                sl.unsqueeze(1).unsqueeze(-2) + # (B, 1, k, s, r, 1, x)
                sr.unsqueeze(1).unsqueeze(-3) + # (B, 1, k, s, 1, r, x)
                rule_prob.unsqueeze(2).unsqueeze(2).unsqueeze(-1) # (B, S, 1, 1, r, r, 1)
            ) # (B, s, y, k, r, r, x)
            b, s, y, k, r, _, x = beta.shape
            beta, best_lr = beta.view((b, s, y, -1, x)).max(-2)
            best_s = best_lr.div((r * r), rounding_mode='floor') # // (r * r) # splits of all y * x areas along y
            best_lr = best_lr % (r * r) # areas >= 2 x 2 so must be r * r
            best_l = best_lr.div(r, rounding_mode='floor') # // r
            best_r = best_lr  % r
            best_lr = (best_l << 16) + best_r
            return beta, best_s, best_lr      
            
        # area >= 2 x 2
        for h in range(2, H + 1):
            for w in range(2, W + 1):
                if verbose:
                    print(f"h {h} w {w}")

                y = torch.arange(H + 1 - h).to(device)
                x = torch.arange(W + 1 - w).to(device)
                
                # case 1: left-right composition
                sub_beta_area = beta_area[:, :, y, y + h].clone().reshape(
                    B * 2, H + 1 - h, W, W + 1, NT
                ).permute(0, 2, 3, 4, 1)
                sl = stripe(sub_beta_area, W + 1 - w, w - 1, (0, 1), 1).view(
                    B, 2, W + 1 - w, w - 1, NT, H + 1 - h
                ) # (B, kx, s, r, y)
                sr = stripe(sub_beta_area, W + 1 - w, w - 1, (1, w), 0).view(
                    B, 2, W + 1 - w, w - 1, NT, H + 1 - h
                ) # (B, kx, s, r, y)
                betas = [
                    xyz(sl[:, 0], sr[:, 0], lr_H_H),
                    xyz(sl[:, 0], sr[:, 1], lr_H_V),
                    xyz(sl[:, 1], sr[:, 0], lr_V_H),
                    xyz(sl[:, 1], sr[:, 1], lr_V_V),
                ] # (B, S, kx, y) -> (B, y, kx, S)
                beta1 = torch.stack([beta[0].permute(0, 3, 2, 1) for beta in betas], -1)
                best1 = torch.stack([beta[1].permute(0, 3, 2, 1) for beta in betas], -1)
                type1 = torch.stack([beta[2].permute(0, 3, 2, 1) for beta in betas], -1)
                
                beta1, k0 = beta1.max(-1) # (B, S, y, x)
                type1 = type1.gather(-1, k0.unsqueeze(-1)).squeeze(-1)
                best1 = best1.gather(-1, k0.unsqueeze(-1)).squeeze(-1)
                
                dl = k0.div(2, rounding_mode='floor') # // r
                dr = k0 % 2
                best1 = (best1 + 1) + (dl << 18) + (dr << 16)
                

                # case 2: above-below composition
                sub_beta_area = beta_area[..., x, x + w, :].clone().reshape(
                    B * 2, H, H + 1, W + 1 - w, NT
                ).permute(0, 1, 2, 4, 3)
                sa = stripe(sub_beta_area, H + 1 - h, h - 1, (0, 1), 1).view(
                    B, 2, H + 1 - h, h - 1, NT, W + 1 - w
                ) # (B, ky, s, r, x)
                sb = stripe(sub_beta_area, H + 1 - h, h - 1, (1, h), 0).view(
                    B, 2, H + 1 - h, h - 1, NT, W + 1 - w
                ) # (B, ky, s, r, x)
                betas = [
                    xyz(sa[:, 0], sb[:, 0], ab_H_H),
                    xyz(sa[:, 0], sb[:, 1], ab_H_V),
                    xyz(sa[:, 1], sb[:, 0], ab_V_H),
                    xyz(sa[:, 1], sb[:, 1], ab_V_V),
                ] # (B, S, kx, y) -> (B, y, kx, S)
                beta2 = torch.stack([beta[0].permute(0, 2, 3, 1) for beta in betas], -1)
                best2 = torch.stack([beta[1].permute(0, 2, 3, 1) for beta in betas], -1)
                type2 = torch.stack([beta[2].permute(0, 2, 3, 1) for beta in betas], -1)
                
                beta2, k0 = beta2.max(-1) # (B, S, y, x)
                type2 = type2.gather(-1, k0.unsqueeze(-1)).squeeze(-1)
                best2 = best2.gather(-1, k0.unsqueeze(-1)).squeeze(-1)
                
                da = k0.div(2, rounding_mode='floor') # // r
                db = k0 % 2
                best2 = (best2 + 1) + (da << 18) + (db << 16)
                
                
                # summarize betas
                y = y.repeat_interleave(W + 1 - w)
                x = x.repeat(H + 1 - h)
                beta_area[:, 0, y, y + h, x, x + w] = beta1[:, y, x]
                best_type[:, 0, y, y + h, x, x + w] = type1[:, y, x]
                best_area[:, 0, y, y + h, x, x + w] = best1[:, y, x]
                
                beta_area[:, 1, y, y + h, x, x + w] = beta2[:, y, x]
                best_type[:, 1, y, y + h, x, x + w] = type2[:, y, x]
                best_area[:, 1, y, y + h, x, x + w] = best2[:, y, x]
                
        final = beta_area[:, :, 0, H, 0, W] + root_prob
        # final = beta_area[:, 0, H, 0, W]
        ll, k = final.view(B, -1).max(-1) # s: best type
        d = k.div(NT, rounding_mode='floor') # // r
        s = k % NT
        
        def _extract_parse(ll, areas, types, exclude_trivial=False, **kwargs):

            def backtrack(best, kind, y, x, h, w, s, d):
                if h == 1 and w == 1: # s may be a pre-terminal and will be skipped by this condition, or exceptions will be raised (s >= nt).
                    return [] if exclude_trivial else [(y, x, y + h, x + w)]                
                k = best[d, y, y + h, x, x + w, s].cpu().item()
                d1 = (k >> 18)
                d2 = (k >> 16) & ((1 << 2) - 1)
                k = k & ((1 << 16) - 1)
                
                z = kind[d, y, y + h, x, x + w, s].cpu().item()
                l = z >> 16
                r = z & ((1 << 16) - 1)
                
                if d == 0: # left-right composition
                    area1 = backtrack(best, kind, y, x, h, k, l, d1)
                    area2 = backtrack(best, kind, y, x + k, h, w - k, r, d2)
                elif d == 1: # above-below composition
                    area1 = backtrack(best, kind, y, x, k, w, l, d1)
                    area2 = backtrack(best, kind, y + k, x, h - k, w, r, d2)
                else:
                    raise KeyError(f"({y}, {x}, {h}, {w}, {s}, {d}) k {k} d1 {d1} d2 {d2} l {l} r {r}")
                return area1 + area2 + [(y, x, y + h, x + w)]

            parses = [
                backtrack(
                    areas[i], types[i], 0, 0, H, W, s[i].cpu().item(), d[i].cpu().item()
                ) for i in range(B)
            ]
            return parses
        
        argmax = (
            None if not infer else _extract_parse(ll, best_area, best_type, **kwargs)
        )
        
        if not require_marginal:
            return ll, argmax, None, {"argmax": argmax, "marginal": None}
        
        marginal = self._compute_marginal(ll, type_area, **kwargs)
        return ll, argmax, marginal, {"argmax": argmax, "marginal": marginal}

    def _ll_1d(self, beta_area, root_prob, H, W):
        row_ll = 0
        for y in range(H): # area 1 x w
            sub_beta_area = beta_area[:, :, y, y + 1, 0, W] # (B, 2, NT)
            sub_ll = (sub_beta_area + root_prob).logsumexp((-1, -2)) 
            row_ll += sub_ll
#             print(sub_ll)

        col_ll = 0 
        for x in range(W): # area h x 1
            sub_beta_area = beta_area[:, :, 0, H, x, x + 1] # (B, 2, NT)
            sub_ll = (sub_beta_area + root_prob).logsumexp((-1, -2))
            col_ll += sub_ll
#             print(sub_ll)
        
        return (row_ll + col_ll) / (H + W)

    def _dp_parallel(
        self, infer=False, require_marginal=False, verbose=False, shape=None,
        drop_1d_fn=None, drop_2d_fn=None, require_1d_ll=False, **kwargs,
    ):
        if infer and not kwargs.get("mbr", False):
            return self._viterbi_2d_auto(
                require_marginal=require_marginal, verbose=verbose, shape=shape, **kwargs
            ) if kwargs.get("auto_infer", True) else self._viterbi_2d_manual(
                require_marginal=require_marginal, verbose=verbose, shape=shape, **kwargs
            ) # argmax version of _dp_parallel(...)
        
        rule_prob, root_prob, term_prob = self.pcfgs
        
        device = rule_prob.device
        NT = rule_prob.shape[1]
        B, L, T = term_prob.shape[:3]
        H, W = (int(np.sqrt(L)),) * 2 if shape is None else shape[:2]
        
        T = T // 2 # w/ latent annotation
        
        root_prob = root_prob.view(B, 2, -1)
        term_prob = term_prob.view(B, H, W, 2, -1)
        
        lr_H_H, lr_V_V, lr_H_V, lr_V_H, lr_rule_prob, _ = self._slice_rule_prob(rule_prob[:, :, 0], NT, T) # left-right rules
        ab_H_H, ab_V_V, ab_H_V, ab_V_H, _, ab_rule_prob = self._slice_rule_prob(rule_prob[:, :, 1], NT, T) # above-below rules
        
        beta_area = torch.zeros( # the real location of inside scores
            (B, 2, H, H + 1, W, W + 1, NT), device=device
        ).fill_(-MAXVAL)
        
        type_area = torch.zeros( # to record grad-able operations
            (B, 2, H, H + 1, W, W + 1, NT), device=device
        ).requires_grad_(infer or require_marginal)
        
        for y in range(H): # area 1 x w
            sub_type_area = type_area[:, 0, y, y + 1] # (B, W, W + 1, NT)
            sub_beta_area = beta_area[:, 0, y, y + 1].clone() # (B, W, W + 1, NT)
            new_beta_area = self._inside_1d_serial(
                sub_type_area, sub_beta_area, lr_rule_prob, term_prob[:, y, :, 0], verbose=False
            )
            beta_area[:, 0, y, y + 1] = new_beta_area #+ sub_type_area
        
        for x in range(W): # area h x 1
            sub_type_area = type_area[:, 1, ..., x, x + 1, :]
            sub_beta_area = beta_area[:, 1, ..., x, x + 1, :].clone()
            new_beta_area = self._inside_1d_serial(
                sub_type_area, sub_beta_area, ab_rule_prob, term_prob[..., x, 1, :]
            )
            beta_area[:, 1, ..., x, x + 1, :] = new_beta_area #+ sub_type_area

        def xyz(sl, sr, rule_prob):
            beta = ( # bkr, bkr, bslr
                sl.unsqueeze(1).unsqueeze(-2) + # (B, 1, k, s, r, 1, x)
                sr.unsqueeze(1).unsqueeze(-3) + # (B, 1, k, s, 1, r, x)
                rule_prob.unsqueeze(2).unsqueeze(2).unsqueeze(-1) # (B, S, 1, 1, r, r, 1)
            ).logsumexp((-2, -3, -4)) # (B, S, k, x)
            return beta
            
        # area >= 2 x 2
        for h in range(2, H + 1):
            for w in range(2, W + 1):
                if verbose:
                    print(f"h {h} w {w}")
                        
                ##################################
                # enumerate all possible ways of #
                #  composing a given rectangle   #
                ##################################

                y = torch.arange(H + 1 - h).to(device)
                x = torch.arange(W + 1 - w).to(device)
                
                # case 1: left-right composition
                # --------
                # |  |   |
                # |  |   |
                # |  |   |
                # |  |   |
                # --------
                sub_beta_area = beta_area[:, :, y, y + h].clone().reshape(
                    B * 2, H + 1 - h, W, W + 1, NT
                ).permute(0, 2, 3, 4, 1)
                sl = stripe(sub_beta_area, W + 1 - w, w - 1, (0, 1), 1).view(
                    B, 2, W + 1 - w, w - 1, NT, H + 1 - h
                ) # (B, kx, s, r, y)
                sr = stripe(sub_beta_area, W + 1 - w, w - 1, (1, w), 0).view(
                    B, 2, W + 1 - w, w - 1, NT, H + 1 - h
                ) # (B, kx, s, r, y)
                beta1 = [
                    xyz(sl[:, 0], sr[:, 0], lr_H_H).permute(0, 3, 2, 1),
                    xyz(sl[:, 0], sr[:, 1], lr_H_V).permute(0, 3, 2, 1),
                    xyz(sl[:, 1], sr[:, 0], lr_V_H).permute(0, 3, 2, 1),
                    xyz(sl[:, 1], sr[:, 1], lr_V_V).permute(0, 3, 2, 1),
                ] # (B, S, kx, y) -> (B, y, kx, S)
                beta1 = torch.stack(beta1, dim=1).logsumexp(1)

                # case 2: above-below composition
                # --------
                # |      |
                # |      |
                # |~~~~~~|
                # |      |
                # --------
                sub_beta_area = beta_area[..., x, x + w, :].clone().reshape(
                    B * 2, H, H + 1, W + 1 - w, NT
                ).permute(0, 1, 2, 4, 3)
                sa = stripe(sub_beta_area, H + 1 - h, h - 1, (0, 1), 1).view(
                    B, 2, H + 1 - h, h - 1, NT, W + 1 - w
                ) # (B, ky, s, r, x)
                sb = stripe(sub_beta_area, H + 1 - h, h - 1, (1, h), 0).view(
                    B, 2, H + 1 - h, h - 1, NT, W + 1 - w
                ) # (B, ky, s, r, x)
                beta2 = [
                    xyz(sa[:, 0], sb[:, 0], ab_H_H).permute(0, 2, 3, 1),
                    xyz(sa[:, 0], sb[:, 1], ab_H_V).permute(0, 2, 3, 1),
                    xyz(sa[:, 1], sb[:, 0], ab_V_H).permute(0, 2, 3, 1),
                    xyz(sa[:, 1], sb[:, 1], ab_V_V).permute(0, 2, 3, 1),
                ] # (B, S, ky, x) -> (B, ky, x, S)
                beta2 = torch.stack(beta2, dim=1).logsumexp(1)
                
                # drop some betas / areas
                beta1 = beta1 if drop_2d_fn is None else (
                    self._drop_beta(h, w, beta1, drop_2d_fn, device)
                )
                beta2 = beta2 if drop_2d_fn is None else (
                    self._drop_beta(h, w, beta2, drop_2d_fn, device)
                )
                
                # summarize betas
                y = y.repeat_interleave(W + 1 - w)
                x = x.repeat(H + 1 - h)
                beta_area[:, 0, y, y + h, x, x + w] = beta1[:, y, x] + type_area[:, 0, y, y + h, x, x + w]
                beta_area[:, 1, y, y + h, x, x + w] = beta2[:, y, x] + type_area[:, 1, y, y + h, x, x + w]
                        
        final = beta_area[:, :, 0, H, 0, W] + root_prob
        # final = beta_area[:, 0, H, 0, W]
        ll = final.logsumexp((-1, -2))

        # compute ll along columns or rows
        ll_1d = self._ll_1d(beta_area, root_prob, H, W) if require_1d_ll else torch.tensor(0).to(ll)
        
        argmax = (
            None if not infer else self._extract_parse(ll, type_area, **kwargs)
        )
        
        if not require_marginal:
            return ll, argmax, None, {"argmax": argmax, "marginal": None, "ll1d": ll_1d}
        
        marginal = self._compute_marginal(ll, type_area, **kwargs)
        return ll, argmax, marginal, {"argmax": argmax, "marginal": marginal, "ll1d": ll_1d}
        
    def _dp_serial(
        self, infer=False, require_marginal=False, verbose=False, shape=None,
        drop_1d_fn=None, drop_2d_fn=None, require_1d_ll=False, **kwargs
    ):
        rule_prob, root_prob, term_prob = self.pcfgs
        
        device = rule_prob.device
        NT = rule_prob.shape[1]
        B, L, T = term_prob.shape[:3]
        H, W = (int(np.sqrt(L)),) * 2 if shape is None else shape[:2]
        
        T = T // 2 # w/ latent annotation
        
        root_prob = root_prob.view(B, 2, -1)
        term_prob = term_prob.view(B, H, W, 2, -1)
        
        lr_H_H, lr_V_V, lr_H_V, lr_V_H, lr_rule_prob, _ = self._slice_rule_prob(rule_prob[:, :, 0], NT, T) # left-right rules
        ab_H_H, ab_V_V, ab_H_V, ab_V_H, _, ab_rule_prob = self._slice_rule_prob(rule_prob[:, :, 1], NT, T) # above-below rules
        
        beta_area = torch.zeros( # the real location of inside scores
            (B, 2, H, H + 1, W, W + 1, NT), device=device
        ).fill_(-MAXVAL)
        
        type_area = torch.zeros( # to record grad-able operations
            (B, 2, H, H + 1, W, W + 1, NT), device=device
        ).requires_grad_(infer or require_marginal)
        
        for y in range(H): # area 1 x w
            sub_type_area = type_area[:, 0, y, y + 1] # (B, W, W + 1, NT)
            sub_beta_area = beta_area[:, 0, y, y + 1].clone() # (B, W, W + 1, NT)
            new_beta_area = self._inside_1d_serial(
                sub_type_area, sub_beta_area, lr_rule_prob, term_prob[:, y, :, 0], verbose=False
            )
            beta_area[:, 0, y, y + 1] = new_beta_area #+ sub_type_area
        
        for x in range(W): # area h x 1
            sub_type_area = type_area[:, 1, ..., x, x + 1, :]
            sub_beta_area = beta_area[:, 1, ..., x, x + 1, :].clone()
            new_beta_area = self._inside_1d_serial(
                sub_type_area, sub_beta_area, ab_rule_prob, term_prob[..., x, 1, :]
            )
            beta_area[:, 1, ..., x, x + 1, :] = new_beta_area #+ sub_type_area
            
        def xyz(sl, sr, rule_prob):
            beta = ( # bl, br, bslr
                sl.unsqueeze(1).unsqueeze(-1) +
                sr.unsqueeze(1).unsqueeze(1) +
                rule_prob
            ).logsumexp((2, 3))
            return beta
            
        # area >= 2 x 2
        for h in range(2, H + 1):
            for w in range(2, W + 1):
                if verbose:
                    print(f"h {h} w {w}")
                for y in range(H + 1 - h):
                    for x in range(W + 1 - w):
                        
                        ##################################
                        # enumerate all possible ways of #
                        #  composing a given rectangle   #
                        ##################################
                        
                        # case 1: left-right composition
                        # --------
                        # |  |   |
                        # |  |   |
                        # |  |   |
                        # |  |   |
                        # --------
                        beta_all = [] #0.
                        for sx in range(1, w):
                            sl = beta_area[:, :, y, y + h, x, x + sx] # (B, 2, ...)
                            sr = beta_area[:, :, y, y + h, x + sx, x + w] # (B, 2, ...)
                            
                            beta = [
                                xyz(sl[:, 0], sr[:, 0], lr_H_H),
                                xyz(sl[:, 0], sr[:, 1], lr_H_V),
                                xyz(sl[:, 1], sr[:, 0], lr_V_H),
                                xyz(sl[:, 1], sr[:, 1], lr_V_V),  
                            ]
                            beta = torch.stack(beta, dim=1).logsumexp(1)
                            beta_all.append(beta)
                        beta = torch.stack(beta_all, dim=1).logsumexp(1)
                        beta_area[:, 0, y, y + h, x, x + w] = beta + type_area[:, 0, y, y + h, x, x + w]

                        # case 2: above-below composition
                        # --------
                        # |      |
                        # |      |
                        # |~~~~~~|
                        # |      |
                        # --------
                        beta_all = [] #0.
                        for sy in range(1, h):
                            sa = beta_area[:, :, y, y + sy, x, x + w] # (B, 2, ...)
                            sb = beta_area[:, :, y + sy, y + h, x, x + w] # (B, 2, ...)
                            
                            beta = [
                                xyz(sa[:, 0], sb[:, 0], ab_H_H),
                                xyz(sa[:, 0], sb[:, 1], ab_H_V),
                                xyz(sa[:, 1], sb[:, 0], ab_V_H),
                                xyz(sa[:, 1], sb[:, 1], ab_V_V),   
                            ]
                            beta = torch.stack(beta, dim=1).logsumexp(1)
                            beta_all.append(beta)
                        beta = torch.stack(beta_all, dim=1).logsumexp(1)
                        beta_area[:, 1, y, y + h, x, x + w] = beta + type_area[:, 1, y, y + h, x, x + w]
                        
        final = beta_area[:, :, 0, H, 0, W] + root_prob
        # final = beta_area[:, 0, H, 0, W]
        ll = final.logsumexp((-1, -2))

        # compute ll along columns or rows
        ll_1d = self._ll_1d(beta_area, root_prob, H, W) if require_1d_ll else torch.tensor(0).to(ll)
        
        argmax = (
            None if not infer else self._extract_parse(ll, type_area, **kwargs)
        )
        
        if not require_marginal:
            return ll, argmax, None, {"argmax": argmax, "marginal": None, "ll1d": ll_1d}
        
        marginal = self._compute_marginal(ll, type_area, **kwargs)
        return ll, argmax, marginal, {"argmax": argmax, "marginal": marginal, "ll1d": ll_1d}
