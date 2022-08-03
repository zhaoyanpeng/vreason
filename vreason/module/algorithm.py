import torch
import numpy as np
from torch import nn, Tensor

from vreason.util import stripe

MAXVAL = 1e9

__all__ = ["InsideAlg1D", "InsideAlg2D"]


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
    def _inside_1d_serial(type_area, beta_area, rule_prob, term_prob, verbose=False):
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
    def _inside_1d_parallel(type_area, beta_area, rule_prob, term_prob, verbose=False):
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
        if marginals.dim() == 6: # unlabled
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
            sub_span_area = areas[:, y, y + 1]
            sub_type_area = type_area[:, y, y + 1] # (B, W, W + 1)
            sub_beta_area = beta_area[:, y, y + 1] # (B, W, W + 1)
            new_beta_area = self._mbr_1d_auto(
                sub_type_area, sub_beta_area, sub_span_area, verbose=verbose
            )
            beta_area[:, y, y + 1] = new_beta_area #+ sub_type_area
        
        for x in range(W): # area h x 1
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
                sub_beta_area = beta_area[:, y, y + h].permute(0, 2, 3, 1)
                sl = stripe(sub_beta_area, W + 1 - w, w - 1, (0, 1), 1) # (B, kx, s, y)
                sr = stripe(sub_beta_area, W + 1 - w, w - 1, (1, w), 0) # (B, kx, s, y)
#                 beta1, k1 = (sl + sr).permute(0, 3, 1, 2).max(-1) # (B, y, kx, w - 1)
                beta1 = (sl + sr).permute(0, 3, 1, 2)
                
                # case 2: above-below composition
                sub_beta_area = beta_area[..., x, x + w]
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
        
        marginals = marginals.permute(0, 1, 3, 2, 4)
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
            sub_span_area = areas[:, y, y + 1]
            sub_best_area = best_area[:, y, y + 1] # (B, W, W + 1)
            sub_beta_area = beta_area[:, y, y + 1] # (B, W, W + 1)
            new_beta_area, new_best_area = self._mbr_1d_manual(
                sub_best_area, sub_beta_area, sub_span_area, verbose=verbose
            )
            beta_area[:, y, y + 1] = new_beta_area #+ sub_type_area
            best_area[:, y, y + 1] = new_best_area
        
        for x in range(W): # area h x 1
            sub_span_area = areas[..., x, x + 1]
            sub_best_area = best_area[..., x, x + 1]
            sub_beta_area = beta_area[..., x, x + 1]
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
                sub_beta_area = beta_area[:, y, y + h].permute(0, 2, 3, 1)
                sl = stripe(sub_beta_area, W + 1 - w, w - 1, (0, 1), 1) # (B, kx, s, y)
                sr = stripe(sub_beta_area, W + 1 - w, w - 1, (1, w), 0) # (B, kx, s, y)
#                 beta1, k1 = (sl + sr).permute(0, 3, 1, 2).max(-1) # (B, y, kx, w - 1)
                beta1 = (sl + sr).permute(0, 3, 1, 2)
                
                # case 2: above-below composition
                sub_beta_area = beta_area[..., x, x + w]
                sa = stripe(sub_beta_area, H + 1 - h, h - 1, (0, 1), 1) # (B, ky, s, x)
                sb = stripe(sub_beta_area, H + 1 - h, h - 1, (1, h), 0) # (B, ky, s, x)
#                 beta2, k2 = (sa + sb).permute(0, 1, 3, 2).max(-1) # (B, ky, x, h - 1)
                beta2 = (sa + sb).permute(0, 1, 3, 2)
    
                # oh, gotta record which case has been selected; the trick is to use signed split!
    
                # record the best
                beta, k0 = torch.cat((beta1, beta2), dim=-1).max(-1) # (B, y, x, w - 1 + h - 1)
                k1 = (k0 < (w - 1)).long() + (k0 >= (w - 1)) * -1 # which case: horion (1) & vertical (-1)
                k2 = (k0 % (w - 1)).long() + 1 # which split
                best = k1 * k2 # positive split for left-right composition and negative for above-below
                
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

    def _dp_parallel(self, infer=False, require_marginal=False, verbose=False, **kwargs):
        rule_prob, root_prob, term_prob = self.pcfgs
        
        device = rule_prob.device
        NT = rule_prob.shape[1]
        B, L, T = term_prob.shape[:3]
        H = W = int(np.sqrt(L))
        
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
            sub_beta_area = beta_area[:, y, y + 1] # (B, W, W + 1, NT)
            new_beta_area = self._inside_1d_parallel(
                sub_type_area, sub_beta_area, lr_rule_prob, term_prob[:, y], verbose=False
            )
            beta_area[:, y, y + 1] = new_beta_area #+ sub_type_area
        
        for x in range(W): # area h x 1
            sub_type_area = type_area[..., x, x + 1, :]
            sub_beta_area = beta_area[..., x, x + 1, :]
            new_beta_area = self._inside_1d_parallel(
                sub_type_area, sub_beta_area, ab_rule_prob, term_prob[..., x, :]
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
                sub_beta_area = beta_area[:, y, y + h].permute(0, 2, 3, 4, 1)
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
                sub_beta_area = beta_area[..., x, x + w, :].permute(0, 1, 2, 4, 3)
                sa = stripe(sub_beta_area, H + 1 - h, h - 1, (0, 1), 1) # (B, ky, s, r, x)
                sb = stripe(sub_beta_area, H + 1 - h, h - 1, (1, h), 0) # (B, ky, s, r, x)
                beta2 = xyz(sa, sb, ab_Y_Z).permute(0, 2, 3, 1) # (B, S, ky, x) -> (B, ky, x, S)
                
                # summarize betas
                beta = torch.stack((beta1, beta2), dim=-1).logsumexp(-1) # (B, S, y, x)
                
                y = y.repeat_interleave(W + 1 - w)
                x = x.repeat(H + 1 - h)
                beta_area[:, y, y + h, x, x + w] = beta[:, y, x] + type_area[:, y, y + h, x, x + w]
                        
        final = beta_area[:, 0, H, 0, W] + root_prob
        # final = beta_area[:, 0, H, 0, W]
        ll = final.logsumexp(-1)
        
        argmax = (
            None if not infer else self._extract_parse(ll, type_area, **kwargs)
        )
        
        if not require_marginal:
            return ll, argmax, None, {"argmax": argmax, "marginal": None}
        
        marginal = self._compute_marginal(ll, type_area, **kwargs)
        return ll, argmax, marginal, {"argmax": argmax, "marginal": marginal}
        
    def _dp_serial(self, infer=False, require_marginal=False, verbose=False, **kwargs):
        rule_prob, root_prob, term_prob = self.pcfgs
        
        device = rule_prob.device
        NT = rule_prob.shape[1]
        B, L, T = term_prob.shape[:3]
        H = W = int(np.sqrt(L))
        
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
            sub_beta_area = beta_area[:, y, y + 1] # (B, W, W + 1, NT)
            new_beta_area = self._inside_1d_serial(
                sub_type_area, sub_beta_area, lr_rule_prob, term_prob[:, y], verbose=False
            )
            beta_area[:, y, y + 1] = new_beta_area #+ sub_type_area
        
        for x in range(W): # area h x 1
            sub_type_area = type_area[..., x, x + 1, :]
            sub_beta_area = beta_area[..., x, x + 1, :]
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
        
        argmax = (
            None if not infer else self._extract_parse(ll, type_area, **kwargs)
        )
        
        if not require_marginal:
            return ll, argmax, None, {"argmax": argmax, "marginal": None}
        
        marginal = self._compute_marginal(ll, type_area, **kwargs)
        return ll, argmax, marginal, {"argmax": argmax, "marginal": marginal}