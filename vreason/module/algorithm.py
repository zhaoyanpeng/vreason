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
        
    def _compute_marginal(self, ll, area):
        raise NotImplementedError

    def _dp_parallel(self, **kwargs):
        raise NotImplementedError

    def _dp_serial(self, **kwargs):
        raise NotImplementedError
        

class InsideAlg1D(InsideAlg): # standalone class, for sequential inputs

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

class InsideAlg2D(InsideAlg1D): # standalone class, for two-dimensional inputs
    
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
                        sl = beta_area[:, y, y + h, x, x + 1 : x + w] # (B, k, l)
                        sr = beta_area[:, y, y + h, x + 1 : x + w, x + w] # (B, k, r)
                        beta1 = ( # bl, br, bslr
                            sl.unsqueeze(1).unsqueeze(-1) + # (B, 1, k, l, 1)
                            sr.unsqueeze(1).unsqueeze(-2) + # (B, 1, k, l, r)
                            lr_Y_Z.unsqueeze(2) # (B, S, 1, l, r)
                        ).logsumexp((3, 4)) # (B, S, k)

                        # case 2: above-below composition
                        # --------
                        # |      |
                        # |      |
                        # |~~~~~~|
                        # |      |
                        # --------
                        sa = beta_area[:, y, y + 1 : y + h, x, x + w]
                        sb = beta_area[:, y + 1 : y + h, y + h, x, x + w] 
                        beta2 = ( # bl, br, bslr
                            sa.unsqueeze(1).unsqueeze(-1) + # (B, 1, k, l, 1)
                            sb.unsqueeze(1).unsqueeze(-2) + # (B, 1, k, l, r)
                            ab_Y_Z.unsqueeze(2) # (B, S, 1, l, r)
                        ).logsumexp((3, 4)) # (B, S, k)
                        
                        beta = torch.cat((beta1, beta2), dim=-1).logsumexp(-1)
                        beta_area[:, y, y + h, x, x + w] = beta + type_area[:, y, y + h, x, x + w]
                        
        final = beta_area[:, 0, H, 0, W] + root_prob
        # final = beta_area[:, 0, H, 0, W]
        ll = final.logsumexp(-1)
        
        argmax = (
            None if not infer else self._extract_parses(ll, type_area)
        )
        
        if not require_marginal:
            return ll, argmax, None, {}
        
        marginal = self._compute_marginal(ll, type_area)
        return ll, argmax, marginal, {}
    
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
        
#         # copy pre-terminals
#         x = torch.arange(W).repeat(H).to(device)
#         y = torch.arange(H).repeat_interleave(W).to(device)
#         beta_area[:, y, y + 1, x, x + 1] = (
#             type_area[:, y, y + 1, x, x + 1] + term_prob[:, y, x] # NT ? (>, <, or =) T
#         )

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
            None if not infer else self._extract_parses(ll, type_area)
        )
        
        if not require_marginal:
            return ll, argmax, None, {}
        
        marginal = self._compute_marginal(ll, type_area)
        return ll, argmax, marginal, {}
