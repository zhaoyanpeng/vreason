import numpy as np
import os, sys, time, math
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

from fvcore.common.registry import Registry

LOSS_HEADS_REGISTRY = Registry("LOSS_HEADS")
LOSS_HEADS_REGISTRY.__doc__ = """
Registry for encoder heads.
"""

def build_loss_head(cfg, vocab, **kwargs):
    return LOSS_HEADS_REGISTRY.get(cfg.name)(cfg, vocab, **kwargs)

class MetaLossHead(nn.Module):
    def __init__(self, cfg, token_vocab):
        super().__init__()
        pass
    def infer(self):
        pass
    def report(self):
        return ""

@LOSS_HEADS_REGISTRY.register()
class DummyLossHead(MetaLossHead):
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, token_vocab)
        pass
    def _reset_parameters(self):
        pass
    def output_size(self):
        return 0 
    def forward(self, *args, **kwargs):
        return None, None 

@LOSS_HEADS_REGISTRY.register()
class LMLossHead(MetaLossHead):
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, token_vocab)
        self.token_vocab = token_vocab
        self.logit_scale = (
            nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) if cfg.scaling else
            torch.ones([], requires_grad=False) * np.log(1 / 1)
        )
        self.ignore_index = self.token_vocab.PAD_IDX
        self.loss_fn = nn.CrossEntropyLoss(
            reduction="none", ignore_index=self.ignore_index
        )
        self.accuracies = {word: [0] * 2 for word in ["overall"]}
        self.infer_mode = cfg.infer_mode
        self.reduce = False 

    def report(self, gold_file=None):
        # compute accuracies, called every epoch
        result = " ".join(
            ["ACC:"] + [f"{k}: {(v[0] / v[1]) * 100:7.3f} (101x)" for k, v in self.accuracies.items()]
        )
        self.accuracies = {k: [0] * 2 for k, _ in self.accuracies.items()} # reset
        return result 

    def _estimate_loss(self, logits, x2, *args, **kwargs):
        losses = self.loss_fn(
            logits.reshape(-1, logits.shape[-1]), x2.reshape(-1)
        ) 
        losses = losses.view(x2.size())

        loss_sum = losses.sum() 
        ntoken = (x2 != self.ignore_index).sum()
        loss = (loss_sum / ntoken) if ntoken > 0 else loss_sum
        return loss, (ntoken, losses)

    def infer(self, x1, x2, *args, **kwargs): 
        results = self._estimate_loss(x1, x2)

        def calculate_acc(x1, x2, key="overall"):
            x1 = x1.argmax(dim=-1).reshape(-1)
            x2 = x2.reshape(-1)
            mask = x2 != self.ignore_index
            if key not in self.accuracies:
                self.accuracies[key] = [0., 0.]
            metric = self.accuracies[key]
            metric[0] += ((x1 == x2) * mask).sum() 
            metric[1] += mask.sum()

        # overall accuracy
        calculate_acc(x1, x2)

        # individual accuracy
        if self.infer_mode == "ll" and x1.dim() == 4: # hack
            K = x1.shape[1]
            for i in range(K):
                a = x1[:, i]
                b = x2[:, i]
                calculate_acc(a, b, key=f"acc{i}")
        return results 

    def predict(self, x1, x2, negative):
        diff = x2.shape[-1] - negative.shape[-1]
        if diff == 2:
            sli = slice(1, -1)  
        elif diff == 1:
            sli = slice(None, -1)  
        else:
            sli = slice(None)  
        if self.infer_mode == "ll" and x1.dim() == 4: # hack
            x1 = x1.log_softmax(-1)
            log_probs = torch.gather(x1, -1, x2.unsqueeze(-1))
            log_probs = log_probs.squeeze(-1)[..., sli].sum(-1)

            prediction = log_probs.argmax(dim=-1)

            ntrue = (prediction == 0).sum()
            return x1.shape[0], ntrue
        else:
            x1 = x1[:, sli].argmax(dim=-1) # (B, L)
            x2 = x2[:, sli].unsqueeze(1)   # (B, L) 

            labels = torch.cat([x2, negative], dim=1) # (B, K, L)

            logits = (x1.unsqueeze(1) == labels).sum(-1) 
            prediction = logits.argmax(dim=-1)

            ntrue = (prediction == 0).sum()
            return x1.shape[0], ntrue

    def forward(self, x1, x2, *args, negative=None, **kwargs):
        nsample, ntrue = self.predict(x1, x2, negative)
        if not self.training:
            loss, (ntoken, losses) = self.infer(x1, x2, *args, **kwargs)
            extra = {"ntoken": ntoken, "main_loss": loss, "nsample": nsample, "ntrue": ntrue}
            return loss, (ntoken, extra)
        logits = self.logit_scale.exp() * x1
        loss, (ntoken, losses) = self._estimate_loss(logits, x2, *args, **kwargs)
        extra = {"ntoken": ntoken, "main_loss": loss, "nsample": nsample, "ntrue": ntrue}
        return loss, (ntoken, extra)

@LOSS_HEADS_REGISTRY.register()
class MSELossHead(MetaLossHead):
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, token_vocab)
        self.loss_fn = nn.MSELoss()
        self.reduce = False 

    def report(self, gold_file=None):
        # compute accuracies, called every epoch
        result = ""
        return result 

    def _estimate_loss(self, logits, x2, *args, **kwargs):
        loss = self.loss_fn(logits, x2) 
        losses = loss.detach() * x2.shape[0]
        return loss, (x2.shape[0], losses.item())

    def infer(self, x1, x2, *args, **kwargs): 
        results = self._estimate_loss(x1, x2)
        return results 

    def predict(self, x1, x2, negative):
        pass

    def forward(self, x1, x2, *args, negative=None, **kwargs):
        if not self.training:
            loss, (ntoken, losses) = self.infer(x1, x2, *args, **kwargs)
            extra = {"nsample": ntoken, "main_loss": losses}
            return loss, (ntoken, extra)
        loss, (ntoken, losses) = self._estimate_loss(x1, x2, *args, **kwargs)
        extra = {"nsample": ntoken, "main_loss": losses}
        return loss, (ntoken, extra)

@LOSS_HEADS_REGISTRY.register()
class CoupleLMLossHead(MetaLossHead):
    def __init__(self, cfg, token_vocab, **kwargs):
        super().__init__(cfg, token_vocab)
        self.token_vocab = token_vocab
        self.logit_scale = (
            nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) if cfg.scaling else
            torch.ones([], requires_grad=False) * np.log(1 / 1)
        )
        self.ignore_index = -100 #self.token_vocab.PAD_IDX
        self.loss_fn = nn.CrossEntropyLoss(
            reduction="none", ignore_index=self.ignore_index
        )
        self.accuracies = {word: [0] * 2 for word in ["overall"]}
        self.infer_mode = cfg.infer_mode
        self.vloss_beta = cfg.vloss_beta
        self.reduce = False 

    def report(self, gold_file=None):
        # compute accuracies, called every epoch
        result = ""
        return result 

    def _estimate_loss(self, logits, x2, *args, **kwargs):
        losses = self.loss_fn(
            logits.reshape(-1, logits.shape[-1]), x2.reshape(-1)
        ) 
        losses = losses.view(x2.size())

        loss_sum = losses.sum() 
        ntoken = (x2 != self.ignore_index).sum()
        loss = (loss_sum / ntoken) if ntoken > 0 else loss_sum
        return loss, (ntoken, losses)

    def infer(self, x1, x2, *args, **kwargs): 
        results = self._estimate_loss(x1, x2)
        return results 

    def predict(self, x1, x2):
        pred = x1.reshape(-1, x1.shape[-1]).argmax(-1)
        mask = x2 != self.ignore_index
        ntrue = ((pred == x2.view(-1)) * mask.view(-1)).sum()
        return x1.shape[0], ntrue

    def main_forward(self, x1, x2, *args, flag="", **kwargs):
        nsample, ntrue = self.predict(x1, x2)
        if not self.training:
            loss, (ntoken, losses) = self.infer(x1, x2, *args, **kwargs)
            extra = {f"{flag}ntoken": ntoken, f"{flag}main_loss": loss, f"{flag}nsample": nsample, f"{flag}ntrue": ntrue}
            return loss, (ntoken, extra)
        logits = self.logit_scale.exp() * x1
        loss, (ntoken, losses) = self._estimate_loss(logits, x2, *args, **kwargs)
        extra = {f"{flag}ntoken": ntoken, f"{flag}main_loss": loss, f"{flag}nsample": nsample, f"{flag}ntrue": ntrue}
        return loss, (ntoken, extra)

    def forward(self, x1, x2, *args, **kwargs):
        assert isinstance(x1, (list, tuple))
        assert isinstance(x2, (list, tuple))
        assert len(x1) == len(x2), f"x1 and x2 have different numbers of items"
        
        t_loss, (t_ntoken, t_extra) = self.main_forward(x1[0], x2[0], flag="t_")
        v_loss, (v_ntoken, v_extra) = self.main_forward(x1[1], x2[1], flag="v_")

        loss = (t_loss + v_loss * self.vloss_beta) / (self.vloss_beta + 1)
        ntrue = t_extra["t_ntrue"] + v_extra["v_ntrue"]
        nsample = t_extra["t_nsample"]
        ntoken = t_ntoken + v_ntoken

        extra = {"nstep": 1, "ntoken": ntoken, "main_loss": loss, "nsample": nsample, "ntrue": ntrue}
        extra.update(t_extra)
        extra.update(v_extra)
        return loss, (ntoken, extra) 
