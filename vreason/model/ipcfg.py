import math
import os, re
import torch
import numpy as np
from torch import autocast, nn, Tensor

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import data_parallel

from . import from_pretrained_vqgan, load_checkpoint, Dalle 
from ..util import Stats, enable_print, disable_print, top_k_top_p_filtering
from ..module import build_encoder_head, build_decoder_head, build_loss_head


class IPCFG(Dalle):
    def __init__(self, cfg, echo):
        super().__init__(cfg, echo)

    def infer(self, text=None, image=None, text_mask=None, analyze=False, device_ids=[0], **kwargs):
        pass

    def infer_dp(self, text=None, image=None, text_mask=None, analyze=False, device_ids=[0], **kwargs):
        pass

    def forward_dp(self, text=None, image=None, text_mask=None, analyze=False, device_ids=[0], **kwargs):
        t_emb = v_emb = None
        v_seq = data_parallel(self.vq, image, device_ids=device_ids)
        #*_ = self.embedder_head(text, v_seq=v_seq)
        #*_ = self.encoder_head(t_emb, v_emb, t_seq=text, v_seq=v_seq, self_key_padding_mask=text_mask, **kwargs)

        kwargs.update({
            "t_seq": text, "v_seq": v_seq,
            "self_key_padding_mask": text_mask,
            "memo_key_padding_mask": text_mask,
        })
        ll, kl, targets, argmax, marginal, dec_extra = data_parallel(
            self.decoder_head, (t_emb, v_emb), device_ids=device_ids, module_kwargs=kwargs
        )
        
        loss, outs, *_ = self.loss_head(targets, ll, kl=kl)
        self.set_stats(dec_extra, outs[-1])

        if analyze:
            self.analyze(
                t_emb=t_emb, v_emb=v_emb, t_seq=text, v_seq=v_seq, ll=ll, kl=kl,
                targets=targets, argmax=argmax, marginal=marginal
            )
        return loss, outs 

    def forward(self, text=None, image=None, text_mask=None, analyze=False, device_ids=[0], **kwargs):
        infer = kwargs.get("infer", False)
        if infer and not self.training:
            inference_fn = self.infer_dp if len(device_ids) > 1 else self.infer
            with torch.autocast("cuda", enabled=False): # has to be disabled when using cache at test time
                return inference_fn(
                    text=text, image=image, text_mask=text_mask, analyze=analyze, device_ids=device_ids, **kwargs
                )
        if len(device_ids) > 1:
            return self.forward_dp(
                text=text, image=image, text_mask=text_mask, analyze=analyze, device_ids=device_ids, **kwargs
            )
        
        t_emb = v_emb = None
        v_seq = self.tokenize_images(image)
        #*_ = self.embedder_head(text, v_seq=v_seq)
        #*_ = self.encoder_head(t_emb, v_emb, t_seq=text, v_seq=v_seq, self_key_padding_mask=text_mask, **kwargs)

        ll, kl, targets, argmax, marginal, dec_extra = self.decoder_head(
            t_emb, v_emb, t_seq=text, v_seq=v_seq, memo_key_padding_mask=text_mask, **kwargs
        )
        
        loss, outs, *_ = self.loss_head(targets, ll, kl=kl)
        self.set_stats(dec_extra, outs[-1])

        if analyze:
            self.analyze(
                t_emb=t_emb, v_emb=v_emb, t_seq=text, v_seq=v_seq, ll=ll, kl=kl,
                targets=targets, argmax=argmax, marginal=marginal
            )
        return loss, outs 

    def stats(self): 
        meter = self.meter_train if self.training else self.meter_infer
        stats = meter.stats

        nstep = stats["nstep"]
        alpha = 1 / nstep if nstep > 0 else 0
        loss = stats["main_loss"] * alpha 

        nsample = stats["nsample"]
        alpha = 1 / nsample * 1 if nsample > 0 else 0
        ll = stats["ll"] * alpha * -1
        kl = stats["kl"] * alpha

        ntoken = stats["ntoken"]
        alpha = 1 / ntoken * 1 if ntoken > 0 else 0
        elbo = np.exp((stats["kl"] - stats["ll"]) * alpha)
        ppl = np.exp(-stats["ll"] * alpha) 

        info = f"loss {loss:.5f} elbo {elbo:.3f} ppl {ppl:.3f} ll {ll:.3f} kl {kl:.4f}"

        info = f"{info} ({nsample})"
        return info

    def eval_metric(self):
        meter = self.meter_train if self.training else self.meter_infer
        stats = meter.stats

        ntoken = stats["ntoken"]
        alpha = 1 / ntoken * 1 if ntoken > 0 else 0
        ppl = np.exp(-stats["ll"] * alpha) 
        return -ppl 
