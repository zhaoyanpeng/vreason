import math
import os, re
import torch
import numpy as np
from torch import autocast, nn, Tensor

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import data_parallel

from . import from_pretrained_vqgan, load_checkpoint, Dalle 
from ..util import is_dist_avail_and_initialized, is_main_process, get_rank 
from ..util import Stats, enable_print, disable_print, top_k_top_p_filtering
from ..module import build_encoder_head, build_decoder_head, build_loss_head


class IPCFG(Dalle):
    def __init__(self, cfg, echo):
        super().__init__(cfg, echo)

    def forward(
        self, text=None, image=None, text_mask=None, analyze=False, device_ids=[0], **kwargs
    ):
        if len(device_ids) > 1 and not is_dist_avail_and_initialized():
            return self.forward_dp(image=image, device_ids=device_ids, **kwargs)
        return self.forward_ddp(image=image, device_ids=device_ids, **kwargs)

    def forward_dp(self, image=None, device_ids=[0], **kwargs):
        v_seq = (
            data_parallel(self.vq, image, device_ids=device_ids)
            if image.dim() == 4 else image # discretize images
        )

        mean, lvar, *_ = data_parallel(
            self.encoder_head, (v_seq,), device_ids=device_ids, module_kwargs=kwargs
        )

        kwargs.update({"v_seq": v_seq, "mean": mean, "lvar": lvar})
        ll, kl, targets, argmax, marginal, dec_extra = data_parallel(
            self.decoder_head, (None,) * 2, device_ids=device_ids, module_kwargs=kwargs
        )
        
        loss, outs, *_ = self.loss_head(targets, ll, kl=kl)
        self.set_stats({}, outs[-1])
        return loss, outs 

    def forward_ddp(self, image=None, device_ids=[0], infer=False, mbr=False, **kwargs):
        kwargs.update({
            "infer": infer, "auto_infer": False, "exclude_trivial": True,
            "require_marginal": False, "marginal_as_dict": True, "mbr": mbr,
        })

        v_seq = self.tokenize_images(image) if image.dim() == 4 else image

        mean, lvar, *_ = self.encoder_head(v_seq, **kwargs) 

        ll, kl, targets, argmax, marginal, dec_extra = self.decoder_head(
            None, None, v_seq=v_seq, mean=mean, lvar=lvar, **kwargs
        )
        ll1d = dec_extra.get("ll1d", torch.tensor(0).to(ll)) # row- and col-wise ll 

        #print(dec_extra["argmax"][0])
        #print(dec_extra["marginal"][0].shape)
        ##print(dec_extra["marginal"].keys())
        #import sys; sys.exit(0)
        
        loss, outs, *_ = self.loss_head(targets, ll, kl=kl, ll1d=ll1d, pcfgs=self.decoder_head.pcfgs)
        self.set_stats({}, outs[-1])

        if infer: # the saving function of iPCFG
            outs[-1].update({"save_fn": "save_parses", "best": argmax})

        v_peep_topk = kwargs.get("v_peep_topk", 0)
        if (v_peep_topk is None or v_peep_topk > 0) and is_main_process(): # master's job: keep it simpler
            print(f"peep me {v_peep_topk}")
            with torch.autocast("cuda", enabled=False): # has to be disabled when using cache at test time
                extra_outs = self.infer(image=image, device_ids=device_ids, **kwargs)
            outs[-1].update(extra_outs)
        return loss, outs 

    @torch.no_grad()
    def infer(
        self, image=None, device_ids=[0], v_peep_topk=None, **kwargs
    ):
        image = image[:v_peep_topk]
        v_seq = self.tokenize_images(image) if image.dim() == 4 else image 
        
        kwargs.update({
            "infer": True,
            "auto_infer": True,
            "exclude_trivial": True,
            "require_marginal": False,
            "marginal_as_dict": False,
        }) # do not need span marginals
        v_seq = self.tokenize_images(image) if image.dim() == 4 else image
        ll, kl, targets, argmax, marginal, dec_extra = \
            self.decoder_head(None, None, v_seq=v_seq, **kwargs)
        # TODO eval 
        outs = {"nsample": v_peep_topk}
        return outs

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
        
        be = stats["be"] * alpha # rule entropy
        se = stats["se"] * alpha # root entropy
        te = stats["te"] * alpha # term entropy

        lr = stats["lr"] * alpha # lr rule entropy
        ab = stats["ab"] * alpha # ab rule entropy
        la = stats["la"] * alpha # la rule entropy
        
        ll2 = stats["ll2"] * alpha * -1 if "ll2" in stats else None# 1d / 2d ll

        ntoken = stats["ntoken"]
        alpha = 1 / ntoken * 1 if ntoken > 0 else 0
        elbo = np.exp(((stats["kl"] - stats["ll"]) * alpha).cpu())
        ppl = np.exp((-stats["ll"] * alpha).cpu()) 

        #info = f"loss {loss:.5f} elbo {elbo:.3f} ppl {ppl:.3f} ll {ll:.3f} kl {kl:.4f}"
        info = f"elbo {elbo:.3f} ll {ll:.3f} kl {kl:.4f} be {be:.4f} se {se:.3f} te {te:.3f}"
        if ll2 is not None:
            info = f"{info} ll2 {ll2:.3f}" 
        if not self.training:
            info = f"{info} lr {lr:.3f} ab {ab:.3f}" # la {la:.3f}"

        loss_info = self._report() # debug
        info = f"{info} {loss_info}".strip()

        info = f"{info} ({nsample})"
        return info

    def eval_metric(self):
        meter = self.meter_train if self.training else self.meter_infer
        stats = meter.stats

        ntoken = stats["ntoken"]
        alpha = 1 / ntoken * 1 if ntoken > 0 else 0
        ppl = np.exp((-stats["ll"] * alpha).cpu()) 
        return -ppl 

    def _report(self, **kwargs):
        return self.loss_head.report() if self.loss_head is not None else "" 
