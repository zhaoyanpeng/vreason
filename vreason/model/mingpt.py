import math
import os, re, copy
import torch
import numpy as np
from torch import autocast, nn, Tensor

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import data_parallel

from . import from_pretrained_vqgan, load_checkpoint, Dalle 
from ..util import Stats, enable_print, disable_print, top_k_top_p_filtering
from ..module import build_encoder_head, build_decoder_head, build_loss_head

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class MinIGPT(Dalle):
    """ The same as the Dalle model but w/ text tokens removed in order to get an image gpt model.
    """
    def __init__(self, cfg, echo):
        super().__init__(cfg, echo)

    def _seq2seq(
        self, t, v, t_seq=None, v_seq=None, text_mask=None, 
        enc_cache=None, dec_cache=None, use_cache=False, **kwargs,
    ):
        logits, targets, dec_cache, dec_extra = self.decoder_head(
            t, v, t_seq=t_seq, v_seq=v_seq, cache=dec_cache, use_cache=use_cache,
            memo_key_padding_mask=text_mask, **kwargs
        )
        enc_cache = enc_extra = None
        return logits, targets, enc_cache, dec_cache, enc_extra, dec_extra

    def _seq2seq_dp(self, t, v, enc_cache=None, dec_cache=None, device_ids=[0], **kwargs):
        logits, targets, dec_cache, dec_extra = data_parallel(
            self.decoder_head, (t, v), device_ids=device_ids, module_kwargs=kwargs
        )
        enc_cache = enc_extra = None
        return logits, targets, enc_cache, dec_cache, enc_extra, dec_extra

    def forward_dp(self, text=None, image=None, text_mask=None, analyze=False, device_ids=[0], **kwargs):
        v_seq = data_parallel(self.vq, image, device_ids=device_ids)
        v_seq = F.pad(v_seq, (1, 0), value=0) # BOS of visual seq

        kwargs.update({"t_seq": text, "v_seq": v_seq})
        logits, targets, *_, enc_extra, dec_extra = self._seq2seq_dp(
            None, None, device_ids=device_ids, **kwargs
        )
        logits, targets = logits[0], targets[0]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        B, L = v_seq.shape[:2]
        self.set_stats({}, {
            "main_loss": loss.detach().item() * B, "nsample": B, "ntoken": B * (L - 1), "nstep": 1
        })

        outs = (B * (L - 1), {})
        v_peep_topk = kwargs.get("v_peep_topk", 0)
        if v_peep_topk is None or v_peep_topk > 0:
            with torch.autocast("cuda", enabled=False): # has to be disabled when using cache at test time
                outs = self.peep_dp(
                    text=text, image=image, text_mask=text_mask, analyze=analyze, device_ids=device_ids, **kwargs
                )
            outs = (B * (L - 1), outs)
        return loss, outs 

    def forward(self, text=None, image=None, text_mask=None, analyze=False, device_ids=[0], **kwargs):
        text = text[:, :0] # removing text tokens results in an image gpt model -- the only diff. from Dalle model
        infer = kwargs.get("infer", False)
        peep_image = kwargs.get("peep_image", False)
        if False and infer and not self.training: # FIXME disabled
            inference_fn = self.infer_dp if len(device_ids) > 1 else self.infer
            with torch.autocast("cuda", enabled=False): # has to be disabled when using cache at test time
                return inference_fn(
                    text=text, image=image, text_mask=text_mask, analyze=analyze, device_ids=device_ids, **kwargs
                )
        if len(device_ids) > 1:
            return self.forward_dp(
                text=text, image=image, text_mask=text_mask, analyze=analyze, device_ids=device_ids, **kwargs
            )

        v_seq = self.tokenize_images(image) # discretize images 
        v_seq = F.pad(v_seq, (1, 0), value=0) # BOS of visual seq
        
        logits, targets, *_, enc_extra, dec_extra = self._seq2seq(
            None, None, t_seq=text, v_seq=v_seq, text_mask=text_mask, **kwargs,
        )
        logits, targets = logits[0], targets[0]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        B, L = v_seq.shape[:2]
        self.set_stats({}, {
            "main_loss": loss.detach().item() * B, "nsample": B, "ntoken": B * (L - 1), "nstep": 1
        })
        
        outs = (B * (L - 1), {})
        v_peep_topk = kwargs.get("v_peep_topk", 0)
        if v_peep_topk is None or v_peep_topk > 0:
            with torch.autocast("cuda", enabled=False): # has to be disabled when using cache at test time
                outs = self.peep_dp(
                    text=text, image=image, text_mask=text_mask, analyze=analyze, device_ids=device_ids, **kwargs
                )
            outs = (B * (L - 1), outs)
        return loss, outs 

    @staticmethod
    def _topk(x, k):
        v, _ = torch.topk(x, k)
        o = x.clone()
        o[o < v[..., [-1]]] = -float("inf")
        return o

    def sampling(
        self, text, c_seq, nstep, temperature=1., topk=100, random=True, device_ids=[0], **kwargs
    ):
        kwargs = copy.deepcopy(kwargs)
        for k in range(nstep):
            kwargs.update({"t_seq": text, "v_seq": c_seq, "force_infer": True})
            logits, *_ = self._seq2seq_dp(
                None, None, device_ids=device_ids, **kwargs
            )
            logits = logits[-1][:, -1] / temperature
            if topk is not None:
                logits = self._topk(logits, topk)
            pdists = F.softmax(logits, dim=-1)
            if random:
                idx = torch.multinomial(pdists, num_samples=1)
            else:
                *_, idx = torch.topk(pdists, k=1, dim=-1)
            c_seq = torch.cat((c_seq, idx), dim=1)
        return c_seq[:, 1:] # rm BOS
    
    @torch.no_grad()
    def peep_dp(
        self, text=None, image=None, text_mask=None, analyze=False, device_ids=[0], v_peep_topk=None, **kwargs
    ):
        text = text[:v_peep_topk]
        image = image[:v_peep_topk]
        v_seq = data_parallel(self.vq, image, device_ids=device_ids)
        v_seq = F.pad(v_seq, (1, 0), value=0) # BOS of visual seq

        B, L = v_seq.shape[:2]

        # given a half
        c_seq = v_seq[:, :(L - 1) // 2 + 1]
        nstep = L - c_seq.shape[1]
        v = self.sampling(text, c_seq, nstep, device_ids=device_ids, random=True)
        sample_half = self.generate_images(v)
        
        # sampling 
        c_seq = v_seq[:, :1]
        nstep = L - c_seq.shape[1]
        v = self.sampling(text, c_seq, nstep, device_ids=device_ids, random=True)
        sample_full = self.generate_images(v)

        # deterministic
        c_seq = v_seq[:, :1]
        nstep = L - c_seq.shape[1]
        v = self.sampling(text, c_seq, nstep, device_ids=device_ids, random=False)
        sample_hard = self.generate_images(v)

        # reconstruction
        sample_pred = self.generate_images(v_seq[:, 1:])
        
        outs = {
            "_sample_half": sample_half,
            "_sample_full": sample_full,
            "_sample_hard": sample_hard,
            "_sample_pred": sample_pred,
            "nsample": v_peep_topk,
        }
        return outs

    def infer_dp(self, text=None, image=None, text_mask=None, analyze=False, device_ids=[0], **kwargs):
        raise NotImplementedError

    def infer(
        self, text=None, image=None, text_mask=None, analyze=False, 
        device_ids=[0], nsampling=1, topk=0.2, topp=0.995, debug=True, **kwargs
    ):
        return self.infer_dp(
            text=text, image=image, text_mask=text_mask, analyze=analyze, device_ids=device_ids, **kwargs
        )

    def stats(self): 
        meter = self.meter_train if self.training else self.meter_infer
        stats = meter.stats

        nsample = stats["nsample"]
        alpha = 1 / nsample * 1 if nsample > 0 else 0
            
        loss = stats["main_loss"] * alpha 
        info = f"loss {loss:.5f}"

        info = f"{info} ({nsample})"
        return info

    def eval_metric(self):
        assert not self.training, f"valid only in eval mode."
        stats = self.meter_infer.stats

        nsample = stats["nsample"]
        alpha = 1 / nsample * 1 if nsample > 0 else 0

        loss = stats["main_loss"] * alpha 
        return -loss

    def get_optimizer(self):
        return self.decoder_head.configure_optimizers(self.cfg.optimizer)
