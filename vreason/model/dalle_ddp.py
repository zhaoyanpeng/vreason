import math
import copy
import os, re
import torch
import numpy as np
from torch import autocast, nn, Tensor

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import data_parallel

from . import from_pretrained_vqgan, load_checkpoint, Dalle 
from ..util import is_main_process, get_rank 


class DalleDDP(Dalle):
    """ A shared embedder (b/w encoder and decoder) to mimic GPT 
        with bi-directional memory or an encoder-only GPT model.
    """
    def __init__(self, cfg, echo):
        super().__init__(cfg, echo)

    def _seq2seq(
        self, t, v, t_seq=None, v_seq=None, text_mask=None, 
        enc_cache=None, dec_cache=None, use_cache=False, **kwargs,
    ):
        memory, _, enc_cache, enc_extra = self.encoder_head(
            t, v, t_seq=t_seq, v_seq=v_seq, cache=enc_cache, use_cache=use_cache,
            self_key_padding_mask=text_mask, **kwargs
        )
        logits, targets, dec_cache, dec_extra = self.decoder_head(
            t, v, t_seq=t_seq, v_seq=v_seq, cache=dec_cache, use_cache=use_cache,
            memo_key_padding_mask=text_mask, memory=memory, **kwargs
        )
        return logits, targets, enc_cache, dec_cache, enc_extra, dec_extra

    def forward(self, text=None, image=None, text_mask=None, analyze=False, device_ids=[0], **kwargs):
        v_seq = self.tokenize_images(image) if image.dim() == 4 else image # discretize images 
        t_emb, v_emb, emb_extra = self.embedder_head(text, v_seq=v_seq)

        logits, targets, *_, enc_extra, dec_extra = self._seq2seq(
            t_emb, v_emb, t_seq=text, v_seq=v_seq, text_mask=text_mask, sampling=False, **kwargs,
        )
        
        loss, outs, *_ = self.loss_head(logits, targets)
        self.set_stats({}, outs[-1]) # private statistics

        v_peep_topk = kwargs.get("v_peep_topk", 0)
        if (v_peep_topk is None or v_peep_topk > 0) and is_main_process(): # master's job: keep it simpler
            with torch.autocast("cuda", enabled=False): # has to be disabled when using cache at test time
                extra_outs = self.peep_dp(
                    text=text, image=image, text_mask=text_mask, analyze=analyze, device_ids=device_ids, **kwargs
                )
            outs[-1].update(extra_outs)
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
            t_emb, v_emb, emb_extra = self.embedder_head(text, v_seq=c_seq)
            logits, *_ = self._seq2seq(
                t_emb, v_emb, t_seq=text, v_seq=c_seq, force_infer=True, **kwargs,
            )
            logits = logits[-1][:, -1] / temperature
            logits = logits[:, :self.vq.vocab_size].detach().clone() # (B, V)
            if topk is not None:
                logits = self._topk(logits, topk)
            pdists = F.softmax(logits, dim=-1)
            if random:
                idx = torch.multinomial(pdists, num_samples=1)
            else:
                *_, idx = torch.topk(pdists, k=1, dim=-1)
            c_seq = torch.cat((c_seq, idx), dim=1)
        return c_seq
    
    @torch.no_grad()
    def peep_dp(
        self, text=None, image=None, text_mask=None, analyze=False, device_ids=[0], v_peep_topk=None, **kwargs
    ):
        text = text[:v_peep_topk]
        image = image[:v_peep_topk]
        v_seq = self.tokenize_images(image) if image.dim() == 4 else image# discretize images 

        B, L = v_seq.shape[:2]

        # given a half
        c_seq = v_seq[:, :(L - 0) // 2]
        nstep = L - c_seq.shape[1]
        v = self.sampling(text, c_seq, nstep, device_ids=device_ids, random=True)
        sample_half = self.generate_images(v)
        
        # sampling 
        c_seq = v_seq[:, :0]
        nstep = L - c_seq.shape[1]
        v = self.sampling(text, c_seq, nstep, device_ids=device_ids, random=True)
        sample_full = self.generate_images(v)

        # deterministic
        c_seq = v_seq[:, :0]
        nstep = L - c_seq.shape[1]
        v = self.sampling(text, c_seq, nstep, device_ids=device_ids, random=False)
        sample_hard = self.generate_images(v)

        # reconstruction
        sample_pred = self.generate_images(v_seq)
        
        outs = {
            "_sample_half": sample_half,
            "_sample_full": sample_full,
            "_sample_hard": sample_hard,
            "_sample_pred": sample_pred,
            "nsample": v_peep_topk,
        }
        return outs
