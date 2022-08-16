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

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer


class IGPT(Dalle):
    """ The same as the Dalle model but w/ text tokens removed in order to get an image gpt model.
    """
    def __init__(self, cfg, echo):
        super().__init__(cfg, echo)

    def forward(self, text=None, image=None, text_mask=None, analyze=False, device_ids=[0], **kwargs):
        text = text[:, :0] # removing text tokens results in an image gpt model -- the only diff. from Dalle model
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

        v_seq = self.tokenize_images(image) # discretize images 
        t_emb, v_emb, emb_extra = self.embedder_head(text, v_seq=v_seq)

        logits, targets, *_, enc_extra, dec_extra = self._seq2seq(
            t_emb, v_emb, t_seq=text, v_seq=v_seq, text_mask=text_mask, **kwargs,
        )
        
        loss, outs, *_ = self.loss_head(logits, targets)
        self.set_stats(dec_extra, outs[-1])

        v_peep_topk = kwargs.get("v_peep_topk", 0)
        if v_peep_topk > 0 or v_peep_topk is None:
            with torch.autocast("cuda", enabled=False): # has to be disabled when using cache at test time
                extra_outs = self.peep_dp(
                    text=text, image=image, text_mask=text_mask, analyze=analyze, device_ids=device_ids, **kwargs
                )
            outs[-1].update(extra_outs)

        if analyze:
            self.analyze(
                t_emb=t_emb, v_emb=v_emb, t_seq=text, v_seq=v_seq, logits=logits, targets=targets,
            )
        return loss, outs 
