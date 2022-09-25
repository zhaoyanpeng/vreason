import math
import copy
import os, re
import torch
import numpy as np
from torch import autocast, nn, Tensor

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import data_parallel

from ..module import build_encoder_head


class DalleMiniVQEncoder(nn.Module):
    def __init__(self, cfg, echo):
        super().__init__()
        self.echo = echo
        self.cfg = cfg

    def stats(self): 
        return ""

    def reset(self):
        pass
    
    def report(self, **kwargs):
        return ""

    def collect_state_dict(self):
        pass

    def forward(self, image=None, text_mask=None, device_ids=[0], **kwargs):
        #v_seq = self.tokenize_images(image) # discretize images 
        v_seq = data_parallel(self.vq, image, device_ids=device_ids)
        return None, (v_seq, None)

    def tokenize_images(self, v):
        return self.vq.encode(v)

    def generate_images(self, v):
        return self.vq.decode(v)

    def register_vq(self, mcfg):
        vq = build_encoder_head(mcfg.vq, None)
        return vq.emb_weight, vq

    def build(self, encoder_vocab, decoder_vocab=None, **kwargs):
        assert self.cfg.eval, "VQEncoder supports only eval mode"
        tunable_params = dict()
        mcfg = self.cfg.model 
        *_, self.vq = self.register_vq(mcfg)
        self.cuda(self.cfg.rank)
        return tunable_params
