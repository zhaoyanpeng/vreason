import math
import os, re
import torch
from torch import nn

from . import from_pretrained_vqgan

class VQGANEncoder(nn.Module):
    def __init__(self, cfg, echo):
        super().__init__()
        self.cfg = cfg
        self.echo = echo

    def forward(self, x, **kwargs):
        z, _, [_, _, indice] = self.vqgan.encode(x)
        indice = indice.reshape(z.shape[0], -1)
        indice = indice.cpu().numpy()
        loss = torch.tensor(0., device=x.device)
        return loss, indice

    def collect_state_dict(self):
        pass

    def stats(self): 
        return ""

    def reset(self):
        pass
    
    def reduce_grad(optim_rate, sync=False):
        raise NotImplementedError("Gradient Reduce")

    def report(self, gold_file=None):
        return ""

    def init_weights(self) -> None:
        pass
    
    def build(self, **kwargs):
        tunable_params = dict()
        assert self.cfg.eval, "support only eval mode"

        self.vqgan = from_pretrained_vqgan(self.cfg)

        self.cuda(self.cfg.rank)
        return tunable_params
