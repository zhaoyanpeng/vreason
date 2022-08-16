from omegaconf import OmegaConf
import os, re
from collections import defaultdict
from omegaconf.listconfig import ListConfig

import json
import time
import torch
import random
import datetime
import numpy as np
from torch import nn, Tensor

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from .dalle import Monitor as Meta
from ..model import build_main_model
from ..data import build_clevr_image_text_data

from ..util import numel, shorten_name, save_image_local, ExpDecayLR, SlotattnLR
from ..module import LARS, exclude_bias_or_norm, adjust_learning_rate

class Monitor(Meta):
    def build_optimizer(self, tunable_params={}, verbose=True):
        if not self.model.training:
            return
        self.params = (
            list(tunable_params.values())
        )
        for k, v in tunable_params.items():
            if self.cfg.rank == 0:
                pass #self.echo(f"{k} {v.size()}")
        ddp = isinstance(self.model, DistributedDataParallel)
        for k, v in self.model.named_parameters():
            k = re.sub("^module\.", "", k) if ddp else k
            if f"{k}" not in tunable_params:
                v.requires_grad = False
        self.echo(f"# param {numel(self.model) / 1e6:.2f}M # tunable {numel(self.model, True) / 1e6:.2f}M.")
        
        model = self.model.module if ddp else self.model
        self.optimizer = model.get_optimizer() 
        self.scheduler = None

        if not self.cfg.verbose or not verbose:
            return
        self.echo(f"Gradienting The Following Parameters:")
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                self.echo(f"{k} {v.size()}")
        self.echo(f"\n{self.model}")
