from omegaconf import OmegaConf
import os, re
from collections import defaultdict

import json
import time
import torch
import random
import numpy as np
from torch import nn, Tensor

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from .meta import Monitor as Meta 
from ..model import build_main_model
from ..data import build_raven_for_vqgan_encoder as build_data

class Monitor(Meta):
    def __init__(self, cfg, echo, device):
        super(Monitor, self).__init__()
        self.cfg = cfg
        self.echo = echo
        self.device = device
        self.build_data()
        model = build_main_model(cfg, echo)
        tunable_params = model.build()
        self.model = DistributedDataParallel(
            model, device_ids=[cfg.rank], find_unused_parameters=True
        ) if torch.distributed.is_initialized() else model 
        self.model.train(not cfg.eval)
        self.build_optimizer(tunable_params)

    def show_batch(self, batch, meta):
        pass

    def build_data(self):
        dcfg = self.cfg.data
        root = f"{dcfg.data_root}/{dcfg.name}"
        nsample = dcfg.test_samples if self.cfg.eval else dcfg.train_samples 
        self.dataloader = build_data(dcfg, root, dcfg.splits, dcfg.tasks, int(nsample), self.echo)

        for task in dcfg.tasks:
            save_path = f"{dcfg.save_root}/{task}" 
            if not os.path.exists(save_path):
                os.makedirs(save_path)

    def make_batch(self, sample):
        task, fname = sample["file_path_"].split("/")[-2:]
        save_file = f"{self.cfg.data.save_root}/{task}/{fname}"

        x = torch.from_numpy(sample["image"]).to(self.device)

        return x, save_file[:-4], sample["target"]

    def pre_step(self, step, warmup_step_rate, inc=0):
        pass

    def post_step(
        self, iepoch, epoch_step, force_eval, warmup, nchunk, num_batch
    ):
        pass

    def build_optimizer(self, tunable_params={}, verbose=True):
        pass

    def epoch(self, iepoch):
        return 0.
        
    def evaluate(self, dataloader, samples=float("inf"), iepoch=0):
        self.model.reset()
        losses, istep, nsample, nchunk, nbatch = 0, 0, 0, 1, len(dataloader)
        device_ids = [i for i in range(self.cfg.num_gpus)]
        if isinstance(self.model, DistributedDataParallel):
            dataloader.sampler.set_epoch(iepoch)
            nchunk = self.cfg.num_gpus
        peep_rate = max(10, (len(dataloader) // 10))
        epoch_step = total_word = 0
        start_time = time.time()
        for ibatch, batch in enumerate(dataloader):
            if nsample >= samples:
                #print(f"{nsample}\t{ibatch}/{nbatch} continue")
                break #continue # iterate through every batch

            #self.show_batch(batch, meta)

            x, save_file, target = self.make_batch(batch)

            loss, indice = self.model(x) 

            np.savez_compressed(save_file, target=target, indice=indice)

            epoch_step += 1
            total_word += 1 

            nsample += x.shape[0] * nchunk
            losses += loss or 0.
            if self.cfg.rank == 0 and (ibatch + 1) % peep_rate == 0:
                self.echo(
                    f"step {istep} / {ibatch}\t" + #gnorm {grad_norm():.2f} " +
                    f"loss {losses / total_word:.8f} " # (istep + 0):.8f} " # (ibatch + 1):.8f} " +
                    f"{nsample / (time.time() - start_time):.2f} samples/s"
                )

        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        self.echo(f"# sample {nsample}; {nsample / (time.time() - start_time):.2f} samples/s")
        return model.report()
