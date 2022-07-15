from omegaconf import OmegaConf
import os, re
from collections import defaultdict
from omegaconf.listconfig import ListConfig

import json
import time
import torch
import random
import numpy as np
from torch import nn, Tensor

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from .raven_solver import Monitor as Meta
from ..model import build_main_model
from ..data import build_clevr_image_data, build_abscene_image_data

from ..util import numel, shorten_name, ExpDecayLR, SlotattnLR
from ..module import LARS, exclude_bias_or_norm, adjust_learning_rate

class Monitor(Meta):
    def __init__(self, cfg, echo, device):
        super(Monitor, self).__init__(cfg, echo, device)

    def show_batch(self, batch):
        pass

    def build_data(self):
        self.dataloader, self.evalloader, self.testloader, \
        self.encoder_vocab, self.decoder_vocab, self.cate_vocab = \
        eval(f"build_{self.cfg.data.name.lower()}_image_data")(
            self.cfg.data, not self.cfg.eval, self.echo
        )

    def make_batch(self, sample):
        image = sample["image"]
        image = torch.from_numpy(image).to(self.device)
        image = image.permute(0, 3, 1, 2)

        mask = sample["mask"]
        mask = torch.from_numpy(mask).to(self.device)

        return {"image": image, "mask": mask, "vfile": sample["vfile"]}

    def epoch(self, iepoch):
        self.model.reset()
        all_time = defaultdict(list)
        self.timeit(all_time)        
        device_ids = [i for i in range(self.cfg.num_gpus)]
        nchunk = dist.get_world_size() if torch.distributed.is_initialized() else 1  
        warmup_step_rate = max(self.cfg.optimizer.warmup_steps // self.cfg.optimizer.warmup_times, 1)
        def do_batch(batch, step):
            epoch_step = step #% len(self.dataloader)

            #self.show_batch(batch, meta)

            batch_dict = self.make_batch(batch) 

            self.optim_step += 1 
            bsize = batch_dict["image"].shape[0]
            force_eval, warmup = self.pre_step(step, warmup_step_rate)

            self.timeit(all_time, key="data")

            batch_dict = self.make_batch(batch)

            loss, _ = self.model(**batch_dict)
            loss.backward()
            if self.optim_step % self.cfg.running.optim_rate == 0: 
                self.step()

            self.timeit(all_time, key="model")

            self.num_sents += bsize 

            self.total_step += 1
            self.epoch_step += 1
            self.total_loss += loss.detach()
            self.epoch_loss += loss.detach()
            self.total_inst += bsize * nchunk

            criteria = self.post_step(
                iepoch, epoch_step, force_eval, warmup, nchunk, 
            )

            self.timeit(all_time, key="report")
            return criteria

        for epoch_step, batch_data in enumerate(self.dataloader):
            if epoch_step < 9544:
                pass #continue
            criteria = do_batch(batch_data, epoch_step + 1)

        if not self.cfg.optimizer.use_lars and not self.cfg.optimizer.batch_sch and \
            self.scheduler is not None:
            self.scheduler.step()
        self.timeit(all_time, show=True)

        #self.total_step = self.total_loss = self.total_inst = 0
        #self.start_time = time.time()
        return criteria 

    def infer(self, dataloader, samples=float("inf"), iepoch=0):
        report = self.main_eval(dataloader, samples=samples, iepoch=iepoch)
        #self.echo(f"EVAL {report}")
        return ""

    def evaluate(self, dataloader, samples=float("inf"), iepoch=0):
        report = self.main_eval(dataloader, samples=samples, iepoch=iepoch)
        #self.echo(f"TEST {report}")
        return ""
        
    def main_eval(self, dataloader, samples=float("inf"), iepoch=0):

        if isinstance(samples, (tuple, list, ListConfig)): 
            samples = list(samples)
            samples = samples[1] - samples[0]
        elif samples is None:
            samples = float("inf")

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

            #self.show_batch(batch)

            batch_dict = self.make_batch(batch)
            bsize = batch_dict["image"].shape[0]

            batch_dict["infer"] = True
            batch_dict["analyze"] = True

            loss_mean, (ntoken, _) = self.model(**batch_dict)
            loss = loss_mean * ntoken

            epoch_step += 1
            total_word += ntoken 

            nsample += bsize * nchunk
            losses += loss or 0.
            if self.cfg.rank == 0 and (ibatch + 1) % peep_rate == 0:
                self.echo(
                    f"step {istep} / {ibatch}\t" + #gnorm {grad_norm():.2f} " +
                    f"loss {losses / total_word:.8f} " # (istep + 0):.8f} " # (ibatch + 1):.8f} " +
                    f"{nsample / (time.time() - start_time):.2f} samples/s"
                )

        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        self.echo(f"# sample {nsample}; {nsample / (time.time() - start_time):.2f} samples/s")
        stats = model.stats()
        if stats != "": # could be empty
            self.echo(f"EVAL STATS: {model.stats()}")
        return model.report()
