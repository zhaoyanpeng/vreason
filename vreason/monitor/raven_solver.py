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
from ..data import build_raven_for_gpt as build_data

from ..util import numel, shorten_name
from ..module import LARS, exclude_bias_or_norm, adjust_learning_rate

class Monitor(Meta):
    def __init__(self, cfg, echo, device):
        super(Monitor, self).__init__()
        self.cfg = cfg
        self.echo = echo
        self.device = device
        self.build_data()
        model = build_main_model(cfg, echo)
        tunable_params = model.build(**{
            "encoder_vocab": self.encoder_vocab, 
            "decoder_vocab": self.decoder_vocab,
        })
        self.model = DistributedDataParallel(
            model, device_ids=[cfg.rank], find_unused_parameters=True
        ) if torch.distributed.is_initialized() else model 
        self.model.train(not cfg.eval)
        self.build_optimizer(tunable_params)

    def show_batch(self, batch):
        pass

    def build_data(self):
        self.dataloader, self.evalloader, self.testloader, \
        self.encoder_vocab, self.decoder_vocab = \
        build_data(
            self.cfg.data, not self.cfg.eval, self.echo
        )

    def make_batch(self, sample):
        context = sample["context"]
        positive = sample["positive"]
        negative = sample["negative"]

        negative = torch.from_numpy(negative).to(self.device)
        x = np.concatenate([context, positive], axis=1)
        x = torch.from_numpy(x).to(self.device)

        return {"sequence": x, "negative": negative}

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
            bsize = batch_dict["sequence"].shape[0]
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
        assert isinstance(dataloader, dict), f"expect a dict of dataloaders"
        for task, dloader in dataloader.items():
            report = self.main_eval(dloader, samples=samples, iepoch=iepoch)
            name = shorten_name(task)
            self.echo(f"Task `{name}' {report}")
        return ""

    def evaluate(self, dataloader, samples=float("inf"), iepoch=0):
        assert isinstance(dataloader, dict), f"expect a dict of dataloaders"
        for task, dloader in dataloader.items():
            report = self.main_eval(dloader, samples=samples, iepoch=iepoch)
            name = shorten_name(task)
            self.echo(f"Task `{name}' {report}")
        return ""
        
    def main_eval(self, dataloader, samples=float("inf"), iepoch=0):
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
            bsize = batch_dict["sequence"].shape[0]

            batch_dict["infer"] = True

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

        # create param. groups by `decay/no_decay'
        decay = set()
        no_decay = set()
        for k, v in self.model.named_parameters():
            k = re.sub("^module\.", "", k) if ddp else k
            if f"{k}" not in tunable_params:
                continue
            if v.ndim < 2 or "_embed" in k: # FIXME hack
                no_decay.add(k)
            elif v.ndim > 1:
                decay.add(k)

        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"{inter_params} in both decay / no_decay sets"
        assert set(union_params) == set(tunable_params.keys()), f"not all the tunable are assigned to decay / no_decay sets"

        param_groups = [
            {"params": [tunable_params[k] for k in sorted(list(decay))], "weight_decay": self.cfg.optimizer.weight_decay},
            {"params": [tunable_params[k] for k in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        if self.cfg.optimizer.use_lars:
            self.optimizer = LARS(
                param_groups,
                lr=0.,
                weight_decay=self.cfg.optimizer.weight_decay,
                weight_decay_filter=exclude_bias_or_norm,
                lars_adaptation_filter=exclude_bias_or_norm,
            )
        else:
            ocfg = self.cfg.optimizer.optimizer
            scfg = self.cfg.optimizer.scheduler
            self.optimizer = getattr(torch.optim, ocfg[0])(param_groups, **ocfg[1])
            self.scheduler = None if len(scfg) < 2 else getattr(
                torch.optim.lr_scheduler, scfg[0]
            )(self.optimizer, **scfg[1])
        if not self.cfg.verbose or not verbose:
            return
        self.echo(f"Gradienting The Following Parameters:")
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                if k in decay:
                    dm = "dt"
                else:
                    dm = "#"
                self.echo(f"{k} {v.size()} {dm}")
        self.echo(f"\n{self.model}")
