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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..util import numel
from ..module import LARS, exclude_bias_or_norm, adjust_learning_rate

class Monitor(object):
    def __init__(self):
        super(Monitor, self).__init__()
        # save the k-best checkpoints
        self.save_metric = None
        self.kbest_cache = None 

    def build_data(self):
        pass

    def learn(self):
        if not self.model.training:
            self.echo("Evaluating started...")
            with torch.no_grad():
                report = self.evaluate(self.dataloader, samples=self.cfg.data.eval_samples)
                self.echo(f"{report}")
                return None 
        self.echo("Training started...")
        self.last_time = 0.
        self.epoch_loss = 0
        self.epoch_step = 0
        self.total_loss = 0
        self.total_step = 0
        self.total_inst = 0
        self.optim_step = 0
        self.start_time = time.time()
        self.scaler = torch.cuda.amp.GradScaler()
        #self.save() 
        if self.cfg.data.data_seed is not None: # reset data randomness
            self.echo(f"Random seed ({self.cfg.data.data_seed}) for data sampling.")
            seed_all_rng(self.cfg.data.data_seed)
        for iepoch in range(self.cfg.optimizer.epochs):
            if isinstance(self.model, DistributedDataParallel):
                self.dataloader.sampler.set_epoch(iepoch)
            if iepoch >= 1:
                pass #break
            self.num_sents = self.num_words = 0.
            self.all_stats = [[0., 0., 0.]]
            self.epoch(iepoch)

    def make_batch(self, union):
        pass

    def timeit(self, time_dict, key=None, show=False):
        if self.cfg.rank != 0:
            return 
        if show: # print
            report = ""
            for k, v in time_dict.items():
                report += f"{k} {np.mean(v):.2f} "
            self.echo(f"Time (s): {report.strip()}; # step {self.total_step} # sample {self.total_inst}")
            return
        if key is None: # initialize
            self.last_time = time.time()
        else: # update
            this_time = time.time()
            time_dict[key].append(this_time - self.last_time)
            self.last_time = this_time

    def pre_step(self, step, warmup_step_rate, inc=0):
        if self.cfg.optimizer.use_lars:
            adjust_learning_rate(self.cfg.optimizer, self.optimizer, self.dataloader, step)

        inc = 1 
        force_eval = False # recommended by SGDR
        warmup = not self.cfg.optimizer.use_lars and self.cfg.optimizer.warmup and \
            (self.total_step + inc) <= self.cfg.optimizer.warmup_steps
        # it is important to always warm up lr at the first step otherwise
        # the optimizer will use the default / initial lr
        if warmup and ((self.total_step + inc) % warmup_step_rate == 0 or self.total_step == 0):
            ratio = ((self.total_step + inc) / self.cfg.optimizer.warmup_steps) # * self.cfg.optimizer.lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = ratio * param_group["initial_lr"]
            lrs = [param_group['lr'] for param_group in self.optimizer.param_groups]
            force_eval = lrs == self.scheduler.base_lrs
            lrs = [f"{lr:.2e}" for lr in lrs]
            if (self.total_step + inc) <= 50 or (
                self.cfg.optimizer.warmup_steps > 50 and (self.total_step + inc) % self.cfg.running.peep_rate == 0
            ): 
                self.echo(f"warmup lr: {' '.join(lrs)} @ {self.total_step}")

        return force_eval, warmup 

    def post_step(
        self, iepoch, epoch_step, force_eval, warmup, nchunk
    ):
        if not self.cfg.optimizer.use_lars and self.cfg.optimizer.batch_sch and \
            not warmup and self.scheduler is not None:
            #old_lrs = " ".join([f"{x:.2e}" for x in self.scheduler.get_last_lr()])
            self.scheduler.step() # after all warmup is completed
            if isinstance(self.scheduler, (CosineAnnealingWarmRestarts,)):
                force_eval = self.scheduler.get_last_lr() == self.scheduler.base_lrs
            #self.echo(f"do step lr {old_lrs}")

        if force_eval or (self.cfg.rank == 0 and epoch_step % self.cfg.running.peep_rate == 0):
            msg = self.model.stats()
            if msg != "":
                #self.echo(msg)
                msg = f"{msg} "
            # example output if there is any
            #
            # overall stats 
            lr_w = self.optimizer.param_groups[0]['lr']
            lr_b = self.optimizer.param_groups[1]['lr']
            self.echo(
                f"epoch {iepoch:>4} step {epoch_step} / {self.total_step}\t" + 
                f"lr_w {lr_w:.2e} lr_b {lr_b:.2e} loss {self.epoch_loss / self.epoch_step:.3f} " + 
                f"{msg}{self.total_inst / (time.time() - self.start_time):.2f} samples/s" 
            )
            self.epoch_loss = self.epoch_step = 0

        ppl_criteria = -1
        if force_eval or self.total_step % self.cfg.running.save_rate == 0 or (
                self.cfg.running.save_epoch and epoch_step % len(self.dataloader) == 0
            ): # distributed eval
            report = ""
            if self.evalloader is not None:
                self.model.train(False)
                with torch.no_grad():
                    report = self.infer(
                        self.evalloader, samples=self.cfg.data.eval_samples, iepoch=iepoch
                    )
                self.model.train(True)
            if report != "":
                self.echo(f"{report}")
            if (self.cfg.rank == 0 and not self.cfg.running.skip_save) or (
                    self.cfg.running.save_last and self.cfg.running.epochs == iepoch + 1
                ):
                self.save()

        # global across epochs 
        if self.optim_step % self.cfg.running.optim_rate == 0: 
            self.model.zero_grad()
        # used for initialization search
        return ppl_criteria 

    def step(self):
        if self.cfg.running.optim_rate > 1:
            self.model.reduce_grad(cfg.running.optim_rate, sync=False)
        if self.cfg.optimizer.max_gnorm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.optimizer.max_gnorm
            )
        self.optimizer.step()

    def epoch(self, iepoch):
        pass

    def infer(self, dataloader, samples=float("inf"), iepoch=0):
        pass
        
    def evaluate(self, dataloader, samples=float("inf"), iepoch=0):
        pass

    def save(self, metric=-float("inf")):
        def update_kbest_cache(kbest_cache, metric, ckpt):
            save = False
            for split, (old_metric, _) in enumerate(kbest_cache):
                if metric > old_metric:
                    save = True
                    break
            if not save:
                return kbest_cache, (-float("inf"), None), False 
            a = kbest_cache[:split]
            c = kbest_cache[split:-1]
            b = [(metric, ckpt)]
            removed = kbest_cache[-1]
            return a + b + c, removed, True

        fsave = f"{self.cfg.alias_root}/{self.cfg.alias_name}/{self.total_step:08d}.pth"

        if self.kbest_cache is not None:
            metric = self.save_metric or metric 
            self.kbest_cache, removed, save = update_kbest_cache(self.kbest_cache, metric, fsave) 
            if not save:
                fsave = f"{self.cfg.alias_root}/{self.cfg.alias_name}/last.pth"
                pass #return None # exit # save the last anyways 
            if removed[-1] is not None:
                try:
                    os.remove(removed[-1])
                    self.echo(f"Del {removed[-1]}")
                except OSError as e:
                    self.echo(f"Error: {e.filename} - {e.strerror}.")

        self.echo(f"Saving the checkpoint to {fsave}")
        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        checkpoint = {
            "cfg": self.cfg, "model": model.collect_state_dict(), "vocab": (self.encoder_vocab, self.decoder_vocab)
        }
        torch.save(checkpoint, fsave)

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
        param_groups = [
            {"params": [p for p in self.params if p.ndim > 1]},
            {"params": [p for p in self.params if p.ndim < 2]},
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
                self.echo(f"{k} {v.size()}")
        self.echo(f"\n{self.model}")

