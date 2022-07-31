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
from ..data import build_clevr_image_text_data

from ..util import numel, shorten_name, save_image_local, ExpDecayLR, SlotattnLR
from ..module import LARS, exclude_bias_or_norm, adjust_learning_rate

class Monitor(Meta):
    def __init__(self, cfg, echo, device):
        super(Monitor, self).__init__(cfg, echo, device)
        # save the k-best checkpoints
        topk = cfg.running.topk_best
        self.kbest_cache = [(-float("inf"), None)] * topk if topk > 1 else None 

    def show_batch(self, batch):
        pass

    def step(self, loss=None, **kwargs):
        pass

    def build_data(self):
        self.dataloader, self.evalloader, self.testloader, \
        self.encoder_vocab, self.decoder_vocab, self.cate_vocab = \
        eval(f"build_{self.cfg.data.name.lower()}_image_text_data")(
            self.cfg.data, not self.cfg.eval, self.echo
        )

    def make_batch(self, sample):
        image = sample["image"]
        image = torch.from_numpy(image).to(self.device)
        image = image.permute(0, 3, 1, 2)

        text = sample["text"]
        text = torch.from_numpy(text).to(self.device)
        
        text_mask = text == self.encoder_vocab.PAD_IDX

        return {"image": image, "text": text, "text_mask": text_mask, "vfile": sample["vfile"]}

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
            batch_dict["device_ids"] = device_ids

            self.optim_step += 1 
            bsize = batch_dict["image"].shape[0]
            force_eval, warmup = self.pre_step(step, warmup_step_rate)

            self.timeit(all_time, key="data")
            

            with torch.cuda.amp.autocast():
                loss, _ = self.model(**batch_dict)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()


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

    def save_images(
        self, sampled_images, text=None, image=None, text_mask=None, vfile=None, **kwargs
    ):
        if sampled_images is None:
            return
        # text prompts
        prompts = ["" for _ in range(sampled_images.shape[0])]
        if text is not None:
            lengths = ((~text_mask).sum(-1) - 1).cpu().tolist()
            prompts = text[:, 1:].cpu().tolist()
            prompts = self.encoder_vocab(prompts)
            prompts = [" ".join(txt[:l]) for l, txt in zip(lengths, prompts)]
        # sampled images
        images = (
            (image.detach().cpu().clamp(-1, 1) + 1.) / 2 * 255.
        ).permute(0, 2, 3, 1).numpy().astype(np.uint8)
        all_images = np.concatenate(
            (images[..., None], sampled_images), axis=-1
        )
        # TODO try ... except ...
        fnames = [fname.rsplit("/", 1)[1].rsplit(".", 1)[0] for fname in vfile]
        root = f"{self.cfg.alias_root}/{self.cfg.alias_name}/sample"
        save_image_local(root, fnames, prompts, all_images)
        
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
            batch_dict["infer"] = self.cfg.running.infer_mode 
            batch_dict["debug"] = False 
            batch_dict["analyze"] = True
            batch_dict["nsampling"] = self.cfg.running.nsampling 
            batch_dict["device_ids"] = device_ids

            bsize = batch_dict["image"].shape[0]

            with torch.cuda.amp.autocast():
                loss_mean, (ntoken, ret_dict) = self.model(**batch_dict)
            loss = loss_mean * 1 
            
            self.save_images(ret_dict.pop("image", None), **batch_dict) # let us log some images

            epoch_step += 1
            total_word += ntoken 

            nsample += bsize * nchunk
            losses += loss or 0.
            if self.cfg.rank == 0 and (ibatch + 1) % peep_rate == 0:
                self.echo(
                    f"step {istep} / {ibatch}\t" + #gnorm {grad_norm():.2f} " +
                    f"loss {losses / epoch_step:.8f} " # (istep + 0):.8f} " # (ibatch + 1):.8f} " +
                    f"{nsample / (time.time() - start_time):.2f} samples/s"
                )

        model = self.model.module if isinstance(self.model, DistributedDataParallel) else self.model
        self.save_metric = model.eval_metric() #-losses / epoch_step
        self.echo(f"# sample {nsample}; metric {self.save_metric:.3f}; {nsample / (time.time() - start_time):.2f} samples/s")
        stats = model.stats()
        if stats != "": # could be empty
            self.echo(f"EVAL STATS: {model.stats()}")
        return model.report()
