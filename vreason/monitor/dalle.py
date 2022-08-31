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

from .raven_solver import Monitor as Meta
from ..model import build_main_model
from ..data import build_clevr_image_text_data

from ..util import numel, shorten_name, save_image_local, is_main_process, get_rank
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
        if self.cfg.optimizer.max_gnorm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.optimizer.max_gnorm
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def build_data(self):
        self.dataloader, self.evalloader, self.testloader, \
        self.encoder_vocab, self.decoder_vocab, self.cate_vocab = \
        eval(f"build_{self.cfg.data.name.lower()}_image_text_data")(
            self.cfg.data, not self.cfg.eval, self.echo, ddp_mode=(self.cfg.mode == "ddp")
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
        self.model_pointer.reset()
        all_time = defaultdict(list)
        self.timeit(all_time)        
        device_ids = [i for i in range(self.cfg.num_gpus)]
        nchunk = dist.get_world_size() if torch.distributed.is_initialized() else 1  
        warmup_step_rate = (-1 if self.scheduler is None else
            max(self.cfg.optimizer.warmup_steps // self.cfg.optimizer.warmup_times, 1)
        )
        visual_peep_rate = max(len(self.dataloader) // self.cfg.running.v_peep_time, 1)
        def do_batch(batch, step):
            epoch_step = step #% len(self.dataloader)

            #self.show_batch(batch, meta)

            batch_dict = self.make_batch(batch) 
            batch_dict["device_ids"] = device_ids

            v_peep_topk = self.cfg.running.v_peep_topk if step % visual_peep_rate == 0 else -1
            batch_dict["v_peep_topk"] = v_peep_topk # save some sampled images

            self.optim_step += 1 
            bsize = batch_dict["image"].shape[0]
            force_eval, warmup = self.pre_step(step, warmup_step_rate)

            self.timeit(all_time, key="data")

            with torch.cuda.amp.autocast():
                loss, (ntoken, ret_dict) = self.model(**batch_dict)
            self.scaler.scale(loss).backward()
            self.step()


            self.timeit(all_time, key="model")

            self.save_images(
                ret_dict, outdir="train", prefix=f"e{iepoch:04d}-s{epoch_step:07d}-", **batch_dict
            ) # let us log some images

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

        if self.scheduler is not None and not self.cfg.optimizer.use_lars and \
            not self.cfg.optimizer.batch_sch:
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

        self.model_pointer.reset()
        losses, istep, nsample, nchunk, nbatch = 0, 0, 0, 1, len(dataloader)
        device_ids = [i for i in range(self.cfg.num_gpus)]
        if isinstance(self.model, DistributedDataParallel):
            dataloader.sampler.set_epoch(iepoch)
            nchunk = self.cfg.num_gpus

        visual_peep_rate = int(max(len(dataloader) // self.cfg.running.v_peep_time, 1))
        odir = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S") + f"-{self.cfg.alias_odir}" 
        odir = odir if self.cfg.running.infer_mode else "val"

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

            v_peep_topk = self.cfg.running.v_peep_topk if ibatch % visual_peep_rate == 0 else -1
            batch_dict["v_peep_topk"] = v_peep_topk # save some sampled images

            bsize = batch_dict["image"].shape[0]

            with torch.cuda.amp.autocast():
                loss_mean, (ntoken, ret_dict) = self.model(**batch_dict)
            loss = loss_mean * 1 
            
            self.save_images(
                ret_dict, outdir=odir, prefix=f"e{iepoch:04d}-s{epoch_step:07d}-", **batch_dict
            ) # let us log some images

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

        self.save_metric = self.model_pointer.eval_metric() #-losses / epoch_step
        self.echo(f"# sample {nsample}; metric {self.save_metric:.3f}; {nsample / (time.time() - start_time):.2f} samples/s")
        stats = self.model_pointer.stats()
        if stats != "": # could be empty
            self.echo(f"EVAL STATS: {self.model_pointer.stats()}")
        return self.model_pointer.report()

    def save_images(
        self, sampled, text=None, image=None, text_mask=None, vfile=None, outdir="", prefix="", **kwargs
    ):
        if sampled is None or len(sampled) == 0 or not is_main_process():
            return # FIXME only save on the master process
        labeled = [(k, v) for k, v in sampled.items() if k.startswith("_sample")]
        if len(labeled) <= 0:
            return
        nsample = sampled["nsample"]
        image = image[:nsample]
        vfile = vfile[:nsample]
        # text prompts
        prompts = ["" for _ in range(image.shape[0])]
        if text is not None:
            text = text[:nsample]
            text_mask = text_mask[:nsample]
            lengths = ((~text_mask).sum(-1) - 1).cpu().tolist()
            prompts = text[:, 1:].cpu().tolist()
            prompts = self.encoder_vocab(prompts)
            prompts = [" ".join(txt[:l]) for l, txt in zip(lengths, prompts)]
        # gold images
        images = (
            (image.detach().cpu().clamp(-1, 1) + 1.) / 2 * 255.
        ).permute(0, 2, 3, 1).numpy().astype(np.uint8)
        # sampled images
        all_labels = ["Gold"] + [k for k, v in labeled]
        all_images = [images] + [v for k, v in labeled]
        all_images = np.concatenate(
            [x[..., None] for x in all_images], axis=-1
        )
        # TODO try ... except ...
        fnames = [prefix + fname.rsplit("/", 1)[1].rsplit(".", 1)[0] for fname in vfile]
        root = f"{self.cfg.alias_root}/{self.cfg.alias_name}/{outdir}"
        save_image_local(root, fnames, prompts, all_images, labels=all_labels)
