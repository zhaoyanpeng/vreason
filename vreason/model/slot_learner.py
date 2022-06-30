import math
import os, re
import torch
from torch import nn

import torch.distributed as dist
import torch.nn.functional as F

from . import from_pretrained_vqgan, load_checkpoint
from ..util import Stats, AGI, enable_print, disable_print
from ..module import build_encoder_head, build_decoder_head, build_loss_head

class SlotLearner(nn.Module):
    def __init__(self, cfg, echo):
        super().__init__()
        self.cfg = cfg
        self.echo = echo
        self.agi = AGI() # adjusted rand index 
        self.meter_train = Stats()
        self.meter_infer = Stats()

    def forward(self, image, mask=None, nobj=None, analyze=False, vfile=None, **kwargs):
        slots, attns, *_ = self.encoder_head(image) 
        x_out, masks, x_all, *_ = self.decoder_head(slots)
        loss, outs = self.loss_head(image, x_out)
        self.set_stats(outs[1], mask, masks, nobj) 
        if analyze:
            self.analyze(
                true_image=image, pred_image=x_out, true_mask=mask, pred_mask=masks,
                image_file=vfile, slot_image=x_all
            )
        return loss, outs 

    def analyze(self, **kwargs):
        self.last_batch = {k: v for k, v in kwargs.items()}

    def collect_state_dict(self):
        return { 
            "encoder_head": self.encoder_head.state_dict(), 
            "decoder_head": self.decoder_head.state_dict(), 
            "loss_head": self.loss_head.state_dict(),
        } 

    def set_stats(self, loss_dict, true_mask, pred_mask, nobj):
        meter = self.meter_train if self.training else self.meter_infer
        meter(**loss_dict)
        if true_mask.shape[-1] != 0:
            ari = self.agi(true_mask, pred_mask)
            meter(**{"ari": ari.sum().item()})

    def stats(self): 
        meter = self.meter_train if self.training else self.meter_infer
        stats = meter.stats
        nsample = max(stats["nsample"], 1)
        loss = stats["main_loss"] / nsample
        info = [f"main loss {loss:.5f}"]
        if "ari" in stats:
            ari = stats["ari"] / nsample
            info.append(f"ari {ari:.5f}")
        info.append(f"({nsample})")
        return " ".join(info)

    def reset(self):
        meter = self.meter_train if self.training else self.meter_infer
        meter.reset()
    
    def reduce_grad(optim_rate, sync=False):
        raise NotImplementedError("Gradient Reduce")

    def report(self, gold_file=None):
        return ""

    def init_weights(self) -> None:
        pass

    def build(self, encoder_vocab, decoder_vocab, **kwargs):
        tunable_params = dict()
        mcfg = self.cfg.model 
        
        if self.cfg.eval:
            local_cfg, head_sd = load_checkpoint(self.cfg, self.echo)

            self.encoder_head = build_encoder_head(mcfg.encoder, encoder_vocab)
            self.encoder_head.load_state_dict(head_sd["encoder_head"])

            self.decoder_head = build_decoder_head(mcfg.decoder, decoder_vocab)
            self.decoder_head.load_state_dict(head_sd["decoder_head"])

            self.loss_head = build_loss_head(mcfg.loss, decoder_vocab)
            self.loss_head.load_state_dict(head_sd["loss_head"])
        else:
            self.encoder_head = build_encoder_head(mcfg.encoder, encoder_vocab)
            self.decoder_head = build_decoder_head(mcfg.decoder, decoder_vocab)
            self.loss_head = build_loss_head(mcfg.loss, decoder_vocab)
            tunable_params = {
                f"encoder_head.{k}": v for k, v in self.encoder_head.named_parameters()
            }
            tunable_params.update({
                f"decoder_head.{k}": v for k, v in self.decoder_head.named_parameters()
            })
            tunable_params.update({
                f"loss_head.{k}": v for k, v in self.loss_head.named_parameters()
            })
        self.cuda(self.cfg.rank)
        return tunable_params
