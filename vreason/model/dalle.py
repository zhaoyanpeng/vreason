import math
import os, re
import torch
from torch import nn

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import data_parallel

from . import from_pretrained_vqgan, load_checkpoint, MetaSolver
from ..util import Stats, enable_print, disable_print
from ..module import build_encoder_head, build_decoder_head, build_loss_head


class Dalle(MetaSolver):
    def __init__(self, cfg, echo):
        super().__init__(cfg, echo)

    def tokenize_images(self, v):
        return self.vq.encode(v)

    def forward(self, text=None, image=None, text_mask=None, analyze=False, device_ids=[0], **kwargs):
        if len(device_ids) > 1:
            return self.forward_dp(
                text=text, image=image, text_mask=text_mask, analyze=analyze, device_ids=device_ids, **kwargs
            )
        # a shared embedder (b/w encoder and decoder) to mimic GPT 
        # with bi-directional memory or an encoder-only GPT model.
        v_seq = self.tokenize_images(image) # discretize images 
        t_emb, v_emb, emb_extra = self.embedder_head(text, v_seq=v_seq)
        
        memory, _, _, enc_extra = self.encoder_head(
            t_emb, v_emb, t_seq=text, v_seq=v_seq,
            self_key_padding_mask=text_mask, **kwargs
        )
        
        logits, targets, _, dec_extra = self.decoder_head(
            t_emb, v_emb, t_seq=text, v_seq=v_seq,
            memo_key_padding_mask=text_mask,  memory=memory, **kwargs
        )
        
        _, outs = self.loss_head(logits, targets)
        loss = self.set_stats(dec_extra, outs[-1])

        if analyze:
            self.analyze(
                t_emb=t_emb, v_emb=v_emb, t_seq=text, v_seq=v_seq,
                logits=logits, targets=targets, memory=memory,
            )
        return loss, outs 

    def forward_dp(self, text=None, image=None, text_mask=None, analyze=False, device_ids=[0], **kwargs):
        v_seq = data_parallel(
            self.vq, image, device_ids=device_ids
        )

        t_emb, v_emb, emb_extra = self.embedder_head(text, v_seq=v_seq)

        kwargs.update({
            "t_seq": text, "v_seq": v_seq,
            "self_key_padding_mask": text_mask,
            "memo_key_padding_mask": text_mask
        })
        memory, _, _, enc_extra = data_parallel(
            self.encoder_head, (t_emb, v_emb), device_ids=device_ids, module_kwargs=kwargs
        )
        
        kwargs.update({"memory": memory})
        logits, targets, _, dec_extra = data_parallel(
            self.decoder_head, (t_emb, v_emb), device_ids=device_ids, module_kwargs=kwargs
        )
        
        _, outs = self.loss_head(logits, targets)
        loss = self.set_stats(dec_extra, outs[-1])
        
        if analyze:
            self.analyze(
                t_emb=t_emb, v_emb=v_emb, t_seq=text, v_seq=v_seq,
                logits=logits, targets=targets, memory=memory,
            )
        return loss, outs 

    def stats(self): 
        meter = self.meter_train if self.training else self.meter_infer
        stats = meter.stats
        
        if not self.embedder_head.is_bart:
            nstep = stats["nstep"]
            alpha = 1 / nstep if nstep > 0 else 0
            t_loss = stats["t_main_loss"] * alpha 
            v_loss = stats["v_main_loss"] * alpha 
            info = f"t_loss {t_loss:.5f} v_loss {v_loss:.5f} "

            t_ntoken = stats["t_ntoken"]
            alpha = 1 / t_ntoken * 100 if t_ntoken > 0 else 0
            t_acc = stats["t_ntrue"] * alpha

            v_ntoken = stats["v_ntoken"]
            alpha = 1 / v_ntoken * 100 if v_ntoken > 0 else 0
            v_acc = stats["v_ntrue"] * alpha

            ntoken = stats["ntoken"]
            alpha = 1 / ntoken if ntoken > 0 else 0
            acc = stats["ntrue"] * alpha

            info += f"t_acc {t_acc:.3f} v_acc {v_acc:.3f} acc {acc:.3f}"
        else:
            nstep = stats["nstep"]
            alpha = 1 / nstep if nstep > 0 else 0
            loss = stats["main_loss"] * alpha 

            ntoken = stats["ntoken"]
            alpha = 1 / ntoken * 100 if ntoken > 0 else 0
            acc = stats["ntrue"] * alpha
            info = f"loss {loss:.5f} acc {acc:.3f}"

        nsample = stats["nsample"]
        info = f"{info} ({nsample})"
        return info

    def register_vq(self, mcfg):
        vq = build_encoder_head(mcfg.vq, None)
        return vq.emb_weight, vq

    def build(self, encoder_vocab, decoder_vocab=None, **kwargs):
        tunable_params = dict()
        mcfg = self.cfg.model 
        
        vq_embed, vq = self.register_vq(mcfg)
        vq_vocab_size, emb_size = vq_embed.shape[:2]

        if decoder_vocab is None:
            decoder_vocab = [f"{i}" for i in range(vq_vocab_size)]

        if self.cfg.eval:
            self.vq = vq # might need the model to decode codes
            local_cfg, head_sd = load_checkpoint(self.cfg, self.echo)

            self.embedder_head = build_encoder_head(
                mcfg.embedder, encoder_vocab, vis_token_vocab=decoder_vocab
            )
            self.embedder_head.load_state_dict(head_sd["embedder_head"])

            self.encoder_head = build_encoder_head(mcfg.encoder, encoder_vocab, vis_token_vocab=decoder_vocab)
            self.encoder_head.load_state_dict(head_sd["encoder_head"])

            self.decoder_head = build_decoder_head(mcfg.decoder, encoder_vocab, vis_token_vocab=decoder_vocab)
            self.decoder_head.load_state_dict(head_sd["decoder_head"])

            self.loss_head = build_loss_head(mcfg.loss, encoder_vocab, vis_token_vocab=decoder_vocab)
            self.loss_head.load_state_dict(head_sd["loss_head"])
        else:
            self.vq = vq # only need the weights of VQ 
            self.embedder_head = build_encoder_head(
                mcfg.embedder, encoder_vocab, vis_token_vocab=decoder_vocab
            )
            self.encoder_head = build_encoder_head(mcfg.encoder, encoder_vocab, vis_token_vocab=decoder_vocab)
            self.decoder_head = build_decoder_head(mcfg.decoder, encoder_vocab, vis_token_vocab=decoder_vocab)
            self.loss_head = build_loss_head(mcfg.loss, encoder_vocab, vis_token_vocab=decoder_vocab)
            tunable_params = {
                f"embedder_head.{k}": v for k, v in self.embedder_head.named_parameters()
            } 
            tunable_params.update({
                f"encoder_head.{k}": v for k, v in self.encoder_head.named_parameters()
            })
            tunable_params.update({
                f"decoder_head.{k}": v for k, v in self.decoder_head.named_parameters()
            })
            tunable_params.update({
                f"loss_head.{k}": v for k, v in self.loss_head.named_parameters()
            })
        self.cuda(self.cfg.rank)
        return tunable_params

