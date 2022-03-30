import math
import os, re
import torch
from torch import nn

import torch.distributed as dist
import torch.nn.functional as F

from . import from_pretrained_vqgan, load_checkpoint
from ..util import Stats, enable_print, disable_print
from ..module import build_encoder_head, build_decoder_head, build_loss_head

class MetaSolver(nn.Module):
    def __init__(self, cfg, echo):
        super().__init__()
        self.cfg = cfg
        self.echo = echo
        self.meter_train = Stats()
        self.meter_infer = Stats()
        
    def __repr__(self):
        def _addindent(s_, numSpaces):
            s = s_.split('\n')
            # don't do anything for single-line stuff
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(numSpaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            if key == "vqgan": # filter out VAGAN module
                continue
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def analyze(self, **kwargs):
        """ This records stats from the last batch.
        """
        self.last_batch = {k: v for k, v in kwargs.items()}
        return None

    def set_stats(self, dec_extra, loss_dict):
        meter = self.meter_train if self.training else self.meter_infer
        meter(**loss_dict)
        loss = sum([v for k, v in loss_dict.items() if k.endswith("_loss")])
        loss = loss_dict["main_loss"]
        return loss

    def stats(self): 
        meter = self.meter_train if self.training else self.meter_infer
        stats = meter.stats
        nsample = stats["nsample"]
        success = stats["ntrue"] / nsample if nsample > 0 else 0.
        info = f"acc {success * 100:7.3f} ({nsample})"
        return info

    def reset(self):
        meter = self.meter_train if self.training else self.meter_infer
        meter.reset()

    def report(self, gold_file=None):
        if (not dist.is_initialized() or dist.get_rank() == 0) and self.loss_head is not None:
            return self.loss_head.report(gold_file=gold_file) 
        else:
            return ""

    def train(self, mode):
        self.training = mode # so we could filter out some modules
        for module in [self.embedder_head, self.encoder_head, self.decoder_head, self.loss_head]:
            module.train(mode)
        return self

    def collect_state_dict(self):
        return { 
            "embedder_head": self.embedder_head.state_dict(), 
            "encoder_head": self.encoder_head.state_dict(), 
            "decoder_head": self.decoder_head.state_dict(), 
            "loss_head": self.loss_head.state_dict(),
        } 

    def reduce_grad(optim_rate, sync=False):
        raise NotImplementedError("Gradient Reduce")

    def build(self, encoder_vocab, decoder_vocab, **kwargs):
        tunable_params = dict()
        mcfg = self.cfg.model 
        
        vq_embed, vqgan = self.register_vqgan(mcfg)
        vq_vocab_size, emb_size = vq_embed.shape

        if self.cfg.eval:
            self.vqgan = vqgan # might need the model to decode codes
            local_cfg, head_sd = load_checkpoint(self.cfg, self.echo)

            self.embedder_head = build_encoder_head(
                mcfg.embedder, encoder_vocab, emb_size=emb_size, fixed_weight=vq_embed
            )
            self.embedder_head.load_state_dict(head_sd["embedder_head"])

            self.encoder_head = build_encoder_head(mcfg.encoder, encoder_vocab)
            self.encoder_head.load_state_dict(head_sd["encoder_head"])

            self.decoder_head = build_decoder_head(mcfg.decoder, decoder_vocab)
            self.decoder_head.load_state_dict(head_sd["decoder_head"])

            self.loss_head = build_loss_head(mcfg.loss, decoder_vocab)
            self.loss_head.load_state_dict(head_sd["loss_head"])
        else:
            self.vqgan = None # only need the weights of VQ 
            self.embedder_head = build_encoder_head(
                mcfg.embedder, encoder_vocab, emb_size=emb_size, fixed_weight=vq_embed
            )
            self.encoder_head = build_encoder_head(mcfg.encoder, encoder_vocab)
            self.decoder_head = build_decoder_head(mcfg.decoder, decoder_vocab)
            self.loss_head = build_loss_head(mcfg.loss, decoder_vocab)
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


class RavenSolver(MetaSolver):
    """ Need a VQGAN to encode discrete visual tokens.
    """
    def __init__(self, cfg, echo):
        super().__init__(cfg, echo)

    def forward(self, sequence, negative, analyze=False, **kwargs):
        x = seq_emb = self.encode_code(sequence)
        # a shared embedder (b/w encoder and decoder) to mimic GPT with bi-directional memory
        src_emb, tgt_emb, tgt_seq, emb_extra = self.embedder_head(seq_emb, sequence)
        memory, _, _, enc_extra = self.encoder_head(src_emb)
        
        token_embed = None
        if not self.training: # hack for inference
            kwargs.update({
                "infer_type": self.cfg.model.decoder.infer_mode,
                "beg_dec_len": self.cfg.model.decoder.beg_dec_len
            }) # hack
            kwargs.update(emb_extra)
            n_vq_embed = self.vq_embed.shape[0]
            token_embed = self.embedder_head.token_embed.weight # [vq_embed; special] 
            token_embed = torch.cat([self.vq_embed, token_embed[n_vq_embed:]], dim=0)

        logits, targets, _, dec_extra = self.decoder_head(
            tgt_emb, x_seq=tgt_seq, memory=memory, negative_seq=negative,
            token_embed=token_embed, **kwargs
        )

        _, outs = self.loss_head(logits, targets, negative=negative)
        loss = self.set_stats(dec_extra, outs[-1])
        
        if analyze:
            self.analyze(
                logits=logits, targets=targets, memory=memory, token_embed=token_embed
            )
        return loss, outs 

    def encode_code(self, sequence):
        shape = sequence.shape + (-1,)
        sequence = sequence.view(-1)
        if self.vq_embed is None:# method A 
            seq_emb = self.vqgan.quantize.get_codebook_entry(sequence, None)
            seq_emb = seq_emb.view(shape)
        else: # method B
            seq_emb_new = self.vq_embed[sequence]
            seq_emb_new = seq_emb_new.view(shape)
            seq_emb = seq_emb_new
        #print((seq_emb == seq_emb_new).all())
        return seq_emb

    def register_vqgan(self, mcfg):
        vqgan = from_pretrained_vqgan(mcfg.vqgan)
        quantizer = vqgan.quantize
        vocab_size, emb_size = quantizer.n_e, quantizer.e_dim
        self.register_buffer("vq_embed", vqgan.quantize.embedding.weight)
        return self.vq_embed, vqgan 


class NeoRavenSolver(MetaSolver):
    def __init__(self, cfg, echo):
        super().__init__(cfg, echo)

    def register_vqgan(self, mcfg):
        vqgan = from_pretrained_vqgan(mcfg.vqgan)
        return vqgan.quantize.embedding.weight, vqgan

    def forward(self, sequence, negative, analyze=False, **kwargs):
        # a shared embedder (b/w encoder and decoder) to mimic GPT 
        # with bi-directional memory or an encoder-only GPT model.
        (ctx_emb, ctx_seq), (tgt_emb, tgt_seq), (neg_emb, neg_seq), emb_extra = \
            self.embedder_head(sequence, negative)
        memory, _, _, enc_extra = self.encoder_head(ctx_emb)
        
        if not self.training:
            kwargs.update(emb_extra)

        logits, targets, _, dec_extra = self.decoder_head(
            tgt_emb, x_seq=tgt_seq, memory=memory, neg_emb=neg_emb, neg_seq=neg_seq, **kwargs
        )

        _, outs = self.loss_head(logits, targets, negative=negative)
        loss = self.set_stats(dec_extra, outs[-1])
        
        if analyze:
            self.analyze(
                logits=logits, targets=targets, memory=memory, token_embed=token_embed
            )
        return loss, outs 
