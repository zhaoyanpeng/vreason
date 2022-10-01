import math
import copy
import os, re
import torch
import numpy as np
from torch import autocast, nn, Tensor

import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import data_parallel

from . import from_pretrained_vqgan, load_checkpoint, MetaSolver
from ..util import Stats, enable_print, disable_print, top_k_top_p_filtering, get_world_size, get_rank
from ..module import build_encoder_head, build_decoder_head, build_loss_head


class Dalle(MetaSolver):
    """ A shared embedder (b/w encoder and decoder) to mimic GPT 
        with bi-directional memory or an encoder-only GPT model.
    """
    def __init__(self, cfg, echo):
        super().__init__(cfg, echo)

    def _debug_cache(self, old_cache, new_cache, istep):
        if old_cache is not None and self.embedder_head.is_bart:
            v_old, *_ = [x for x in old_cache]
            v_new, *_ = [x for x in new_cache]

            eq1 = (v_old == v_new[:, :-1]).all().cpu().item()
            df1 = (v_old -  v_new[:, :-1]).abs().sum().cpu().item()

            eq11 = torch.allclose(v_old, v_new[:, :-1], atol=1e-6)
            df11 = (v_old - v_new[:, :-1]).abs().max().cpu().item()
            print(f"step {istep:3}", list(v_old.shape), list(v_new.shape), eq1, df1, eq11, df11)

        if old_cache is not None and not self.embedder_head.is_bart:
            t_old, v_old = [x for x in old_cache]
            t_new, v_new = [x for x in new_cache]

            eq1 = (v_old == v_new[:, :-1]).all().cpu().item()
            df1 = (v_old -  v_new[:, :-1]).abs().sum().cpu().item()

            eq2 = (t_old == t_new).all().cpu().item()
            df2 = (t_old -  t_new).abs().sum().cpu().item()

            eq11 = torch.allclose(v_old, v_new[:, :-1], atol=1e-6)
            df11 = (v_old - v_new[:, :-1]).abs().max().cpu().item()
            print(f"step {istep:3}", list(v_old.shape), list(v_new.shape), eq2, df2, eq1, df1, eq11, df11)
        return new_cache

    def _seq2seq(
        self, t, v, t_seq=None, v_seq=None, text_mask=None, 
        enc_cache=None, dec_cache=None, use_cache=False, **kwargs,
    ):
        memory, _, enc_cache, enc_extra = self.encoder_head(
            t, v, t_seq=t_seq, v_seq=v_seq, cache=enc_cache, use_cache=use_cache,
            self_key_padding_mask=text_mask, **kwargs
        )
        logits, targets, dec_cache, dec_extra = self.decoder_head(
            t, v, t_seq=t_seq, v_seq=v_seq, cache=dec_cache, use_cache=use_cache,
            memo_key_padding_mask=text_mask, memory=memory, **kwargs
        )
        return logits, targets, enc_cache, dec_cache, enc_extra, dec_extra

    def _seq2seq_dp(self, t, v, enc_cache=None, dec_cache=None, device_ids=[0], **kwargs):
        kwargs.update({"cache": enc_cache})
        memory, _, enc_cache, enc_extra = data_parallel(
            self.encoder_head, (t, v), device_ids=device_ids, module_kwargs=kwargs
        )
        kwargs.update({"cache": dec_cache, "memory": memory})
        logits, targets, dec_cache, dec_extra = data_parallel(
            self.decoder_head, (t, v), device_ids=device_ids, module_kwargs=kwargs
        )
        return logits, targets, enc_cache, dec_cache, enc_extra, dec_extra
    
    def infer(
        self, text=None, image=None, text_mask=None, analyze=False, 
        device_ids=[0], nsampling=1, topk=0.2, topp=0.995, debug=True, **kwargs
    ):
        num_dec_step = self.decoder_head.max_dec_len
        vprompt_step = self.decoder_head.beg_dec_len

        v_seq_gold = None  
        if image is not None and vprompt_step > 0:
            v_seq_gold = self.tokenize_images(image) # discretize images 
        
        v_seq_base_prompt = torch.tensor([[]] * text.shape[0], device=text.device).int()
        if vprompt_step > 0 and v_seq_gold is not None:
            v_seq_base_prompt = v_seq_gold[:, :vprompt_step]

        loss_all, outs_all = list(), list()
        ppl_all, image_all = list(), list()
        t_emb, *_ = self.embedder_head(text, v_seq=None)

        # there might be special tokens following the real visual tokens,
        # e.g., self.decoder_head.num_vis_token > self.vq.vocab_size,
        # but we shall generate only visual tokens so that the VQ model can decode
        num_vis_token = self.vq.vocab_size
        peep_kstep = min(1000, num_dec_step - vprompt_step) if debug else float("inf")
        if debug:
            self.echo(f"will decode {num_dec_step} steps {nsampling} times starting from the {vprompt_step}-th step")
        
        for _ in range(nsampling):
            v_seq = v_seq_base_prompt 
            enc_cache = dec_cache = old_cache = None 
            for istep in range(vprompt_step, num_dec_step):
                _, v_emb, *_ = self.embedder_head(None, v_seq=v_seq)
                 
                logits, _, enc_cache, dec_cache, *_ = self._seq2seq(
                    t_emb, v_emb, t_seq=text, v_seq=v_seq, text_mask=text_mask,   
                    enc_cache=enc_cache, dec_cache=dec_cache, use_cache=False, **kwargs,
                )

                old_cache = self._debug_cache(old_cache, logits, istep) if debug else None

                vlogits = logits[-1][:, -1, :num_vis_token].detach().clone() # (B, V)
                topk = int(vlogits.shape[-1] * topk) # topk is a fraction
                vlogits = top_k_top_p_filtering(vlogits, top_k=topk, top_p=topp)
                samples = torch.multinomial(vlogits.softmax(-1), 1)
                v_seq = torch.cat((v_seq, samples), dim=-1)

                if istep - vprompt_step > peep_kstep:
                    import sys; sys.exit(0)
            
            # eval the sampled sequence
            v_seq = v_seq[:, -num_dec_step:]
            _, v_emb, *_ = self.embedder_head(None, v_seq=v_seq)
            logits, targets, *_, enc_extra, dec_extra = self._seq2seq(
                t_emb, v_emb, t_seq=text, v_seq=v_seq, text_mask=text_mask, sampling=False, **kwargs,
            )
            *_, outs, more_dict = self.loss_head(logits, targets)
            outs_all.append(outs)
            
            ppl = more_dict.pop("v_ppl", None) # gpt model
            if ppl is None:
                ppl = more_dict.pop("ppl", None) # bart model
            assert ppl is not None, f"null ppl: something must be wrong."
            ppl_all.append(ppl)
            
            image = self.generate_images(v_seq)
            image_all.append(image)

        # rank images according to ppl
        ppl_all = torch.stack(ppl_all, dim=-1)   # (B, k)
        image_all = np.stack(image_all, axis=-1) # (B, H, W, C, k)
        
        indice = ppl_all.cpu().numpy().argsort(axis=-1)
        indice = np.expand_dims(indice, axis=(1, 2, 3))
        image_all = np.take_along_axis(image_all, indice, axis=-1)

        # post-process caches
        outs0 = np.mean([outs[0] for outs in outs_all])
        records = [outs[1] for outs in outs_all]
        num_record = len(records) if len(records) > 0 else 1
        outs1 = { 
            k: sum([record.get(k) for record in records]) / num_record for k in set().union(*records) 
        }
        outs1["image"] = image_all # sampled images 

        outs = (outs0, outs1)
        loss = self.set_stats({}, outs[-1])
        return loss, outs

    def infer_dp(self, text=None, image=None, text_mask=None, analyze=False, device_ids=[0], **kwargs):
        pass

    def forward(self, text=None, image=None, text_mask=None, analyze=False, device_ids=[0], **kwargs):
        infer = kwargs.get("infer", False)
        if False and infer and not self.training:
            inference_fn = self.infer_dp if len(device_ids) > 1 else self.infer
            with torch.autocast("cuda", enabled=False): # has to be disabled when using cache at test time
                return inference_fn(
                    text=text, image=image, text_mask=text_mask, analyze=analyze, device_ids=device_ids, **kwargs
                )
        if len(device_ids) > 1:
            return self.forward_dp(
                text=text, image=image, text_mask=text_mask, analyze=analyze, device_ids=device_ids, **kwargs
            )

        v_seq = self.tokenize_images(image) # discretize images 
        t_emb, v_emb, emb_extra = self.embedder_head(text, v_seq=v_seq)

        logits, targets, *_, enc_extra, dec_extra = self._seq2seq(
            t_emb, v_emb, t_seq=text, v_seq=v_seq, text_mask=text_mask, sampling=False, **kwargs,
        )
        
        loss, outs, *_ = self.loss_head(logits, targets)
        self.set_stats(dec_extra, outs[-1])

        v_peep_topk = kwargs.get("v_peep_topk", 0)
        if v_peep_topk is None or v_peep_topk > 0:
            with torch.autocast("cuda", enabled=False): # has to be disabled when using cache at test time
                extra_outs = self.peep_dp(
                    text=text, image=image, text_mask=text_mask, analyze=analyze, device_ids=device_ids, **kwargs
                )
            outs[-1].update(extra_outs)

        if analyze:
            self.analyze(
                t_emb=t_emb, v_emb=v_emb, t_seq=text, v_seq=v_seq, logits=logits, targets=targets,
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
            "memo_key_padding_mask": text_mask,
        })
        logits, targets, *_, enc_extra, dec_extra = self._seq2seq_dp(
            t_emb, v_emb, device_ids=device_ids, **kwargs
        )

        loss, outs, *_ = self.loss_head(logits, targets)
        self.set_stats(dec_extra, outs[-1])

        v_peep_topk = kwargs.get("v_peep_topk", 0)
        if v_peep_topk is None or v_peep_topk > 0:
            with torch.autocast("cuda", enabled=False): # has to be disabled when using cache at test time
                extra_outs = self.peep_dp(
                    text=text, image=image, text_mask=text_mask, analyze=analyze, device_ids=device_ids, **kwargs
                )
            outs[-1].update(extra_outs)
        
        if analyze:
            self.analyze(
                t_emb=t_emb, v_emb=v_emb, t_seq=text, v_seq=v_seq, logits=logits, targets=targets,
            )
        return loss, outs 

    @staticmethod
    def _topk(x, k):
        v, _ = torch.topk(x, k)
        o = x.clone()
        o[o < v[..., [-1]]] = -float("inf")
        return o

    def sampling(
        self, text, c_seq, nstep, temperature=1., topk=100, random=True, device_ids=[0], **kwargs
    ):
        kwargs = copy.deepcopy(kwargs)
        for k in range(nstep):
            t_emb, v_emb, emb_extra = self.embedder_head(text, v_seq=c_seq)
            kwargs.update({"t_seq": text, "v_seq": c_seq, "force_infer": True})
            logits, *_ = self._seq2seq_dp(
                t_emb, v_emb, device_ids=device_ids, use_cache=False, **kwargs
            )
            logits = logits[-1][:, -1] / temperature
            logits = logits[:, :self.vq.vocab_size].detach().clone() # (B, V)
            if topk is not None:
                logits = self._topk(logits, topk)
            pdists = F.softmax(logits, dim=-1)
            if random:
                idx = torch.multinomial(pdists, num_samples=1)
            else:
                *_, idx = torch.topk(pdists, k=1, dim=-1)
            c_seq = torch.cat((c_seq, idx), dim=1)
        return c_seq
    
    @torch.no_grad()
    def peep_dp(
        self, text=None, image=None, text_mask=None, analyze=False, device_ids=[0], v_peep_topk=None, **kwargs
    ):
        text = text[:v_peep_topk]
        image = image[:v_peep_topk]
        v_seq = data_parallel(self.vq, image, device_ids=device_ids)
        #v_seq = F.pad(v_seq, (1, 0), value=0) # BOS of visual seq

        B, L = v_seq.shape[:2]

        # given a half
        c_seq = v_seq[:, :(L - 0) // 2]
        nstep = L - c_seq.shape[1]
        v = self.sampling(text, c_seq, nstep, device_ids=device_ids, random=True)
        sample_half = self.generate_images(v)
        
        # sampling 
        c_seq = v_seq[:, :0]
        nstep = L - c_seq.shape[1]
        v = self.sampling(text, c_seq, nstep, device_ids=device_ids, random=True)
        sample_full = self.generate_images(v)

        # deterministic
        c_seq = v_seq[:, :0]
        nstep = L - c_seq.shape[1]
        v = self.sampling(text, c_seq, nstep, device_ids=device_ids, random=False)
        sample_hard = self.generate_images(v)

        # reconstruction
        sample_pred = self.generate_images(v_seq)
        
        outs = {
            "_sample_half": sample_half,
            "_sample_full": sample_full,
            "_sample_hard": sample_hard,
            "_sample_pred": sample_pred,
            "nsample": v_peep_topk,
        }
        return outs

    def stats(self): 
        world_size = 1 #get_world_size()
        meter = self.meter_train if self.training else self.meter_infer
        stats = meter.stats

        if stats.get("couple", False):
            nstep = stats["nstep"]
            alpha = world_size / nstep if nstep > 0 else 0
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
            alpha = 1 / ntoken * 100 if ntoken > 0 else 0
            acc = stats["ntrue"] * alpha

            nsample = stats["nsample"]
            alpha = 1 / nsample * 1 if nsample > 0 else 0
            t_ppl = stats["t_ppl"] * alpha
            v_ppl = stats["v_ppl"] * alpha

            info += f"t_ppl {t_ppl:.3f} v_ppl {v_ppl:.3f} t_acc {t_acc:.3f} v_acc {v_acc:.3f} acc {acc:.3f}"
        else:
            nstep = stats["nstep"]
            alpha = world_size / nstep if nstep > 0 else 0
            loss = stats["main_loss"] * alpha 

            ntoken = stats["ntoken"]
            alpha = 1 / ntoken * 100 if ntoken > 0 else 0
            acc = stats["ntrue"] * alpha

            nsample = stats["nsample"]
            alpha = 1 / nsample * 1 if nsample > 0 else 0
            ppl = stats["ppl"] * alpha

            info = f"loss {loss:.5f} ppl {ppl:.3f} acc {acc:.3f}"

        info = f"{info} ({nsample})"
        return info

    def eval_metric(self):
        meter = self.meter_train if self.training else self.meter_infer
        stats = meter.stats

        nsample = stats["nsample"]
        alpha = 1 / nsample * 1 if nsample > 0 else 0

        if stats.get("couple", False):
            t_ppl = stats["t_ppl"] * alpha
            v_ppl = stats["v_ppl"] * alpha
            ppl = (t_ppl + v_ppl) / 2
        else:
            ppl = stats["ppl"] * alpha
        return -ppl 

    def tokenize_images(self, v):
        return self.vq.encode(v)

    def generate_images(self, v):
        return self.vq.decode(v)

    def register_vq(self, mcfg):
        vq = build_encoder_head(mcfg.vq, None)
        return vq.emb_weight, vq

    def build(self, encoder_vocab, decoder_vocab=None, **kwargs):
        tunable_params = dict()
        mcfg = self.cfg.model 
        
        if mcfg.vq.load: # do we need to load it?
            vq_embed, vq = self.register_vq(mcfg)
            vq_vocab_size, emb_size = vq_embed.shape[:2]

            if decoder_vocab is None:
                decoder_vocab = [f"{i}" for i in range(vq_vocab_size)]
        else:
            vq = None

        if self.cfg.eval:
            self.vq = vq # might need the model to decode codes
            local_cfg, head_sd = load_checkpoint(self.cfg, self.echo)

            self.embedder_head = build_encoder_head(
                mcfg.embedder, encoder_vocab, vis_token_vocab=decoder_vocab
            )
            self.encoder_head = build_encoder_head(mcfg.encoder, encoder_vocab, vis_token_vocab=decoder_vocab)
            self.decoder_head = build_decoder_head(mcfg.decoder, encoder_vocab, vis_token_vocab=decoder_vocab)
            self.loss_head = build_loss_head(mcfg.loss, encoder_vocab, vis_token_vocab=decoder_vocab)
            if head_sd is not None:
                self.embedder_head.load_state_dict(head_sd["embedder_head"])
                self.encoder_head.load_state_dict(head_sd["encoder_head"])
                self.decoder_head.load_state_dict(head_sd["decoder_head"])
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
