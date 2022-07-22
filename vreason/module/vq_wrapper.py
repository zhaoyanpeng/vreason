import numpy as np
import os, sys, re, time
import torch
from typing import Tuple
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from omegaconf import OmegaConf
import torch.nn.functional as F

try:
    from taming.models.vqgan import VQModel, GumbelVQ
except Exception as e:
    VQModel = GumbelVQ = None

from . import MetaEncHead, ENCODER_HEADS_REGISTRY
from . import _get_activation_fn, SoftPositionalEncoder, SlotAttnBlock

__all__ = ["PretrainedVQGAN", "from_pretrained_vqgan"]


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan(config, ckpt_path=None, gumbel=False):
    if gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=True) #False)
        print(f"loaded from {ckpt_path}")
    return model.eval()

def from_pretrained_vqgan(cfg, restore=True):
    model_root, model_time, model_name, model_file = \
        cfg.model_root, cfg.model_time, cfg.model_name, cfg.model_file  

    model_name = f"{model_time}_{model_name}"
    model_file = f"{model_name}/checkpoints/{model_file}"
    model_conf = f"{model_name}/configs/{model_time}-project.yaml"

    configs = load_config(f"{model_root}/{model_conf}", display=False)
    ckpt_path = f"{model_root}/{model_file}" if restore else None
    return load_vqgan(configs, ckpt_path=ckpt_path, gumbel=cfg.gumbel)


@ENCODER_HEADS_REGISTRY.register()
class PretrainedVQGAN(MetaEncHead):
    def __init__(self, cfg, token_vocab, restore=True, **kwargs):
        super().__init__(cfg, token_vocab)
        self.gumbel = cfg.gumbel
        model_root, model_time, model_name, model_file = \
            cfg.model_root, cfg.model_time, cfg.model_name, cfg.model_file  

        model_name = f"{model_time}_{model_name}"
        model_file = f"{model_name}/checkpoints/{model_file}"
        model_conf = f"{model_name}/configs/{model_time}-project.yaml"

        configs = load_config(f"{model_root}/{model_conf}", display=False)
        ckpt_path = f"{model_root}/{model_file}" if restore else None
        self.vq = load_vqgan(configs, ckpt_path=ckpt_path, gumbel=cfg.gumbel)
        self.vocab_size = self.emb_weight.shape[0]
    
    @property
    def emb_weight(self):
        if self.gumbel:
            weight = self.vq.quantize.embed.weight
        else:
            weight = self.vq.quantize.embedding.weight
        return weight 
    
    @torch.no_grad()
    def encode(self, v):
        B = v.shape[0]
        *_, (*_, codes) = self.vq.encode(v)
        codes = codes.reshape((B, -1))
        return codes

    @torch.no_grad()
    def decode(self, v):
        B, L = v.shape[:2]
        H = W = int(L ** 0.5)
        z = F.one_hot(v, num_classes=self.vocab_size).float()
        z = z @ self.emb_weight
        z = z.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        h = self.vq.decode(z)
        
        x = ( # E.g., Image.fromarray(x[0])
            (h.detach().cpu().clamp(-1, 1) + 1.) / 2 * 255.
        ).permute(0, 2, 3, 1).numpy().astype(np.uint8)
        return x 

    @torch.no_grad()
    def forward(self, v, *args, **kwargs):
        return self.encode(v) 
