from omegaconf import OmegaConf
import os, re
import torch

from ..module import from_pretrained_vqgan

def load_checkpoint(cfg, echo):
    model_file = f"{cfg.model_root}/{cfg.model_name}/{cfg.model_file}"
    try:
        checkpoint = torch.load(model_file, map_location="cpu")
        echo(f"Loading from {model_file}")
    except Exception as e:
        echo(f"Failed to load the checkpoint `{model_file}` {e}")
        return (None,) * 2
    local_cfg = checkpoint["cfg"]
    local_str = OmegaConf.to_yaml(local_cfg)
    if cfg.verbose:
        echo(f"Old configs:\n\n{local_str}")
    return local_cfg, checkpoint["model"]
