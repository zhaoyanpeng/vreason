from omegaconf import OmegaConf
import os, re
import torch

try:
    from taming.models.vqgan import VQModel, GumbelVQ
except Exception as e:
    VQModel = GumbelVQ = None

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def from_pretrained_vqgan(cfg):
    model_root, model_time, model_name, model_file = \
        cfg.model_root, cfg.model_time, cfg.model_name, cfg.model_file  

    model_name = f"{model_time}_{model_name}"
    model_file = f"{model_name}/checkpoints/{model_file}"
    model_conf = f"{model_name}/configs/{model_time}-project.yaml"

    configs = load_config(f"{model_root}/{model_conf}", display=False)
    return load_vqgan(configs, ckpt_path=f"{model_root}/{model_file}")

def load_checkpoint(cfg, echo):
    model_file = f"{cfg.model_root}/{cfg.model_name}/{cfg.model_file}"
    try:
        checkpoint = torch.load(model_file, map_location="cpu")
        echo(f"Loading from {model_file}")
    except Exception as e:
        echo(f"Failed to load the checkpoint `{model_file}` {e}")
        return (None,) * 6
    local_cfg = checkpoint["cfg"]
    local_str = OmegaConf.to_yaml(local_cfg)
    if cfg.verbose:
        echo(f"Old configs:\n\n{local_str}")
    return local_cfg, checkpoint["model"]
