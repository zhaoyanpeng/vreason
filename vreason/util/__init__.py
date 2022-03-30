import os, sys
import logging
import random
import numpy
import torch
import torch.distributed as dist

from .module import * 
from .function import *

def seed_all_rng(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

def setup_logger(output_dir=None, name="pcfg", rank=0, output=None, fname="train"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if rank == 0:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(console)
    if os.path.exists(output_dir):
        logger.info(f'Warning: the folder {output_dir} exists.')
    else:
        logger.info(f'Creating {output_dir}')
        if rank == 0: 
            os.makedirs(output_dir)
    if torch.distributed.is_initialized():
        dist.barrier() # output dir should have been ready
    if output is not None:
        filename = os.path.join(output_dir, f'{fname}_{rank}.out')
        handler = logging.FileHandler(filename, 'w')
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

def numel(model: torch.nn.Module, trainable: bool = False):
    parameters = list(model.parameters())
    if trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique) 

def detect_nan(x):
    return torch.isnan(x).any(), torch.isinf(x).any()

def enable_print():
    sys.stdout = sys.__stdout__

def disable_print():
    sys.stdout = open(os.devnull, 'w')
