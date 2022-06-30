import os
import json
import torch
import numpy as np
import itertools
from PIL import Image
from omegaconf.listconfig import ListConfig

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode 

from . import DatasetCatalog, register_indexer, build_dataloader
from ..util import shorten_name 
from .clevr import ClevrImageCollator
from .abscene_function import *

class AbsceneImageData(torch.utils.data.Dataset):
    def __init__(
        self, cfg, data_file, image_path, decoder_vocab=None,
        encoder_vocab=None, cate_vocab=None, train=True, chunk=None, #slice(0, None)
    ):
        # the dataset is provided as a whole, we have to split it into train, val, and test sets.
        self.decoder_vocab = decoder_vocab
        self.encoder_vocab = encoder_vocab
        # should be a f-str or can be formatted
        self.vfile = image_path
        self.train = train

        # center it w/ a hard-coded center
        self.resize_size = (
            [cfg.resize_size] * 2 if isinstance(cfg.resize_size, int)
            else list(cfg.resize_size)[:2]
        )
        self.crop_size = list(cfg.crop_size)[:4]

        self.max_num_obj = cfg.max_num_obj
        self.add_bg_mask = cfg.add_bg_mask

        # read metadata 
        droot = cfg.data_root if cfg.data_root == cfg.more_root else cfg.more_root 
        self.vroot = f"{droot}/RenderedScenes" # input image dir 
        self.oroot = f"{cfg.dump_root}"        # output data dir 
        self.components, self.scene_layouts = load_artifacts(droot, seed=1213)

        # load data
        self.dataset = list()
        self.color = (255, 255, 255) 
        self.num_image_per_class = 10
        self.max_num_image_class = 1002
        keyset = (
            range(chunk[0], min(chunk[1], self.max_num_image_class))
            if chunk is not None else range(0, self.max_num_image_class)
        )
        iskip = jskip = total = 0
        for vclass in keyset:
            for iimage in range(self.num_image_per_class):
                total += 1
                image, confs, vfile = load_image_from_class(
                    self.scene_layouts, self.vroot, vclass, iimage
                )

                nobj = len(confs)
                if nobj > self.max_num_obj:
                    jskip += 1
                    continue # filter

                names, bboxes, masks, arts = get_name_box_mask_art(confs, self.components)

                bin_masks = dict()
                if not train:
                    *_, bin_masks = get_object_masks(
                        [image], reversed(arts[:]), reversed(masks[:]), reversed(bboxes[:]), 
                        n_per_line=1, margin=0, color=self.color, mode="RGBA"
                    )
                    bin_masks = {k: v for k, v in enumerate(bin_masks)}

                item = {
                    "image": vfile,
                    "bbox": {k + 1: bboxes[k] for k in range(nobj)},
                    "mask": bin_masks, # dict
                    "name": {k + 1: names[k] for k in range(nobj)},
                    "nobj": nobj, 
                }
                self.dataset.append(item)
        #print(iskip, jskip, total, keyset, len(self.dataset))
        
        # load only image indice
        self.length = len(self.dataset)
        self.version = cfg.version

    def __len__(self):
        return self.length
    
    def preprocess_image(self, sample, mode="RGB"):
        vfile = sample["image"]

        image_raw = Image.open(vfile)
        if not image_raw.mode == mode:
            image_raw = image_raw.convert(mode)

        image = TF.crop(*([image_raw] + self.crop_size))
        image = TF.resize(image, self.resize_size, interpolation=InterpolationMode.BILINEAR)
        image = (np.array(image) / 127.5 - 1.0).astype(np.float32).clip(-1., 1.)

        mask = np.array([], dtype=np.float32) 
        if len(sample["mask"]) > 0:
            y, x, h, w = self.crop_size
            mask = {
                k: np.array(TF.resize(
                    Image.fromarray(m[y : y + h, x : x + w] * 255),
                    self.resize_size, interpolation=InterpolationMode.BILINEAR
                )) for k, m in sample["mask"].items()
            }
            #mask = np.stack([mask[i] for i in range(sample["nobj"] + 1)], axis=0)
            pad = [np.zeros(self.resize_size, dtype=np.float32)] * (self.max_num_obj - sample["nobj"]) 
            if self.add_bg_mask:
                mask = np.stack([mask[i] for i in range(sample["nobj"] + 1)] + pad, axis=0)
            else:
                mask = np.stack([mask[i + 1] for i in range(sample["nobj"])] + pad, axis=0)
        return image, mask, vfile

    def __getitem__(self, index):
        sample = self.dataset[index]
        image, mask, vfile = self.preprocess_image(sample)
        return {
            "image": image, 
            "vfile": vfile,
            "bbox": sample["bbox"], 
            "name": sample["name"], 
            "nobj": sample["nobj"], 
            "mask": mask, 
        }

def build_abscene_image_data(cfg, train, echo):
    dataloader = evalloader = testloader = encoder_vocab = None
    encoder_vocab = decoder_vocab = cate_vocab = None 

    def get_sample_chunk(chunk):
        if isinstance(chunk, (tuple, list, ListConfig)): 
            return list(chunk)
        return None 

    pin_memory = False

    # train
    chunk = cfg.data_name if train else cfg.eval_name
    chunk = get_sample_chunk(chunk)
    dataset = AbsceneImageData(
        cfg, None, None, encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab,
        cate_vocab=cate_vocab, train=train, chunk=chunk
    )
    dataloader = build_dataloader(
        cfg, dataset, train, ClevrImageCollator(), echo, msg=f"main", pin_memory=pin_memory
    )

    # eval
    chunk = cfg.eval_name if train else None 
    if chunk is not None:
        chunk = get_sample_chunk(chunk)
        dataset = AbsceneImageData(
            cfg, None, None, encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab,
            cate_vocab=cate_vocab, train=False, chunk=chunk
        )
        evalloader = build_dataloader(
            cfg, dataset, False, ClevrImageCollator(), echo, msg=f"eval", pin_memory=pin_memory
        )

    # test
    chunk = cfg.test_name if train else None 
    if chunk is not None:
        chunk = get_sample_chunk(chunk)
        dataset = AbsceneImageData(
            cfg, None, None, encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab,
            cate_vocab=cate_vocab, train=False, chunk=chunk
        )
        testloader = build_dataloader(
            cfg, dataset, False, ClevrImageCollator(), echo, msg=f"test", pin_memory=pin_memory
        )
    return dataloader, evalloader, testloader, encoder_vocab, decoder_vocab, cate_vocab
