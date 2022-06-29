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

def get_binary_masks(masks, width=480, height=320):
    def str2mask(mask):
        cur, img = 0, []
        for v in mask.strip().split(","):
            v = int(v)
            img.extend([cur] * v)
            cur = 1 - cur
        return np.array(img)
    
    bin_masks = dict()
    npixel = width * height
    for imask, mask in masks.items():
        mask = str2mask(mask)
        mask = mask.reshape((height, width)).astype(np.uint8)
        bin_masks[int(imask)] = mask
    return bin_masks

class ClevrImageData(torch.utils.data.Dataset):
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

        # Center it w/ a hard-coded center
        self.resize_size = (
            [cfg.resize_size] * 2 if isinstance(cfg.resize_size, int)
            else list(cfg.resize_size)[:2]
        )
        self.crop_size = list(cfg.crop_size)[:4]

        self.max_num_obj = cfg.max_num_obj

        # load data
        self.dataset = list()
        keyset = set(list(range(chunk[0], chunk[1]))) if chunk is not None else None
        scenes = json.load(open(data_file, "r"))["scenes"]
        iskip = jskip = 0
        for scene in scenes:
            nobj = len(scene["objects"])
            image_idx = scene["image_index"]
            if keyset is not None and image_idx not in keyset:
                iskip += 1
                continue # a chunk
            if nobj > self.max_num_obj:
                jskip += 1
                continue # filter
            bin_masks = dict()
            if not train:
                bin_masks = get_binary_masks(scene["obj_mask"])
            name = {
                obj["idx"]: [obj["size"], obj["color"], obj["material"], obj["shape"]] for obj in scene["objects"]
            }
            item = {
                "image": scene["image_index"],
                "bbox": scene["obj_bbox"],
                "mask": bin_masks, # list 
                "name": name,
                "nobj": nobj, 
            }
            self.dataset.append(item)
        #print(iskip, jskip, len(scenes), len(keyset) if keyset is not None else 0)
        
        # load only image indice
        self.length = len(self.dataset)
        self.version = cfg.version

    def __len__(self):
        return self.length
    
    def preprocess_image(self, sample, mode="RGB"):
        index = sample["image"]
        vfile = self.vfile.format(index)

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
            #mask = np.stack([mask[i + 1] for i in range(len(sample["mask"]))], axis=0)
            pad = [np.zeros(self.resize_size, dtype=np.float32)] * (self.max_num_obj - sample["nobj"]) 
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

class ClevrImageCollator:
    def __init__(self, device=torch.device("cpu"), vocab=None):
        self.device = device
        self.vocab = vocab

    def _naive_collator(self, union):
        image = np.stack(union["image"], axis=0) 
        mask = np.stack(union["mask"], axis=0)
        """
        mask = union["mask"]
        shape = mask[0].shape[-2:]
        if shape != (0,):
            mask = np.array(list(itertools.zip_longest(
                *mask, fillvalue=np.zeros(shape, dtype=np.float32)
            ))).swapaxes(0, 1)
        else:
            mask = np.stack(mask, axis=0)
        """
        return {
            "image": image,
            "vfile": union["vfile"],
            "nobj": union["nobj"],
            "bbox": union["bbox"], 
            "name": union["name"], 
            "mask": mask,
        }

    def __call__(self, records):
        union = { 
            k: [record.get(k) for record in records] for k in set().union(*records) 
        }
        return self._naive_collator(union)

def build_clevr_image_data(cfg, train, echo):
    dataloader = evalloader = testloader = encoder_vocab = None
    encoder_vocab = decoder_vocab = cate_vocab = None 

    def get_image_path(data_file):
        data_file = os.path.basename(data_file)
        if "train" in data_file:
            name = "/images/train/CLEVR_train_{:06}.png"
        elif "test" in data_file:
            name = "/images/test/CLEVR_test_{:06}.png"
        elif "val" in data_file:
            name = "/images/val/CLEVR_val_{:06}.png"
        else:
            raise ValueError("cannot figure out data split")
        return cfg.data_root + name 

    def get_sample_chunk(chunk):
        if isinstance(chunk, (tuple, list, ListConfig)): 
            return list(chunk)
        return None 

    pin_memory = False

    # train
    name = cfg.data_name if train else cfg.eval_name
    ifile = f"{cfg.data_root}/scenes/{name}"
    assert os.path.isfile(ifile), f"not a data file {ifile}"
    image_path = get_image_path(ifile)
    chunk = cfg.train_samples if train else cfg.eval_samples
    chunk = get_sample_chunk(chunk)
    dataset = ClevrImageData(
        cfg, ifile, image_path, encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab,
        cate_vocab=cate_vocab, train=train, chunk=chunk
    )
    dataloader = build_dataloader(
        cfg, dataset, train, ClevrImageCollator(), echo, msg=f"main", pin_memory=pin_memory
    )

    # eval
    ifile = f"{cfg.data_root}/scenes/{cfg.eval_name}" if train else "IGNORE_ME"
    if os.path.isfile(ifile):
        image_path = get_image_path(ifile)
        chunk = get_sample_chunk(cfg.eval_samples)
        dataset = ClevrImageData(
            cfg, ifile, image_path, encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab,
            cate_vocab=cate_vocab, train=False, chunk=chunk
        )
        evalloader = build_dataloader(
            cfg, dataset, False, ClevrImageCollator(), echo, msg=f"eval", pin_memory=pin_memory
        )

    # test
    ifile = f"{cfg.data_root}/scenes/{cfg.test_name}" if train else "IGNORE_ME"
    if os.path.isfile(ifile):
        image_path = get_image_path(ifile)
        chunk = get_sample_chunk(cfg.test_samples)
        dataset = ClevrImageData(
            cfg, ifile, image_path, encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab,
            cate_vocab=cate_vocab, train=False, chunk=chunk
        )
        testloader = build_dataloader(
            cfg, dataset, False, ClevrImageCollator(), echo, msg=f"test", pin_memory=pin_memory
        )
    return dataloader, evalloader, testloader, encoder_vocab, decoder_vocab, cate_vocab
