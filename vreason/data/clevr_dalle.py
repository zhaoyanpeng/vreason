import os
import re
import json
import torch
import warnings
import numpy as np
import itertools
from PIL import Image
import albumentations
import pandas as pd
from functools import partial
from braceexpand import braceexpand
from omegaconf.listconfig import ListConfig

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode 

from . import DatasetCatalog, register_indexer, build_dataloader
from .clevr_function import *
from ..util import shorten_name 


class ClevrImageTextData(torch.utils.data.Dataset):
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
        self.preprocessor = albumentations.Compose([
            albumentations.SmallestMaxSize(max_size=self.resize_size[0]),
            albumentations.CenterCrop(height=self.crop_size[-2],width=self.crop_size[-1])
        ])

        self.max_txt_len = cfg.max_txt_len
        self.max_txt_num = cfg.max_txt_num
        self.min_txt_num = cfg.min_txt_num

        split_name = "train" if self.train else "val"
        scene_file = f"{cfg.more_root}/scenes/CLEVR_{split_name}_scenes.all.json"
        if not os.path.isfile(scene_file):
            txt_data = json.load(open(data_file, "r"))["captions"]
            all_keys = list({int(k) for k, _ in txt_data.items()})
        else: # collect keys from the scene data
            min_obj_num = max(cfg.min_obj_num, 1)
            max_obj_num = min(cfg.max_obj_num, 10)
            scene_data = json.load(open(scene_file, "r"))
            all_keys = list({
                int(k) for k, scene in scene_data["scenes"].items()
                if (
                    len(scene["objects"]) <= max_obj_num and
                    len(scene["objects"]) >= min_obj_num
                )
            })
            txt_data = json.load(open(data_file, "r"))["captions"]
        
        local_seed = 8080
        keyset = all_keys
        # select (a subset of) data keys
        if chunk is not None:
            np.random.default_rng(local_seed).shuffle(all_keys)
            slicer = slice(chunk[0], chunk[1])
            keyset = set(all_keys[slicer])

        iskip = jskip = 0
        self.dataset = list()
        # load (a subset of) data samples
        for key in keyset:
            key = str(key)
            if key in txt_data:
                self.dataset.append((int(key), txt_data[key])) 
            else:
                self.dataset.append((int(key), [])) 
                iskip += 1
        #print(iskip, jskip, len(scenes), len(keyset) if keyset is not None else 0)
        
        self.length = len(self.dataset)
        self.version = cfg.version
        # oops, better do all here but ...
        self._additional_init(cfg, data_file=data_file)

    def _additional_init(self, cfg, **kwargs):
        pass

    def __len__(self):
        return self.length
    
    def preprocess_image(self, sample=None, index=-1, mode="RGB"):
        #index = sample["image"]
        vfile = self.vfile.format(index)
        image_raw = Image.open(vfile)
        if not image_raw.mode == mode:
            image_raw = image_raw.convert(mode)
        image = np.array(image_raw).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image, None, vfile

    def preprocess_text(self, captions): 
        if self.train:
            num_caption = len(captions)
            if self.max_txt_num >= num_caption:
                selected = captions
            else: # sample a number & sample a number of text
                assert self.min_txt_num < self.max_txt_num + 1, f"{self.min_txt_num} < {self.max_txt_num} + 1"
                k = np.random.choice(range(self.min_txt_num, self.max_txt_num + 1), 1)[0]
                selected = np.random.choice(captions, k, replace=False)
        else: # use a single caption at test time
            #k = (self.min_txt_num + self.max_txt_num) // 2
            selected = captions #[:k] #self.min_txt_num]

        selected = " ".join(selected)

        caption = [self.encoder_vocab.BOS] + selected.lower().split() #" ")
        caption = self.encoder_vocab(caption)
        caption = caption + [self.encoder_vocab.PAD_IDX] * (self.max_txt_len - len(caption))
        caption = caption[:self.max_txt_len] # truncate whatever
        return caption

    def __getitem__(self, index):
        index, captions = self.dataset[index]

        caption = self.preprocess_text(captions)
        image, _, vfile = self.preprocess_image(index=index)
        return {
            "text": caption,
            "image": image, 
            "vfile": vfile,
        }

class ClevrImageData(ClevrImageTextData):
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
        self.preprocessor = albumentations.Compose([
            albumentations.SmallestMaxSize(max_size=self.resize_size[0]),
            albumentations.CenterCrop(height=self.crop_size[-2],width=self.crop_size[-1])
        ])

        # load (a subset of) data
        keyset = set(list(range(chunk[0], chunk[1]))) if chunk is not None else None
        max_image_id= 70000 if train else 15000
        valid_keyset = set(list(range(max_image_id)))
        self.dataset = sorted(keyset & valid_keyset) if keyset is not None else sorted(valid_keyset)
        
        # load only image indice
        self.length = len(self.dataset)
        self.version = cfg.version

    def preprocess_text(self, captions):
        caption = [self.encoder_vocab.BOS_IDX]
        return caption

    def __getitem__(self, index):
        index = self.dataset[index]

        caption = self.preprocess_text(None)
        image, _, vfile = self.preprocess_image(index=index)
        return {
            "text": caption,
            "image": image, 
            "vfile": vfile,
        }

class ClevrImageTextPandasForDalleMini(torch.utils.data.Dataset):
    def __init__(
        self, cfg, data_file, image_path, decoder_vocab=None,
        encoder_vocab=None, cate_vocab=None, train=True, chunk=None, #slice(0, None)
    ):
        # the dataset is provided as a whole, we have to split it into train, val, and test sets.
        self.decoder_vocab = decoder_vocab
        self.encoder_vocab = encoder_vocab
        self.train = train

        self.require_txt = getattr(cfg, "require_txt", True)
        self.rot90_image = getattr(cfg, "rot90_image", 0)
        self.object_name = getattr(cfg, "object_name", None)
        self.vis_fake_hw = getattr(cfg, "vis_fake_hw", 0) # H x W
        self.vis_fake_hw = (
            [self.vis_fake_hw] * 2 if isinstance(self.vis_fake_hw, int) else list(self.vis_fake_hw)[:2]
        )
        self.obj_fake_hw = list(getattr(cfg, "obj_fake_hw", [])) # list of H x W
        self.obj_delt_hw = list(getattr(cfg, "obj_delt_hw", [])) # H x W
        self.rng_fake_hw = np.random.default_rng(7879)
        self.max_vis_len = cfg.vis_vocab_size
        self.max_txt_len = cfg.max_txt_len
        self.max_txt_num = cfg.max_txt_num
        self.min_txt_num = cfg.min_txt_num

        if isinstance(self.obj_fake_hw, list): # will fill object areas
            self.obj_fake_id = self.rng_fake_hw.choice(
                range(1, self.max_vis_len), len(self.obj_fake_hw), replace=False
            )
        
        # read
        all_df = list()
        for dfile in braceexpand(data_file):
            df = pd.read_parquet(dfile)
            all_df.append(df)
        all_df = pd.concat(all_df)
        
        # filter
        min_obj_num = max(cfg.min_obj_num, 1)
        max_obj_num = min(cfg.max_obj_num, 10)
        all_df = all_df[(all_df['object_num'] >= min_obj_num) & (all_df['object_num'] <= max_obj_num)]

        # more filter by image index
        if train:
            divider = cfg.train_divider
            min_vid = max(cfg.min_train_vid, 0)
            max_vid = min(cfg.max_train_vid, 1e9)
        else:
            divider = cfg.eval_divider
            min_vid = max(cfg.min_eval_vid, 0)
            max_vid = min(cfg.max_eval_vid, 1e9)
        
        # add a new vid column
        def parse_vid(row):
            try:
                x = row["image"]
                x = re.match(".*?_(\d+)\.png", x)
                x = int(x.groups()[0])
                return x
            except Exception as e:
                return -1
        all_df["vid"] = all_df.apply(parse_vid, axis=1)

        # scene data
        self.H = self.W = 16
        split = "train" if self.train else "val"
        scene_file = f"{cfg.more_root}/scenes/CLEVR_{split}_scenes.all.json"
        scene_data = json.load(open(scene_file, "r"))["scenes"]
        self.bbox_dict = extract_bbox(
            all_df, scene_data, d=cfg.min_pixel_num / self.H, rotate=self.rot90_image > 0.
        )
        
        # filter by remainder
        def filter_by_vid(row, min_vid, max_vid, divider, scenes, object_name):
            try:
                #x = row["image"]
                #x = re.match(".*?_(\d+)\.png", x)
                #x = int(x.groups()[0]) % divider
                x = row.vid % divider
                if x < min_vid or x > max_vid:
                    return False
                if scenes is not None and object_name is not None:
                    for obj in scenes[str(row.vid)]["objects"]:
                        name = obj["category"]
                        if re.search(object_name, name) is not None:
                            return True
                    return False
                return True
            except Exception as e:
                return False 
        partial_vid_filter_fn = partial(
            filter_by_vid,
            min_vid=min_vid,
            max_vid=max_vid,
            divider=divider,
            object_name=self.object_name,
            scenes=scene_data,
        )
        all_df = all_df[all_df.apply(partial_vid_filter_fn, axis=1)]

        # select
        if chunk is not None:
            slicer = slice(chunk[0], chunk[1])
            all_df = all_df[slicer]
        
        self.dataset = all_df
        self.length = len(self.dataset)
        self.version = cfg.version
        # oops, better do all here but ...
        self._additional_init(cfg, data_file=data_file)

    def _additional_init(self, cfg, **kwargs):
        pass

    def __len__(self):
        return self.length

    def preprocess_image(self, sample=None, index=-1, mode="RGB"):
        if isinstance(self.vis_fake_hw, list): # fake images
            codes, vfile = [], sample.image
            #codes = np.random.randint(0, high=self.max_vis_len, size=self.vis_fake_hw)
            #codes = np.random.permutation(self.max_vis_len)[:np.prod(self.vis_fake_hw)]
            cnt, max_try = 0, 16
            create_2obj_fn = create_2obj_image if len(self.obj_delt_hw) == 0 else create_2obj_fixed
            while len(codes) == 0:
                codes, bbox = create_2obj_fn(
                    self.vis_fake_hw, self.obj_fake_hw, self.obj_fake_id, self.rng_fake_hw, self.obj_delt_hw
                )
                cnt += 1
                if cnt >= max_try:
                    codes = np.zeros(self.vis_fake_hw).astype(int)
                    warnings.warn(f"exceed max_try and return all zeros")
                if len(codes) == 0:
                    continue
                for idx, hw in zip(self.obj_fake_id, self.obj_fake_hw):
                    if (idx == codes).sum() != np.prod(hw):
                        warnings.warn(f"bug in \n{codes}")
                        codes = []
            #print(codes, codes.shape)
            codes = codes.flatten() if codes.ndim > 1 else codes
            return codes, bbox, vfile

        codes = np.array(sample.encoding)
        if self.rot90_image > 0. and np.random.uniform() < self.rot90_image:
            codes = np.rot90(codes.reshape(self.H, self.W)).flatten()
        vfile = sample.image
        return codes, None, vfile
        
    def preprocess_text(self, captions): 
        if self.train:
            num_caption = len(captions)
            if self.max_txt_num >= num_caption:
                selected = captions
            else: # sample a number & sample a number of text
                assert self.min_txt_num < self.max_txt_num + 1, f"{self.min_txt_num} < {self.max_txt_num} + 1"
                k = np.random.choice(range(self.min_txt_num, self.max_txt_num + 1), 1)[0]
                selected = np.random.choice(captions, k, replace=False)
        else: # use a single caption at test time
            #k = (self.min_txt_num + self.max_txt_num) // 2
            selected = captions #[:k] #self.min_txt_num]
        
        selected = " ".join(selected)

        caption = [self.encoder_vocab.BOS] + selected.lower().split() #" ")
        caption = self.encoder_vocab(caption)
        caption = caption + [self.encoder_vocab.PAD_IDX] * (self.max_txt_len - len(caption))
        caption = caption[:self.max_txt_len] # truncate whatever
        return caption

    def __getitem__(self, index):
        sep = " SEP " # special symbol
        row = self.dataset.iloc[index]
        caption = (
            self.preprocess_text(row.caption.split(" SEP "))
            if self.require_txt else [self.encoder_vocab.BOS_IDX]           
        )
        image, bbox, vfile = self.preprocess_image(sample=row)
        bbox = self.bbox_dict[row.image] if bbox is None else bbox
        return {
            "bbox": bbox,
            "text": caption,
            "image": image, 
            "vfile": vfile,
        }

class ClevrImageTextDataForDalleMini(ClevrImageTextData):
    def _additional_init(self, cfg, data_file=None):
        scene_file = f"{cfg.more_root}/scenes/CLEVR_{cfg.split}_scenes.all.json"
        
        assert os.path.isfile(scene_file), f"scene file `{scene_file}` does not exist."
        scene_data = json.load(open(scene_file, "r"))
        
        assert os.path.isfile(data_file), f"caption file `{data_file}` does not exist."
        txt_data = json.load(open(data_file, "r"))["captions"]

        self.dataset = [
            (int(k), txt_data[k], len(scene["objects"]))
            if k in txt_data else
            (int(k), [], len(scene["objects"]))
            for k, scene in scene_data["scenes"].items()
        ]

        local_seed = 1213 
        np.random.default_rng(local_seed).shuffle(self.dataset)
        self.length = len(self.dataset)

    def __getitem__(self, index):
        sep = " SEP " # special symbol
        index, captions, num_obj = self.dataset[index]
        caption = sep.join(captions)

        image, _, vfile = self.preprocess_image(index=index)
        return {
            "text": caption,
            "image": image, 
            "vfile": vfile,
            "num_obj": num_obj,
        }

class ClevrImageTextCollator:
    def __init__(self, device=torch.device("cpu"), vocab=None):
        self.device = device
        self.vocab = vocab

    def _naive_collator(self, union):
        text = np.stack(union["text"], axis=0)
        image = np.stack(union["image"], axis=0) 
        new_items = {
            "text": text,
            "image": image,
        }
        union.update(new_items)
        return union

    def __call__(self, records):
        union = { 
            k: [record.get(k) for record in records] for k in set().union(*records) 
        }
        return self._naive_collator(union)

def build_clevr_image_text_data(cfg, train, echo, ddp_mode=False):
    encoder_vocab = decoder_vocab = cate_vocab = None 
    dataloader = evalloader = testloader = None
    try:
        vis_vocab_file = f"{cfg.data_root}/{cfg.vis_vocab_name}"
        register_indexer(
            cfg.vis_vocab_name, vis_vocab_file, add_special=(len(cfg.vis_special_token) > 0),
            specials=cfg.vis_special_token, front_special=False
        )
        decoder_vocab = DatasetCatalog.get(cfg.vis_vocab_name) 
    except Exception as e:
        echo(f"Err: {e}")
        word_list = [f"{i}" for i in range(cfg.vis_vocab_size)]
        register_indexer(
            cfg.vis_vocab_name, None, add_special=(len(cfg.vis_special_token) > 0), override=True,
            extra_keys=word_list, specials=cfg.vis_special_token, front_special=False
        ) # may want to append a <BOS> w/o changing image token indice
        decoder_vocab = DatasetCatalog.get(cfg.vis_vocab_name) 

    extra_keys = [] if cfg.vis_only else [f"pad{i:03}" for i in range(cfg.max_txt_len)]
    try:
        txt_vocab_file = f"{cfg.data_root}/{cfg.txt_vocab_name}"
        special_words = list(cfg.txt_special_token.values())
        register_indexer(
            cfg.txt_vocab_name, txt_vocab_file,
            extra_keys=extra_keys, specials=cfg.txt_special_token, front_special=True
        )
        encoder_vocab = DatasetCatalog.get(cfg.txt_vocab_name) 
    except Exception as e:
        echo(f"Err: {e}")
        register_indexer(
            cfg.txt_vocab_name, None, override=True,
            extra_keys=extra_keys, specials=cfg.txt_special_token, front_special=True
        )
        encoder_vocab = DatasetCatalog.get(cfg.txt_vocab_name)
    

    def get_image_path(data_file):
        #data_file = os.path.basename(data_file)
        data_file = data_file.rsplit("/", 2)[-2:]
        if len(data_file) > 1:
            split, data_file, *_ = data_file
        else:
            split = ""
        if "train" in data_file or "train" in split:
            name = "/images/train/CLEVR_train_{:06}.png"
        elif "test" in data_file or "test" in split:
            name = "/images/test/CLEVR_test_{:06}.png"
        elif "val" in data_file or "val" in split:
            name = "/images/val/CLEVR_val_{:06}.png"
        else:
            raise ValueError(f"cannot figure out data split: {data_file}")
        return cfg.more_root + name 

    def get_sample_chunk(chunk):
        if isinstance(chunk, (tuple, list, ListConfig)): 
            return list(chunk)
        return None 

    pin_memory = False
    dataset_cls = ClevrImageData if cfg.vis_only else ClevrImageTextData
    dataset_cls = ClevrImageTextDataForDalleMini if getattr(cfg, "encode_for_dalle_mini", False) else dataset_cls
    dataset_cls = ClevrImageTextPandasForDalleMini if getattr(cfg, "use_preencoded_pandas", False) else dataset_cls

    # train
    name = cfg.data_name if train else cfg.eval_name
    ifile = f"{cfg.data_root}/{name}" # txt file
    skip = name is None or name == "" # not required when using only images
    assert (os.path.isfile(ifile) or cfg.vis_only or ifile.endswith(".parquet")) and not skip, f"not a data file {ifile}"
    image_path = get_image_path(ifile)
    chunk = cfg.train_samples if train else cfg.eval_samples
    chunk = get_sample_chunk(chunk)
    dataset = dataset_cls(
        cfg, ifile, image_path, encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab,
        cate_vocab=cate_vocab, train=train, chunk=chunk
    )
    dataloader = build_dataloader(
        cfg, dataset, train, ClevrImageTextCollator(), echo, msg=f"main", pin_memory=pin_memory, ddp_mode=ddp_mode
    )

    # eval
    skip = not train or cfg.eval_name is None or cfg.eval_name == ""
    ifile = f"{cfg.data_root}/{cfg.eval_name}" if train else "IGNORE_ME"
    if (os.path.isfile(ifile) or cfg.vis_only or ifile.endswith(".parquet")) and not skip:
        image_path = get_image_path(ifile)
        chunk = get_sample_chunk(cfg.eval_samples)
        dataset = dataset_cls(
            cfg, ifile, image_path, encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab,
            cate_vocab=cate_vocab, train=False, chunk=chunk
        )
        evalloader = build_dataloader(
            cfg, dataset, False, ClevrImageTextCollator(), echo, msg=f"eval", pin_memory=pin_memory, ddp_mode=ddp_mode
        )

    # test
    skip = not train or cfg.test_name is None or cfg.test_name == ""
    ifile = f"{cfg.data_root}/{cfg.test_name}" if train else "IGNORE_ME"
    if (os.path.isfile(ifile) or cfg.vis_only or ifile.endswith(".parquet")) and not skip:
        image_path = get_image_path(ifile)
        chunk = get_sample_chunk(cfg.test_samples)
        dataset = dataset_cls(
            cfg, ifile, image_path, encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab,
            cate_vocab=cate_vocab, train=False, chunk=chunk
        )
        testloader = build_dataloader(
            cfg, dataset, False, ClevrImageTextCollator(), echo, msg=f"test", pin_memory=pin_memory, ddp_mode=ddp_mode
        )
    return dataloader, evalloader, testloader, encoder_vocab, decoder_vocab, cate_vocab
