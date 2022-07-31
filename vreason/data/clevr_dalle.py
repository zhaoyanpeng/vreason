import os
import json
import torch
import numpy as np
import itertools
from PIL import Image
import albumentations
from omegaconf.listconfig import ListConfig

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode 

from . import DatasetCatalog, register_indexer, build_dataloader
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
        self.num_txt_sample = cfg.num_txt_sample

        # load (a subset of) data
        self.dataset = list()
        keyset = set(list(range(chunk[0], chunk[1]))) if chunk is not None else None
        txt_data = json.load(open(data_file, "r"))["captions"]
        iskip = jskip = 0
        if keyset is not None:
            for key in keyset:
                key = str(key)
                if key in txt_data:
                    self.dataset.append((int(key), txt_data[key])) 
                else:
                    iskip += 1
        else:
            self.dataset = [(int(key), captions) for key, captions in txt_data.items()]
        #print(iskip, jskip, len(scenes), len(keyset) if keyset is not None else 0)
        
        # load only image indice
        self.length = len(self.dataset)
        self.version = cfg.version

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
        num_caption = len(captions)
        num_txt_sample = num_caption if num_caption < self.num_txt_sample else self.num_txt_sample
        if self.train: # sample a number & sample a number of text
            num = 1 if num_txt_sample == 1 else np.random.choice(range(1, num_txt_sample + 1), 1)[0]
            caption_indice = np.random.choice(num_caption, num, replace=False)
        else: # use a single caption at test time
            caption_indice = [0]

        selected = " ".join([captions[i] for i in caption_indice])

        caption = [self.encoder_vocab.BOS] + selected.lower().split(" ")
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

class ClevrImageTextCollator:
    def __init__(self, device=torch.device("cpu"), vocab=None):
        self.device = device
        self.vocab = vocab

    def _naive_collator(self, union):
        text = np.stack(union["text"], axis=0)
        image = np.stack(union["image"], axis=0) 
        return {
            "text": text,
            "image": image,
            "vfile": union["vfile"],
        }

    def __call__(self, records):
        union = { 
            k: [record.get(k) for record in records] for k in set().union(*records) 
        }
        return self._naive_collator(union)

def build_clevr_image_text_data(cfg, train, echo):
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
        data_file = os.path.basename(data_file)
        if "train" in data_file:
            name = "/images/train/CLEVR_train_{:06}.png"
        elif "test" in data_file:
            name = "/images/test/CLEVR_test_{:06}.png"
        elif "val" in data_file:
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

    # train
    name = cfg.data_name if train else cfg.eval_name
    ifile = f"{cfg.data_root}/{name}" # txt file
    skip = name is None or name == "" # not required when using only images
    assert (os.path.isfile(ifile) or cfg.vis_only) and not skip, f"not a data file {ifile}"
    image_path = get_image_path(ifile)
    chunk = cfg.train_samples if train else cfg.eval_samples
    chunk = get_sample_chunk(chunk)
    dataset = dataset_cls(
        cfg, ifile, image_path, encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab,
        cate_vocab=cate_vocab, train=train, chunk=chunk
    )
    dataloader = build_dataloader(
        cfg, dataset, train, ClevrImageTextCollator(), echo, msg=f"main", pin_memory=pin_memory
    )

    # eval
    skip = not train or cfg.eval_name is None or cfg.eval_name == ""
    ifile = f"{cfg.data_root}/{cfg.eval_name}" if train else "IGNORE_ME"
    if (os.path.isfile(ifile) or cfg.vis_only) and not skip:
        image_path = get_image_path(ifile)
        chunk = get_sample_chunk(cfg.eval_samples)
        dataset = dataset_cls(
            cfg, ifile, image_path, encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab,
            cate_vocab=cate_vocab, train=False, chunk=chunk
        )
        evalloader = build_dataloader(
            cfg, dataset, False, ClevrImageTextCollator(), echo, msg=f"eval", pin_memory=pin_memory
        )

    # test
    skip = not train or cfg.test_name is None or cfg.test_name == ""
    ifile = f"{cfg.data_root}/{cfg.test_name}" if train else "IGNORE_ME"
    if (os.path.isfile(ifile) or cfg.vis_only) and not skip:
        image_path = get_image_path(ifile)
        chunk = get_sample_chunk(cfg.test_samples)
        dataset = dataset_cls(
            cfg, ifile, image_path, encoder_vocab=encoder_vocab, decoder_vocab=decoder_vocab,
            cate_vocab=cate_vocab, train=False, chunk=chunk
        )
        testloader = build_dataloader(
            cfg, dataset, False, ClevrImageTextCollator(), echo, msg=f"test", pin_memory=pin_memory
        )
    return dataloader, evalloader, testloader, encoder_vocab, decoder_vocab, cate_vocab
