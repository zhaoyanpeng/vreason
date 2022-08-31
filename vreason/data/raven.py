import torch
import numpy as np
from omegaconf.listconfig import ListConfig

try:
    from taming.data.custom import RavenBase, RavenData
    from taming.data.raven import RavenVQGAN
except Exception as e:
    RavenBase = RavenData = RavenVQGAN = object 

from . import DatasetCatalog, register_indexer
from ..util import shorten_name, get_world_size

class RavenVQEncData(RavenVQGAN):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        super().__init__(paths, size=size, random_crop=random_crop, labels=labels)
        self._length = self._length // self.scale_factor
        
    def preprocess_image(self, npz_data, image_idx):
        image = npz_data[image_idx]
        image = np.stack([image] * 3, axis=-1)

        image = self.preprocessor(image=image)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        file_idx = image_idx = i
        image_path = self.labels["file_path_"][file_idx]
        npz_data = np.load(image_path)
        img_data = npz_data["image"]
        
        images = []
        for i_img in range(img_data.shape[0]):
            image = self.preprocess_image(img_data, i_img)
            images.append(image.transpose(2, 0, 1))
        images = np.stack(images)
        
        example = dict()
        example["image"] = images
        example["target"] = npz_data["target"]
        for k in self.labels:
            example[k] = self.labels[k][file_idx]
        return example

class RavenGPTData(RavenVQGAN):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        super().__init__(paths, size=size, random_crop=random_crop, labels=labels)
        self._length = self._length // self.scale_factor

    def __getitem__(self, i):
        file_idx = image_idx = i
        image_path = self.labels["file_path_"][file_idx]
        npz_data = np.load(image_path)
        img_data = npz_data["indice"]

        context = img_data[:8]
        idx_tgt = npz_data["target"]
        positive = img_data[idx_tgt + 8][None]
        sli = [i + 8 for i in range(8) if i != idx_tgt]
        negative = img_data[sli]

        example = dict()
        example["target"] = idx_tgt
        example["context"] = context
        example["positive"] = positive
        example["negative"] = negative
        #example["indice"] = npz_data["indice"]
        for k in self.labels:
            example[k] = self.labels[k][file_idx]
        return example

def build_raven_for_vqgan_encoder(cfg, root, splits, tasks, nsample, echo):
    dataset = RavenData(
        root, splits, tasks, cfg.resolution, dataset_cls=RavenVQEncData, nsample=nsample
    )
    echo(f"Load {len(dataset)} problems from {splits} of {tasks}.")
    return dataset

class SeqRavenCollator:
    def __init__(self, device=torch.device("cpu"), vocab=None):
        self.device = device
        self.vocab = vocab

    def _naive_collator(self, union):
        context = np.stack(union["context"], axis=0) 
        positive = np.stack(union["positive"], axis=0)
        negative = np.stack(union["negative"], axis=0)
        return {
            "context": context,
            "positive": positive,
            "negative": negative,
        }

    def __call__(self, records):
        union = { 
            k: [record.get(k) for record in records] for k in set().union(*records) 
        }
        return self._naive_collator(union)

def build_dataloader(cfg, dataset, train, collate_fn, echo, msg="", pin_memory=True, ddp_mode=False):
    if ddp_mode:
        world_size = get_world_size()
        assert cfg.batch_size % world_size == 0, (
            f"batch size ({cfg.batch_size}) cannot be divided by # device ({world_size})."
        )
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=train)
        per_device_batch_size = cfg.batch_size // world_size
    else:
        sampler = (torch.utils.data.RandomSampler(dataset)
            if train else torch.utils.data.SequentialSampler(dataset)
        )
        per_device_batch_size = cfg.batch_size
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, 
        batch_size=per_device_batch_size,
        num_workers=cfg.num_proc,
        shuffle=False,
        sampler=sampler,
        pin_memory=pin_memory, 
        collate_fn=collate_fn,
        drop_last=(True if ddp_mode else False)
    )
    echo(f"Load {len(data_loader)} ({len(dataset)}) batches ({msg}).")
    return data_loader

def build_raven_for_gpt(cfg, train, echo):
    root = f"{cfg.data_root}/{cfg.name}"
    dataloader = evalloader = testloader = encoder_vocab = None
    
    special_words = list(cfg.special_token.values())
    word_list = [f"{i}" for i in range(cfg.vq_vocab_size)] + special_words
    register_indexer(
        cfg.name, None, extra_keys=word_list, 
        specials=cfg.special_token, front_special=False
    )
    encoder_vocab = decoder_vocab = DatasetCatalog.get(cfg.name) 

    def _build_raven_per_task(splits, nsample, train, msg):
        dl_dict = {} 
        for task in list(cfg.tasks):
            dataset = RavenData(
                root, splits, [task], None, dataset_cls=RavenGPTData, nsample=int(nsample)
            )
            name = shorten_name(task) 
            dataloader = build_dataloader(
                cfg, dataset, train, SeqRavenCollator(), echo, msg=f"{msg}: {name}"
            )
            dl_dict[task] = dataloader
        return dl_dict

    # train
    splits = ["train", "val"] if train else ["test"] #["val"] # 
    nsample = cfg.train_samples if train else cfg.test_samples
    if train:
        dataset = RavenData(
            root, splits, cfg.tasks, None, dataset_cls=RavenGPTData, nsample=int(nsample)
        )
        dataloader = build_dataloader(cfg, dataset, train, SeqRavenCollator(), echo, msg=f"main ({splits})")
    else:
        dataloader = _build_raven_per_task(splits, nsample, train, f"test ({splits})")
    
    # eval
    if train and cfg.eval_samples > 0:
        evalloader = _build_raven_per_task(["val"], cfg.eval_samples, False, f"eval ([val])")

    # test
    if train and cfg.test_samples > 0:
        evalloader = _build_raven_per_task(["test"], cfg.test_samples, False, f"test ([test])")

    return dataloader, evalloader, testloader, encoder_vocab, decoder_vocab
