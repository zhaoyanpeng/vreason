import os
import PIL
import math
import torch
import random
import json, copy
import numpy as np
import itertools
from random import randint
from collections import defaultdict
from torch import nn
from nltk import word_tokenize
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

__all__ = [
    "read_layouts", 
    "load_artifacts", 
    "load_image_from_class", 
    "get_name_box_mask_art",
    "get_object_masks",
]

def read_layouts(ifile):
    layouts = defaultdict(list)
    fr = open(ifile)
    n = int(next(fr).strip())
    while True:
        line = next(fr, None)
        if not line:
            break
        idx, k = list(map(int, line.strip().split()))
        
        arts = []
        for _ in range(k):
            line = next(fr)
            cols = line.strip().split()
            art = cols[:1] + list(map(int, cols[1:]))
            arts.append(art)
        layouts[idx].append(arts)
    return layouts

def create_mask(image, color=[[[255, 0, 0, 255]]]):
    img_np = np.array(image)
    
    bmap = np.all(img_np[..., -1:] != [0], axis=-1)
    mask_np = ((bmap)[..., None] * color).astype(np.uint8)
    contrast = ((~bmap)[..., None] * [[[255] * 4]]).astype(np.uint8)
    mask_np = mask_np + contrast
    
    image = Image.fromarray(mask_np)
    
    img_np[~bmap] = [255, 255, 255, 0]
    old_img = Image.fromarray(img_np)
    return old_img, image

def load_artifacts(droot, seed=1213):
    # read all objects
    arts = dict()
    for root, dir, files in os.walk(f"{droot}/Pngs"):
        for fname in files:
            image = Image.open(f"{root}/{fname}").convert('RGBA')
            arts[fname.split(".")[0]] = image
    
    # create segmentations
    random.seed(seed)
    colors = [[[[randint(0, 255), randint(0, 255), randint(0, 255), 255]]] for _ in range(len(arts))]
    
    components = dict()
    for i, (k, art) in enumerate(arts.items()):
        old_art, mask_art = create_mask(art, color=colors[i]) #[[[255,127,80,255]]])
        components[k] = (old_art, mask_art)
    
    # read all layouts
    ifile = f"{droot}/Scenes_10020.txt"
    scene_layouts = read_layouts(ifile)

    return components, scene_layouts

def load_image_from_class(scene_layouts, vroot, vclass, iimage):
    vname = f"Scene{vclass}_{iimage}.png" 
    vfile = f"{vroot}/{vname}"
    image = Image.open(vfile)

    confs = scene_layouts[vclass][iimage]
    priorities = [(i, 10) if conf[1] == 0 else (i, conf[5]) for i, conf in enumerate(confs)]
    priorities = sorted(priorities, key=lambda x: -x[1])
    confs = [confs[i] for i, _ in priorities]
    return image, confs, vfile

def get_name_box_mask_art(confs, art_mask_dict, fn=math.ceil):
    scales = [1.0, 0.7, 0.49]
    names, boxes, masks, arts = [], [], [], []
    for cf in confs:
        name = cf[0].split(".")[0]
        x, y, z, flip = (int(cf[3]), int(cf[4]), int(cf[5]), int(cf[6]))
        
        art, mask = art_mask_dict[name]
        
        w, h = art.size
        scale = scales[z]
        w, h = w * scale, h * scale

        x = x - fn(w * 0.5)
        y = y - fn(h * 0.5)
        
        if flip:
            mask = mask.transpose(PIL.Image.FLIP_LEFT_RIGHT)
            art = art.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        if scale != 1.0:
            mask = mask.resize((fn(w), fn(h)))
            art = art.resize((fn(w), fn(h)))
        masks.append(mask)
        arts.append(art)
        
        box = (x, y, fn(w), fn(h))
        names.append(name)
        boxes.append(box)
    return names, boxes, masks, arts

def get_object_masks(inputs, arts, outs, bboxes, titles=[], fsize=22, n_per_line=4, margin=2, color=0, mode="RGB"):
    """ inputs: an image in a list; outs: a list of masks; bboxes: a list of bounding boxes.
    """
    n = len(inputs)
    assert n == 1, f"expect a single input image"
    w, h = inputs[0].size[:2]
    image = inputs[0]
    image_np = np.array(image)
    
    nline = int(np.ceil(n / n_per_line))
    
    k = 1
    
    ww = w + margin
    hh = h + margin
    
    # the global image, color is supposed to be (0, 0, 0), i.e., a black image w/ (0, 0, 0, 255) for png
    img = Image.new(mode, (n_per_line * ww - margin, k * hh * nline - margin), color=color)
    
    empty_img_np = np.array(img)
    
    masks = list()
    
    img_objs = list()
    for i, (art, a, b) in enumerate(zip(arts, outs, bboxes)):
        x, y, w, h = b 
        
        # the canvas for each object
        img_obj = Image.new(mode, (n_per_line * ww - margin, k * hh * nline - margin), color=color)
        
        # crop from the global image and find the region w/ black pixels
        crop = img.crop((x, y, x + w, y + h))
        crop_np = np.array(crop)
        crop_obj_mask = np.all(crop_np[..., 0:] == list(color) + [255], axis=-1)[..., None]
        
        # crop from the local image and will be used as mask
        crop_black = img_obj.crop((x, y, x + w, y + h))
        crop_np_black = np.array(crop_black)
        
        # get the real image region and set the others to (0, 0, 0, 0)
        mask_np = np.array(a)
        mask = ~np.all(mask_np[..., 0:] == [255] * 4, axis=-1)[..., None]
        
        # some part of the object region may have been occupied by other objects and will be set to (0, 0, 0, 0)
        mask_ = mask * crop_obj_mask
        
        # will be pasted to the global image: 
        # the overlapped region is ignored in the back object and is kept in the front image
        mask_np_ = mask_np * mask_ + crop_np * (~mask_)
        
        # will be pasted to the local image:
        # the overlapped region is totally ignored in the object
        art_np = np.array(art)
        mask_np = art_np * mask_ + crop_np_black * (~mask_)
        
        # build off the global image from the last iteration
        img_mask = Image.fromarray(mask_np_)
        img.paste(img_mask, (x, y))
        
        # a single object w/o overlapped regions
        img_mask = Image.fromarray(mask_np)
        img_obj.paste(img_mask, (x, y))
        
        # collect me!
        img_objs.append(img_obj)
                
        # collect binary masks
        obj_np = np.array(img_obj)
        mask = np.all(obj_np[..., 0:] == list(color) + [255], axis=-1)
        masks.append((~mask).astype(np.uint8))
        
        if i > -1:
            pass #break
        
    # background image
    img_bg = Image.new(mode, (n_per_line * ww - margin, k * hh * nline - margin), color=color)
    
    img_bg_np = np.array(img)
    img_bg_mask = np.all(img_bg_np[..., 0:] == list(color) + [255], axis=-1)[..., None]
    
    img_np = image_np * img_bg_mask + np.array(img_bg) * (~img_bg_mask)
    img_bg = Image.fromarray(img_np, mode)
    
    img_objs = list(reversed(img_objs))
    img_objs.insert(0, img_bg)
    
    # collect binary masks
    img_np = np.array(img_bg)
    mask = np.all(img_np[..., 0:] == list(color) + [255], axis=-1)
    
    masks = list(reversed(masks))
    masks.insert(0, (~mask).astype(np.uint8))
    
    return img, img_objs, masks
