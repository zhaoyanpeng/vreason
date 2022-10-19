import re
import warnings
import numpy as np

__all__ = ["convex_hull", "extract_bbox", "create_2obj_image", "create_2obj_fixed"]

def convex_hull(boxes, f=16, d=0.25, s=0.8, return_boxes=False, rotate=False):
    # rescale
    xx = np.array(boxes, dtype=np.float32)
    xx[:, 2] += xx[:, 0]
    xx[:, 3] += xx[:, 1]
    xx *= s
    
    # threshold
    min_v = f * d
    max_v = f * (1 - d)
    c = xx // f
    b = xx % f
    
    # adjust hull
    x1 = np.clip(f - b[:, 0], 0, f) < min_v # + 1
    y1 = np.clip(f - b[:, 1], 0, f) < min_v # + 1
    x2 = b[:, 2] > min_v # + 1
    y2 = b[:, 3] > min_v # + 1

    indice = np.stack([x1, y1, x2, y2], axis=-1)
    c[indice] += 1

    if rotate:
        c_new = np.zeros_like(c)
        c_new[:, 0] = c[:, 1]
        c_new[:, 2] = c[:, 3]
        c_new[:, 1] = f - c[:, 2]
        c_new[:, 3] = f - c[:, 0]
        c = c_new
    
    if return_boxes:
        c[:, 2] -= c[:, 0]
        c[:, 3] -= c[:, 1]
        c = c * f / s
    return c

def extract_bbox(all_df, scene_data, d=0.25, rotate=False):
    bbox_dict = dict()
    for _, row in all_df.iterrows():
        try:
            #x = row["image"]
            #x = re.match(".*?_(\d+)\.png", x)
            #x = int(x.groups()[0])
            x = row.vid
        except Exception as e:
            warnings.warn(f"{irow}\n{row}\n{e}")
            continue
        vidx = str(x)
        nobj = row.object_num
        bbox = [None] * nobj
        for obj in scene_data[vidx]["objects"]:
            bbox[obj["id"]] = obj["bbox"]
        new_bbox = convex_hull(bbox, d=d, rotate=rotate)
        bbox_dict[row.image.strip()] = new_bbox.astype(int)
    return bbox_dict

def create_2obj_image(size, obj_hw, ids, rng, *args):
    H, W = size
    bbox = list()
    
    o_h, o_w = obj_hw[0]
    if len(obj_hw) > 1:
        o_h_new, o_w_new = obj_hw[1]
        early_break = False
    else:
        early_break = True
    
    valid_y = H - o_h + 1
    valid_x = W - o_w + 1
    
    valid_x_new = list()
    cnt, max_num_try = 0, H * W
    while True: # make sure there is space for the other object
        cnt += 1
        convas = np.zeros(size)
        
        rnd_x = rng.choice(valid_x, 1)[0].item()
        rnd_y = rng.choice(valid_y, 1)[0].item()

        box = (rnd_x, rnd_y, rnd_x + o_w, rnd_y + o_h)
        convas[rnd_y : rnd_y + o_h, rnd_x : rnd_x + o_w] = ids[0]

        if early_break:
            return convas.astype(int), [box]

        if box[1] >= o_h_new or H - box[3] >= o_h_new:
            valid_x_new = list(range(W - o_w_new + 1))
        else:
            valid_x_new = (
                list(range(0, box[0] - o_w_new + 1)) +
                list(range(box[2], W - o_w_new + 1))
            )
        if cnt >= max_num_try:
#                 print(f"No valid outputs: {box}")
            return [[]] * 2
        if len(valid_x_new) > 0:
            break

    cnt = 0
    while True:
        cnt += 1
        rnd_x = rng.choice(valid_x_new, 1)[0].item()
        if rnd_x <= box[0] - o_w_new or rnd_x >= box[2]:
            valid_y = list(range(H - o_h_new + 1))
        else:
            valid_y = (
                list(range(0, box[1] - o_h_new + 1)) +
                list(range(box[3], H - o_h_new + 1))
            )
        if cnt >= max_num_try:
#                 print(f"No valid outputs: {box} {rnd_x}")
            return [[]] * 2
        if len(valid_y) == 0:
            continue
            
        rnd_y = rng.choice(valid_y, 1)[0].item()
        box2 = (rnd_x, rnd_y, rnd_x + o_w_new, rnd_y + o_h_new)
        convas[rnd_y : rnd_y + o_h_new, rnd_x : rnd_x + o_w_new] = ids[1]
        break
    return convas.astype(int), [box, box2]

def create_2obj_fixed(size, obj_hw, ids, rng, delta):
    H, W = size
    bbox = list()
    o_h, o_w = obj_hw[0]
    if len(obj_hw) > 1:
        o_h_new, o_w_new = obj_hw[1]
        early_break = False
    else:
        early_break = True
    
    cnt, max_num_try = 0, int(H * W * 1.5)
    while True: # make sure there is space for the other object
        cnt += 1
        if cnt > max_num_try:
#                 print(f"No valid outputs: {box}")
            return [[]] * 2
        convas = np.zeros(size)

        valid_y = H - o_h + 1
        valid_x = W - o_w + 1
        
        rnd_x = rng.choice(valid_x, 1)[0].item()
        rnd_y = rng.choice(valid_y, 1)[0].item()

        box = (rnd_x, rnd_y, rnd_x + o_w, rnd_y + o_h)
        convas[rnd_y : rnd_y + o_h, rnd_x : rnd_x + o_w] = ids[0]

        if early_break:
            return convas.astype(int), [box]
        
        # is new x valid?
        rnd_x = rnd_x + delta[1]
        if box[1] >= o_h_new or H - box[3] >= o_h_new:
            valid_x_new = list(range(W - o_w_new + 1))
        else:
            valid_x_new = (
                list(range(0, box[0] - o_w_new + 1)) +
                list(range(box[2], W - o_w_new + 1))
            )
        if rnd_x not in valid_x_new:
            continue
            
        # is new y valid?
        rnd_y = rnd_y + delta[0]
        if rnd_x <= box[0] - o_w_new or rnd_x >= box[2]:
            valid_y = list(range(H - o_h_new + 1))
        else:
            valid_y = (
                list(range(0, box[1] - o_h_new + 1)) +
                list(range(box[3], H - o_h_new + 1))
            )
        if rnd_y not in valid_y:
            continue
    
        box2 = (rnd_x, rnd_y, rnd_x + o_w_new, rnd_y + o_h_new)
        convas[rnd_y : rnd_y + o_h_new, rnd_x : rnd_x + o_w_new] = ids[1]
        break
    return convas.astype(int), [box, box2]
