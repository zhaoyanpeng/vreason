import os
import math
import torch
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

import matplotlib.font_manager
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rcParams["savefig.bbox"] = 'tight'


def shorten_name(x, width=15):
    assert width > 2, f"name has at least two chars"
    if len(x) > width:
        return x[:width - 2] + ".."
    elif len(x) < width:
        return f"{x: <15}" 
    return x

def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)

def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

def bidirectional_causality(forward, reverse):
    forward = forward[:, :-1] # [1, L]
    reverse = reverse[:, 1: ] # [0, L - 1]
    B, _, H = forward.shape[:3] 
    pad = torch.zeros((B, 1, H), device=forward.device)
    forward = torch.cat([pad, forward], dim=1) # [0, L]
    reverse = torch.cat([reverse, pad], dim=1) # [0, L]
    final = torch.cat([forward, reverse], dim=-1)
    return final

def flip_and_shift(x, lengths, batch_first=True):
    """
    Flip a tensor along the time dim and shift all padded tokens to the end.
    """
    if not batch_first:
        x = x.transpose(0, 1)

    B, L = x.shape[:2]
    length_list = lengths.cpu().view(-1).tolist()
    max_xl = max(length_list)

    assert max_xl == x.shape[1] 

    x_reverse = x.flip(1)
    x_reverse = [
        x_reverse[b, max_xl - xl:] for b, xl in enumerate(length_list)
    ]
    x_reverse = pad_sequence(x_reverse, batch_first=True)
    
    assert (x_reverse[:, 1] == x[torch.arange(B, device=x.device), lengths - 2]).all()
    assert (x[:, 1] == x_reverse[torch.arange(B, device=x.device), lengths - 2]).all()

    if not batch_first:
        return x_reverse.transpose(0, 1)
    return x_reverse

def stripe(x, n, w, offset=(0, 0), dim=1):
    assert dim in {0, 1}, f"strip along row (1) or column (0)."
    x = x.contiguous()
    b, H, W = x.shape[:3]
    stride = list(x.stride())
    numel = stride[2]
    stride[1] = (W + 1) * numel
    stride[2] = (1 if dim == 1 else W) * numel
    offset = (offset[0] * W + offset[1]) * numel
    size = (b, n, w) if x.dim() <= 3 else (b, n, w, *list(x.shape[3:]))
    return x.as_strided(
        size=size, stride=stride, storage_offset=offset
    )

def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def save_image_local(
    root, names, prompts, images, labels=None, nrow=-1, ncol=5, dpi=-1, margin=2,
    color=(255,) * 3, mode="RGB", use_plt=False, font=None, font_size=12,
):
    assert len(names) == len(images), f"require one2one mapping between names and images."
    assert isinstance(images, np.ndarray) and images.ndim == 5, f"require 4D numpy images (B, W, H, C, N)"
    
    B, W, H, C, N = images.shape
    ncol = max(5, ncol)
    nrow = int(np.ceil(N / ncol))

    if not use_plt:
        font = font or "Lato-Thin.ttf"
        assert True or os.path.isfile(font), f"please specify a font file."
        # https://stackoverflow.com/a/27564040
        font = matplotlib.font_manager.FontProperties(family='lato:thin')
        font = matplotlib.font_manager.findfont(font)
        font = ImageFont.truetype(font, font_size)
        
        text_line_width = 50
        text_beg_height = 0.75

        ww = W + margin
        hh = H + margin
        pw = ww * ncol - margin
        ph = hh * nrow - margin
        for i, (name, prompt) in enumerate(zip(names, prompts)):
            fname = f"{root}/{name}.png"
            image_set = images[i].transpose(3, 0, 1, 2)

            panel = Image.new(mode, (pw, ph), color=color)
            draw = ImageDraw.Draw(panel)

            for i, image in enumerate(image_set):
                irow = i // ncol
                icol = i %  ncol
                image = Image.fromarray(image)
                x, y = (icol * ww, irow * hh)
                panel.paste(image, (x, y))
                if labels is not None: # each column has a unique label
                    draw.text((x + 2 * margin, y + 2 * margin), labels[i], font=font)
            # draw text
            lines = textwrap.wrap(prompt, width=text_line_width)
            text_h = H * text_beg_height
            for line in lines:
                lw, lh = font.getsize(line)
                draw.text(((W - lw) / 2, text_h), line, font=font)
                text_h += lh
            panel.save(fname)
    else:
        dpi = max(dpi, H // 2)
        text_line_width = 120
        for i, (name, prompt) in enumerate(zip(names, prompts)):
            fname = f"{root}/{name}.png"
            image_set = images[i].transpose(3, 0, 1, 2)

            fig = plt.figure(constrained_layout=True, dpi=dpi)
            gs = gridspec.GridSpec(nrow, ncol, figure=fig, wspace=5e-3, hspace=5e-3)
            #fig_width, fig_height = fig.get_size_inches() * fig.dpi

            for i, image in enumerate(image_set):
                irow = i // ncol
                icol = i %  ncol
                ax = fig.add_subplot(gs[irow, icol])
                ax.imshow(image)
                ax.axis('off')
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            # draw text
            lines = textwrap.wrap(prompt, width=text_line_width)
            #fig.suptitle("\n".join(lines), fontsize=4) #font_size)

            #plt.plot() 
            plt.savefig(fname) 
            plt.clf()
