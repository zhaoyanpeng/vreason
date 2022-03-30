import torch
from torch.nn.utils.rnn import pad_sequence

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
