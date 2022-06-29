import torch
from torch import nn

__all__ = ["_get_activation_fn", "_get_initializr_fn", "MetaModule"]

_get_activation_fn = {
    "elu": nn.ELU(),
    "relu": nn.ReLU(),
    "gelu": nn.GELU(),
    "celu": nn.CELU(),
    "tanh": nn.Tanh(),
    "lelu": nn.LeakyReLU(),
    "none": nn.Identity(),
}

_get_initializr_fn = {
    "norm": nn.init.normal_,
    "xavier": nn.init.xavier_uniform_,
}

class MetaModule(nn.Module):
    """ A nicer __repr__.
    """
    def __init__(self):
        super().__init__()

    def extra_repr(self):
        mod_keys = self._modules.keys()
        all_keys = self._parameters.keys()
        extra_keys = all_keys - mod_keys
        extra_keys = [k for k in all_keys if k in extra_keys]
        extra_lines = []
        for key in extra_keys:
            attr = getattr(self, key)
            if not isinstance(attr, nn.Parameter):
                continue
            extra_lines.append("({}): Tensor{}".format(key, tuple(attr.size())))
        return "\n".join(extra_lines)
