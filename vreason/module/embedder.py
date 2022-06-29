import torch
from torch import nn
import numpy as np

__all__ = ["PartiallyFixedEmbedding", "SoftPositionalEncoder"]

class PartiallyFixedEmbedding(torch.nn.Module):
    def __init__(self, vocab, fixed_weight, word_dim=-1, out_dim=-1):
        super(PartiallyFixedEmbedding, self).__init__()
        nword = len(vocab)
        self.n_fixed, vector_size = fixed_weight.shape
        n_tuned = nword - self.n_fixed

        weight = torch.empty(nword, vector_size)
        weight[:self.n_fixed] = fixed_weight

        self.tuned_weight = torch.nn.Parameter(torch.empty(n_tuned, vector_size)) 
        torch.nn.init.kaiming_uniform_(self.tuned_weight)
        weight[self.n_fixed:] = self.tuned_weight
        self.register_buffer("weight", weight)
         
        add_dim = word_dim - vector_size if word_dim > vector_size else 0 
        self.tuned_vector = torch.nn.Parameter(torch.empty(nword, add_dim))
        if add_dim > 0: 
            torch.nn.init.kaiming_uniform_(self.tuned_vector)
        in_dim = vector_size if add_dim == 0 else word_dim 

        self.linear = (
            torch.nn.Linear(in_dim, out_dim, bias=False)
            if out_dim > 0 else torch.nn.Identity() #None #lambda x: x
        )
        self._output_size = out_dim if out_dim > 0 else in_dim

    @property
    def output_size(self):
        return self._output_size

    def extra_repr(self):
        mod_keys = self._modules.keys()
        all_keys = self._parameters.keys()
        extra_keys = all_keys - mod_keys
        extra_keys = [k for k in all_keys if k in extra_keys]
        extra_lines = []
        for key in extra_keys:
            attr = getattr(self, key)
            if not isinstance(attr, torch.nn.Parameter):
                continue
            extra_lines.append("({}): Tensor{}".format(key, tuple(attr.size())))
        return "\n".join(extra_lines)

    def extra_repr(self):
        mod_keys = self._modules.keys()
        all_keys = self._parameters.keys()
        extra_keys = all_keys - mod_keys
        extra_keys = [k for k in all_keys if k in extra_keys]
        extra_lines = []
        for key in extra_keys:
            attr = getattr(self, key)
            if not isinstance(attr, torch.nn.Parameter):
                continue
            extra_lines.append("({}): Tensor{}".format(key, tuple(attr.size())))
        return "\n".join(extra_lines)

    def full_weight(self):
        self.weight.detach_()
        self.weight[self.n_fixed:] = self.tuned_weight
        weight = torch.cat([self.weight, self.tuned_vector], -1)
        return weight

    def bmm(self, X):
        x_shape = X.shape
        weight = self.full_weight()
        w_logit = torch.matmul(
            X.view(-1, x_shape[-1]), weight.transpose(0, 1)
        )
        w_logit = w_logit.view(x_shape[:-1] + (w_logit.size(-1),))
        return w_logit 

    def forward(self, X):
        if X.dtype != torch.int64:  
            return self.bmm(X) # w/o linear 
        weight = self.full_weight()
        word_emb = torch.nn.functional.embedding(X, weight, None, None, 2.0, False, False)
        word_emb = self.linear(word_emb)
        return word_emb


class SoftPositionalEncoder(nn.Module):
    def __init__(self, output_size, resolution, input_size=4, bias=True):
        super().__init__()
        grid = self.build_grid(resolution)
        self.register_buffer("grid", grid)
        self.linear = nn.Linear(input_size, output_size, bias=bias)
    
    @staticmethod
    def build_grid(resolution):
        size = [np.linspace(0., 1., num=k) for k in resolution]
        grid = np.meshgrid(*size, sparse=False, indexing="ij")
        grid = np.stack(grid, axis=-1)
        grid = np.reshape(grid, [resolution[0], resolution[1], -1])
        grid = grid[None].astype(np.float32)
        grid = np.concatenate([grid, 1.0 - grid], axis=-1)
        return torch.from_numpy(grid) 
    
    def forward(self, x):
        return x + self.linear(self.grid)

    def encode(self, x):
        return self.linear(self.grid)
