from typing import Callable, Dict, List, Optional, Tuple, Union
import torch.nn.functional as F
from torch import nn, Tensor
import torch

from ... import _get_activation_fn, _get_initializr_fn, MetaModule

__all__ = [
    "MiniTF", 
    "MiniTFBlock",
    "MiniTFAttention",
]

def _get_clones(module_fn, N):
    return nn.ModuleList([module_fn(i) for i in range(N)])

class MiniTF(nn.Module):
    """ A shell for both encoder and decoder.
    """
    def __init__(
        self, layer: nn.Module, N: int,
        iln: Callable = lambda x: x,
        oln: Callable = lambda x: x,
    ):
        super().__init__()
        self.layers = _get_clones(layer, N)
        self.iln = iln # norm-last
        self.oln = oln # norm-first

    def forward(
        self, x: Tensor, 
        kv: Tensor = None, 
        self_attn_mask: Tensor = None, 
        self_key_padding_mask: Tensor = None,
        memory: Tensor = None,
        memo_attn_mask: Tensor = None,
        memo_key_padding_mask: Tensor = None,
        require_attn_weight: bool = False,
        **kwargs
    ): 
        attn_weights = list()
        x = self.iln(x) # for norm-last setup 
        for ilayer, layer in enumerate(self.layers):
            x, attn_weight = layer(
                x, kv = kv, 
                self_attn_mask = self_attn_mask, 
                self_key_padding_mask = self_key_padding_mask,
                memory = memory, 
                memo_attn_mask = memo_attn_mask, 
                memo_key_padding_mask = memo_key_padding_mask,
                **kwargs
            )
            attn_weights.append(attn_weight)
        x = self.oln(x) # for norm-first setup 
        return x, (attn_weights if require_attn_weight else None)

class MiniTFBlock(MetaModule):
    """ Encoder or decoder, it is your choice.
    """
    def __init__(
        self, D: int, N: int, F: int, 
        attn_cls_intra, 
        attn_cls_inter: str = None,
        ilayer: int = 0,
        dropout: float = .0, 
        qk_scale: float = None,
        norm_first: bool = False,
        activation: str = "gelu",
        attn_dropout: float = .0,
        proj_dropout: float = .0,
        num_head_intra: int = None,
        num_head_inter: int = None,
        inter_layers: list = [],
        **kwargs,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.intra_attn = eval(attn_cls_intra)(
            D, num_head_intra or N, attn_dropout=attn_dropout, proj_dropout=proj_dropout, **kwargs
        ) 
        self.intra_attn_ln = nn.LayerNorm(D)
        self.intra_attn_dp = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(D, F),
            _get_activation_fn.get(activation, nn.GELU),
            nn.Dropout(dropout),
            nn.Linear(F, D), 
        )
        self.ff_ln = nn.LayerNorm(D) 
        self.ff_dp = nn.Dropout(dropout)

        do_inter = True if ilayer >= len(inter_layers) else inter_layers[ilayer]

        if do_inter and attn_cls_inter is not None:
            self.inter_attn = eval(attn_cls_inter)(
                D, num_head_inter or N, attn_dropout=attn_dropout, proj_dropout=proj_dropout, **kwargs
            )
            self.inter_attn_ln = nn.LayerNorm(D)
            self.inter_attn_dp = nn.Dropout(dropout) 
        else:
            self.register_parameter("inter_attn", None)
            self.register_parameter("inter_attn_ln", None)
            self.register_parameter("inter_attn_dp", None)
        self._reset_parameters()

    def _reset_parameters(self):
        pass

    def _intra_block(self, q, k, v, attn_mask, padding_mask, **kwargs):
        x, intra_attn_weight = self.intra_attn(
            q, k, v, 
            attn_mask = attn_mask, 
            key_padding_mask = padding_mask, 
            **kwargs
        ) 
        x = self.intra_attn_dp(x)
        return x, intra_attn_weight, None 

    def _inter_block(self, x, k, v, attn_mask, padding_mask, **kwargs):
        x, inter_attn_weight = self.inter_attn(
            q, k, v, 
            attn_mask = attn_mask,
            key_padding_mask = padding_mask, 
            **kwargs
        ) 
        x = self.inter_attn_dp(x)
        return x, inter_attn_weight, None

    def _ff_block(self, x, **kwargs):
        x = self.ff(x)
        x = self.ff_dp(x)
        return x, None

    def forward(
        self, q: Tensor,
        kv: Tensor = None,
        self_attn_mask: Tensor = None,
        self_key_padding_mask: Tensor = None,
        memory: Tensor = None,
        memo_attn_mask: Tensor = None,
        memo_key_padding_mask: Tensor = None,
        **kwargs
    ):
        if kv is None:
            k = v = self.intra_attn_ln(q) if self.norm_first else q
        else:
            k = v = kv # might have to be norm-ed outside 

        inter_attn_weight = None # could be None in encoder-only mode

        if self.norm_first:
            x_intra, intra_attn_weight, *_ = self._intra_block(
                self.intra_attn_ln(q), k, v, self_attn_mask, self_key_padding_mask, **kwargs
            )
            q = q + x_intra
            
            if self.inter_attn is not None:
                k = v = memory
                x_inter, inter_attn_weight, *_ = self._inter_block(
                    self.inter_attn_ln(q), k, v, memo_attn_mask, memo_key_padding_mask, **kwargs
                )
                q = q + x_inter

            x_ff, *_ = self._ff_block(
                self.ff_ln(q), **kwargs
            )
            q = q + x_ff
        else:
            x_intra, intra_attn_weight, *_ = self._intra_block(
                q, k, v, self_attn_mask, self_key_padding_mask, **kwargs
            )
            q = self.intra_attn_ln(q + x_intra)
            
            if self.inter_attn is not None:
                k = v = memory
                x_inter, inter_attn_weight, *_ = self._inter_block(
                    q, k, v, memo_attn_mask, memo_key_padding_mask, **kwargs
                )
                q = self.inter_attn_ln(q + x_inter)
            
            x_ff, *_ = self._ff_block(
                q, **kwargs
            )
            q = self.ff_ln(q + x_ff)
        return q, (intra_attn_weight, inter_attn_weight)

    def forward_old(
        self, q: Tensor,
        kv: Tensor = None,
        self_attn_mask: Tensor = None,
        self_key_padding_mask: Tensor = None,
        memory: Tensor = None,
        memo_attn_mask: Tensor = None,
        memo_key_padding_mask: Tensor = None,
        **kwargs
    ):
        if kv is None:
            k = v = q
        else:
            k = v = kv

        residual = q 
        x, intra_attn_weight = self.intra_attn(
            q, k, v, 
            attn_mask = self_attn_mask, 
            key_padding_mask = self_key_padding_mask, 
            **kwargs
        ) 
        x = self.intra_attn_ln(residual + self.intra_attn_dp(x))
        
        inter_attn_weight = None
        if self.inter_attn is not None:
            k = v = memory
            residual = q = x
            x, inter_attn_weight = self.inter_attn(
                q, k, v, 
                attn_mask = memo_attn_mask,
                key_padding_mask = memo_key_padding_mask, 
                **kwargs
            ) 
            x = self.inter_attn_ln(residual + self.inter_attn_dp(x))

        x = self.ff_ln(x + self.ff_dp(self.ff(x)))
        return x, (intra_attn_weight, inter_attn_weight)

class MiniTFAttention(MetaModule):
    """ Light-weight MHA for batch-first inputs.
    """
    def __init__(
        self, 
        D: int, 
        N: int, 
        kdim: int = None,
        vdim: int = None,
        bias: bool = True,
        qk_scale: float = None,
        attn_dropout: float = .0,
        proj_dropout: float = .0,
        **kwargs,
    ):
        super().__init__()
        assert D % N == 0
        self.D = D   
        self.N = N
        self.H = D // N 

        self.qk_scale = qk_scale or self.H ** -0.5
        
        self.kdim = kdim if kdim is not None else D
        self.vdim = vdim if vdim is not None else D

        if self.kdim == self.vdim == D:
            self.proj_weight = nn.Parameter(Tensor(3 * D, D))
            self.register_parameter("q_proj", None)
            self.register_parameter("k_proj", None)
            self.register_parameter("v_proj", None)
            num_bias = 3 * D
        elif self.kdim == self.vdim:
            self.proj_weight = nn.Parameter(Tensor(2 * D, kdim))
            self.q_proj = nn.Linear(D, 1 * D, bias=bias)
            self.register_parameter("k_proj", None)
            self.register_parameter("v_proj", None)
            num_bias = 2 * D
        else:
            self.register_parameter("proj_weight", None)
            self.q_proj = nn.Linear(D, 1 * D, bias=bias)
            self.k_proj = nn.Linear(kdim, 1 * D, bias=bias)
            self.v_proj = nn.Linear(vdim, 1 * D, bias=bias)
            num_bias = 0 * D

        if bias and num_bias > 0:
            self.proj_bias = nn.Parameter(Tensor(num_bias)) 
        else:
            self.register_parameter("proj_bias", None)

        self.proj = nn.Linear(D, D)
        self.proj_dp = nn.Dropout(proj_dropout)
        self.attn_dp = nn.Dropout(attn_dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.proj_weight is not None:
            nn.init.xavier_uniform_(self.proj_weight) 
        if self.proj_bias is not None:
            nn.init.constant_(self.proj_bias, 0.)

    def forward(
        self, 
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor = None,
        key_padding_mask: Tensor = None,
        **kwargs
    ):
        if q.data_ptr() == k.data_ptr() == v.data_ptr():
            k, v, q = self._proj_qkv(q)
        elif k.data_ptr() == v.data_ptr():
            k, v = self._proj_kv(k)
            q = self._proj_q(q) 
        else:
            q = self._proj_q(q)
            k = self._proj_k(k)
            v = self._proj_v(v)

        B, T, S = q.shape[0], q.shape[1], k.shape[1]
        
        # (B, L, D) -> (B, L, N, H) -> (B, N, L, H)
        q = q.contiguous().reshape(B, T, self.N, self.H).permute(0, 2, 1, 3)
        k = k.contiguous().reshape(B, S, self.N, self.H).permute(0, 2, 1, 3)
        v = v.contiguous().reshape(B, S, self.N, self.H).permute(0, 2, 1, 3)

        attn_weight = (q @ k.transpose(-1, -2)) * self.qk_scale # (B, N, T, S)
        
        if attn_mask is not None: 
            if attn_mask.dim() == 3: # (B, T, S) instance-specific 
                attn_mask = attn_mask.unsqueeze(1)
            elif attn_mask.dim() == 2: #  (T, S) shared within the batch
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_weight.masked_fill_(attn_mask, float('-inf'))

        if key_padding_mask is not None: # (B, T) instance-specific
            attn_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weight.masked_fill_(attn_mask, float('-inf'))

        attn_weight = self.attn_dp(attn_weight.softmax(dim=-1))
        x = (attn_weight @ v).transpose(1, 2).reshape(B, T, self.D)
        x = self.proj_dp(self.proj(x))
        return x, attn_weight

    def _proj_qkv(self, x):
        return self._in_proj(x).chunk(3, dim=-1)

    def _proj_kv(self, x):
        return self._in_proj(x, end=2 * self.D).chunk(2, dim=-1)

    def _proj_k(self, x):
        return (self._in_proj(x, end=self.D) 
            if self.k_proj is None else self.k_proj(x)
        )
    def _proj_v(self, x):
        return (self._in_proj(x, start=self.D, end=2 * self.D) 
            if self.v_proj is None else self.v_proj(x)
        )
    def _proj_q(self, x):
        return (self._in_proj(x, start=2 * self.D) 
            if self.q_proj is None else self.q_proj(x)
        )
    def _in_proj(self, x, start=0, end=None):
        weight = self.proj_weight[start : end]
        bias = (
            None if self.proj_bias is None else self.proj_bias[start : end]
        )
        return F.linear(x, weight, bias)
