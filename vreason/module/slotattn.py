import torch
from torch import nn, Tensor

from . import MetaModule, _get_activation_fn, _get_initializr_fn

__all__ = ["SlotAttnBlock", "SlotAttention"]

class SlotAttnBlock(MetaModule):
    def __init__(
        self, D: int, N: int, F: int, S: int,
        attn_cls_intra, 
        attn_cls_inter: str = None,
        ilayer: int = 0,
        qdim: int = None, # dim of slot
        kdim: int = None, # dim of memo
        vdim: int = None, # dim of memo
        dropout: float = .0, 
        activation: str = "gelu",
        num_head_intra: int = None,
        num_head_inter: int = None,
        inter_layers: list = [],
        epsilon: float = 1e-8,
        niter: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.S = S # num of slots

        self.niter = niter 
        self.epsilon = epsilon

        self.slot_attn = eval(attn_cls_intra)(
            D, num_head_intra or N, F, qdim=qdim, kdim=kdim, vdim=vdim, 
            activation=activation, **kwargs
        ) 

        self.slot_mu = nn.Parameter(torch.zeros(1, 1, D))
        self.slot_lsigma = nn.Parameter(torch.zeros(1, 1, D))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_lsigma)

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
        assert kv is not None, f"need contexts (kv)"
        B, *_ = kv.size()

        slots = kv.new_empty(B, self.S, self.slot_attn.D).normal_()
        slots = self.slot_mu + self.slot_lsigma.exp() * slots
        slots, inter_attn_weight = self.slot_attn(
            slots, kv, epsilon=self.epsilon, niter=self.niter
        )

        return slots, (None, inter_attn_weight)

class SlotAttention(MetaModule):
    def __init__(
        self,
        D: int, # dim of slots
        N: int, # num of heads 
        F: int, # hidden dim of mlp 
        qdim: int = None, # dim of slot
        kdim: int = None, # dim of memo
        vdim: int = None, # dim of memo
        bias: bool = True,
        qkv_bias: bool = False,
        qk_scale: float = None,
        attn_dropout: float = .0,
        proj_dropout: float = .0,
        activation: str = "gelu",
        dropout: float = .0,
        **kwargs,
    ):
        super().__init__()
        assert D % N == 0
        self.D = D   
        self.N = N
        self.H = D // N 

        self.qk_scale = qk_scale or self.H ** -0.5
        
        self.qdim = qdim if qdim is not None else D
        self.kdim = kdim if kdim is not None else D
        self.vdim = vdim if vdim is not None else D

        assert self.kdim == self.vdim

        self.q_proj = nn.Linear(self.qdim, 1 * D, bias=qkv_bias)
        self.k_proj = nn.Linear(self.kdim, 1 * D, bias=qkv_bias)
        self.v_proj = nn.Linear(self.vdim, 1 * D, bias=qkv_bias)

        self.slot_ln = nn.LayerNorm(self.qdim)
        self.memo_ln = nn.LayerNorm(self.kdim)
        
        self.gru = nn.GRUCell(D, D)

        self.mlp_ln = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(D, F),
            _get_activation_fn.get(activation, nn.GELU),
            nn.Linear(F, D), 
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for mod in [self.q_proj, self.k_proj, self.v_proj]:
            if hasattr(mod, "weight") and mod.weight is not None:
                nn.init.xavier_uniform_(mod.weight)
            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.zeros_(mod.bias)
        # GRU
        nn.init.xavier_uniform_(self.gru.weight_ih)
        nn.init.orthogonal_(self.gru.weight_hh)
        if self.gru.bias:
            nn.init.zeros_(self.gru.bias_ih)
            nn.init.zeros_(self.gru.bias_ih)

    def forward(
        self, q: Tensor,
        kv: Tensor = None,
        self_attn_mask: Tensor = None,
        self_key_padding_mask: Tensor = None,
        memory: Tensor = None,
        memo_attn_mask: Tensor = None,
        memo_key_padding_mask: Tensor = None,
        epsilon: float = 1e-8,
        niter: int = 3,
        **kwargs,
    ):
        assert kv is not None, f"require contexts (kv)"
        B, T, S = q.shape[0], q.shape[1], kv.shape[1]

        # q (slots) and kv (contexts)
        memo = self.memo_ln(kv)

        k = self.k_proj(memo).reshape(B, S, self.N, self.H).permute(0, 2, 1, 3)
        v = self.v_proj(memo).reshape(B, S, self.N, self.H).permute(0, 2, 1, 3)
        
        slots, attn = q, None
        for _ in range(niter):
            slots_old = slots 

            slots = self.slot_ln(slots)

            q = self.q_proj(slots).reshape(B, T, self.N, self.H).permute(0, 2, 1, 3)

            ## attention
            attn_weight = (k @ q.transpose(-1, -2)) * self.qk_scale # (B, N, S, T)
            
            attn_weight = (attn_weight
                .transpose(1, 2)
                .reshape(B, S, -1)
                .softmax(-1)
                .view(B, S, self.N, T)
                .transpose(1, 2)
            )
            attn = attn_weight.sum(1) # aggregate from heads
            
            ## summary
            attn_weight = attn_weight + epsilon 
            attn_weight = attn_weight / attn_weight.sum(2, keepdim=True)

            slots_new = (attn_weight.transpose(-1, -2) @ v).transpose(1, 2).reshape(B, T, -1) 

            ## update
            slots = self.gru(
                slots_new.view(-1, self.D), slots_old.view(-1, self.D) 
            )
            slots = slots.view(B, T, self.D)
            slots = slots + self.mlp(self.mlp_ln(slots))

        return slots, attn
