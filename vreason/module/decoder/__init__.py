from .decoder_head import build_decoder_head, DECODER_HEADS_REGISTRY
from .gpt_head import GPTDecHead, NeoGPTDecHead

DECODER_HEADS_REGISTRY.register(GPTDecHead)
DECODER_HEADS_REGISTRY.register(NeoGPTDecHead)
