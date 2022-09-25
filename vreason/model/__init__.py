from .helper import * 
from .vq_encoder import VQGANEncoder
from .raven_solver import MetaSolver, RavenSolver, NeoRavenSolver 
from .slot_learner import SlotLearner
from .dalle import Dalle
from .ipcfg import IPCFG 
from .igpt import IGPT 
from .mingpt import MinIGPT
from .dalle_ddp import DalleDDP
from .dalle_mini_vq_encoder import DalleMiniVQEncoder

from fvcore.common.registry import Registry

VQ_MODELS_REGISTRY = Registry("VQ_MODELS")
VQ_MODELS_REGISTRY.__doc__ = """
Registry for parser models.
"""

def build_main_model(cfg, echo):
    return VQ_MODELS_REGISTRY.get(cfg.worker)(cfg, echo)

VQ_MODELS_REGISTRY.register(VQGANEncoder)
VQ_MODELS_REGISTRY.register(RavenSolver)
VQ_MODELS_REGISTRY.register(NeoRavenSolver)
VQ_MODELS_REGISTRY.register(SlotLearner)
VQ_MODELS_REGISTRY.register(Dalle)
VQ_MODELS_REGISTRY.register(IPCFG)
VQ_MODELS_REGISTRY.register(IGPT)
VQ_MODELS_REGISTRY.register(MinIGPT)
VQ_MODELS_REGISTRY.register(DalleDDP)
VQ_MODELS_REGISTRY.register(DalleMiniVQEncoder)
