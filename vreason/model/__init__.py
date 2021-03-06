from .helper import * 
from .vq_encoder import VQGANEncoder
from .raven_solver import MetaSolver, RavenSolver, NeoRavenSolver 
from .slot_learner import SlotLearner
from .dalle import Dalle

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
