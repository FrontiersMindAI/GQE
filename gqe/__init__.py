from gqe.gqe import GQEAttention, GQEOutput, collect_aux_loss

GQE = GQEAttention

__version__ = "1.0.0"
__all__ = ["GQE", "GQEAttention", "GQEOutput", "collect_aux_loss"]

try:
    from importlib.util import find_spec

    if find_spec("lightning") is not None:
        from gqe.gqe_lightning_callback import GQELightningCallback

        __all__ += ["GQELightningCallback"]
except ImportError:
    pass
