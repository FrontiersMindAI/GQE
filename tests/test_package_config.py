import importlib.util

import pytest


def test_base_package():
    import gqe

    assert hasattr(gqe, "GQE")
    assert hasattr(gqe, "GQEAttention")
    assert hasattr(gqe, "collect_aux_loss")
    assert gqe.GQE is gqe.GQEAttention


def test_lightning_import():
    has_lightning = importlib.util.find_spec("lightning") is not None
    import gqe

    if has_lightning:
        assert hasattr(gqe, "GQELightningCallback")
        from gqe import GQELightningCallback

        assert GQELightningCallback is not None
    else:
        assert not hasattr(gqe, "GQELightningCallback")
        with pytest.raises(ImportError):
            from gqe.gqe_lightning_callback import GQELightningCallback
