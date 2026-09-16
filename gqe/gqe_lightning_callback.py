from gqe.gqe import collect_aux_loss

try:
    import lightning as L
except ImportError as exc:
    raise ImportError(
        "PyTorch Lightning is required to use GQELightningCallback. "
        "Please install it with: pip install 'gqe[lightning]'"
    ) from exc


class GQELightningCallback(L.Callback):
    """PyTorch Lightning callback for Grouped Query Experts.

    Before ``loss.backward()``, injects the Switch-style load-balancing
    auxiliary loss from every ``GQEAttention`` module and logs routing stats.
    """

    def __init__(
        self,
        aux_loss_coef: float = 0.01,
        inject_aux_loss: bool = True,
        log_stats: bool = True,
    ) -> None:
        super().__init__()
        self.aux_loss_coef = aux_loss_coef
        self.inject_aux_loss = inject_aux_loss
        self.log_stats = log_stats

    def on_before_backward(self, trainer, pl_module, loss) -> None:
        aux = collect_aux_loss(pl_module)
        if aux is None:
            return

        if self.log_stats and hasattr(pl_module, "log"):
            pl_module.log(
                "gqe_aux_loss",
                aux.detach(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
            )

        if not self.inject_aux_loss:
            return
        scaled = self.aux_loss_coef * aux
        if scaled.requires_grad:
            scaled.backward(retain_graph=True)
