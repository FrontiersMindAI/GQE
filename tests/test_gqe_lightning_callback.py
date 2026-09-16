import pytest
import torch
import torch.nn as nn

try:
    import lightning as L
    from lightning.pytorch import LightningModule, Trainer

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    pytest.skip("Lightning not available", allow_module_level=True)

from gqe import GQEAttention, GQELightningCallback, collect_aux_loss


@pytest.mark.lightning
class TinyGQEModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.attn = GQEAttention(
            dim=32,
            num_query_experts=4,
            num_kv_heads=2,
            top_k=1,
            is_causal=True,
        )
        self.head = nn.Linear(32, 4)

    def forward(self, x):
        return self.head(self.attn(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        return nn.functional.cross_entropy(logits.view(-1, 4), y.view(-1))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)


@pytest.mark.lightning
class TestGQELightningCallback:
    def test_callback_initialization(self):
        callback = GQELightningCallback(aux_loss_coef=0.05, inject_aux_loss=False)
        assert callback.aux_loss_coef == 0.05
        assert callback.inject_aux_loss is False

    def test_callback_injects_aux_grad(self, monkeypatch):
        model = TinyGQEModel()
        x = torch.randn(2, 4, 32)
        y = torch.randint(0, 4, (2, 4))
        model(x)
        aux = collect_aux_loss(model)
        assert aux is not None

        callback = GQELightningCallback(aux_loss_coef=1.0)
        dummy_loss = model.head.weight.sum() * 0 + aux.detach()
        # Inject aux backward without the main LM loss.
        model.zero_grad()
        callback.on_before_backward(None, model, dummy_loss)
        assert model.attn.router.weight.grad is not None
        assert model.attn.router.weight.grad.abs().sum() > 0

    def test_trainer_runs_one_step(self):
        torch.manual_seed(0)
        model = TinyGQEModel()
        x = torch.randn(2, 4, 32)
        y = torch.randint(0, 4, (2, 4))
        trainer = Trainer(
            max_steps=1,
            callbacks=[GQELightningCallback(aux_loss_coef=0.01)],
            enable_checkpointing=False,
            logger=False,
            accelerator="cpu",
        )
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=2,
        )
        trainer.fit(model, loader)
