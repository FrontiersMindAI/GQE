import pytest
import torch
import torch.nn as nn

from gqe import GQEAttention, collect_aux_loss
from gqe.gqe import load_balancing_loss


class TinyLM(nn.Module):
    def __init__(self, attn: GQEAttention):
        super().__init__()
        self.attn = attn
        self.head = nn.Linear(attn.dim, 8)

    def forward(self, x):
        return self.head(self.attn(x))


class TestGQEAttention:
    @pytest.fixture
    def attn(self):
        torch.manual_seed(0)
        return GQEAttention(
            dim=32,
            num_query_experts=4,
            num_kv_heads=2,
            top_k=1,
            is_causal=True,
        )

    def test_invalid_group_split(self):
        with pytest.raises(ValueError, match="divisible"):
            GQEAttention(dim=32, num_query_experts=5, num_kv_heads=2)

    def test_invalid_top_k(self):
        with pytest.raises(ValueError, match="top_k"):
            GQEAttention(dim=32, num_query_experts=4, num_kv_heads=2, top_k=3)

    def test_invalid_variant(self):
        with pytest.raises(ValueError, match="variant"):
            GQEAttention(dim=32, num_query_experts=4, num_kv_heads=2, variant="nope")

    def test_output_shape_and_slots(self, attn):
        x = torch.randn(2, 6, 32)
        y = attn(x)
        assert y.shape == x.shape
        # kG + renormalized slot + shared head = 2 + 1 + 1 = 4
        assert attn.num_output_slots == 4
        assert attn.o_proj.in_features == 4 * attn.head_dim
        assert attn.active_query_heads == 3  # 2 routed + 1 shared

    def test_main_paper_shapes(self):
        attn = GQEAttention(
            dim=64,
            num_query_experts=16,
            num_kv_heads=8,
            top_k=1,
            head_dim=4,
        )
        assert attn.num_output_slots == 10
        assert attn.active_query_heads == 9
        y = attn(torch.randn(1, 3, 64))
        assert y.shape == (1, 3, 64)

    def test_selects_k_per_group(self, attn):
        x = torch.randn(3, 5, 32)
        out = attn(x, return_output=True)
        # (B, S, G, k)
        assert out.expert_indices.shape == (3, 5, 2, 1)
        assert out.router_probs.shape == (3, 5, 2, 2)
        assert torch.allclose(out.router_probs.sum(dim=-1), torch.ones(3, 5, 2))

    def test_causal_mask(self, attn):
        x = torch.randn(1, 8, 32)
        y1 = attn(x)
        x2 = x.clone()
        x2[0, -1] = torch.randn(32)
        y2 = attn(x2)
        assert torch.allclose(y1[:, :-1], y2[:, :-1], atol=1e-5, rtol=1e-5)

    def test_kv_cache_matches_prefill(self):
        torch.manual_seed(1)
        attn = GQEAttention(
            dim=32,
            num_query_experts=4,
            num_kv_heads=2,
            top_k=1,
            is_causal=True,
        )
        attn.eval()
        x = torch.randn(2, 5, 32)
        with torch.no_grad():
            y_full = attn(x)
            past = None
            chunks = []
            for t in range(x.size(1)):
                y_t, past = attn(x[:, t : t + 1], past_key_value=past, use_cache=True)
                chunks.append(y_t)
            y_step = torch.cat(chunks, dim=1)
        assert torch.allclose(y_full, y_step, atol=1e-5, rtol=1e-5)

    def test_router_gradient_through_renormalized_slot(self, attn):
        x = torch.randn(2, 4, 32, requires_grad=True)
        y = attn(x)
        y.sum().backward()
        assert attn.router.weight.grad is not None
        assert attn.router.weight.grad.abs().sum() > 0

    def test_hard_concat_has_no_router_lm_gradient(self):
        torch.manual_seed(0)
        attn = GQEAttention(
            dim=32,
            num_query_experts=4,
            num_kv_heads=2,
            top_k=1,
            variant="hard_concat",
        )
        x = torch.randn(2, 4, 32)
        y = attn(x)
        y.sum().backward()
        assert attn.router.weight.grad is None or attn.router.weight.grad.abs().sum() == 0

    def test_aux_loss_trains_router_on_hard_concat(self):
        torch.manual_seed(0)
        attn = GQEAttention(
            dim=32,
            num_query_experts=4,
            num_kv_heads=2,
            top_k=1,
            variant="hard_concat",
        )
        x = torch.randn(2, 4, 32)
        y = attn(x)
        (y.sum() * 0 + attn.last_aux_loss).backward()
        assert attn.router.weight.grad is not None
        assert attn.router.weight.grad.abs().sum() > 0

    def test_collect_aux_loss(self, attn):
        model = TinyLM(attn)
        x = torch.randn(2, 4, 32)
        model(x)
        aux = collect_aux_loss(model)
        assert aux is not None
        assert aux.ndim == 0
        assert aux.item() > 0

    def test_load_balance_prefers_uniform_dispatch(self):
        probs = torch.zeros(4, 8, 2, 2)
        probs[..., 0] = 0.9
        probs[..., 1] = 0.1
        idx_collapsed = torch.zeros(4, 8, 2, 1, dtype=torch.long)
        collapsed = load_balancing_loss(probs, idx_collapsed)
        idx_uniform = torch.zeros(4, 8, 2, 1, dtype=torch.long)
        idx_uniform[:, :4] = 0
        idx_uniform[:, 4:] = 1
        balanced = load_balancing_loss(probs, idx_uniform)
        assert balanced.item() < collapsed.item()
        assert pytest.approx(balanced.item(), rel=1e-5) == 1.0

    def test_variants_change_output_projection(self):
        kwargs = dict(dim=32, num_query_experts=4, num_kv_heads=2, top_k=1)
        gqe = GQEAttention(**kwargs, variant="gqe")
        hard = GQEAttention(**kwargs, variant="hard_concat")
        weighted = GQEAttention(**kwargs, variant="weighted_concat")
        assert gqe.num_output_slots == 4
        assert hard.num_output_slots == 2
        assert weighted.num_output_slots == 2
        x = torch.randn(1, 3, 32)
        assert gqe(x).shape == hard(x).shape == weighted(x).shape == x.shape

    def test_rope_forward(self):
        attn = GQEAttention(
            dim=32,
            num_query_experts=4,
            num_kv_heads=2,
            top_k=1,
            use_rope=True,
        )
        y = attn(torch.randn(2, 7, 32))
        assert y.shape == (2, 7, 32)

    def test_training_step(self, attn):
        model = TinyLM(attn)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randn(2, 5, 32)
        y = torch.randint(0, 8, (2, 5))
        opt.zero_grad()
        logits = model(x)
        lm_loss = nn.functional.cross_entropy(logits.view(-1, 8), y.view(-1))
        loss = lm_loss + 0.01 * collect_aux_loss(model)
        loss.backward()
        opt.step()
        assert torch.isfinite(loss).item()
