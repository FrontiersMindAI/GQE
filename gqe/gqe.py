"""Grouped Query Experts (GQE): mixture-of-experts on GQA self-attention.

Paper: Tripathi & Kumar, "Grouped Query Experts: Mixture-of-Experts on GQA
Self-Attention", arXiv:2606.20945.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GQEOutput:
    """Return type when ``return_output=True``."""

    hidden_states: torch.Tensor
    aux_loss: torch.Tensor
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    router_probs: Optional[torch.Tensor] = None
    expert_indices: Optional[torch.Tensor] = None


def collect_aux_loss(module: nn.Module) -> Optional[torch.Tensor]:
    """Sum load-balancing losses stored by every ``GQEAttention`` in ``module``."""
    losses = [
        m.last_aux_loss
        for m in module.modules()
        if isinstance(m, GQEAttention) and m.last_aux_loss is not None
    ]
    if not losses:
        return None
    return torch.stack(losses).sum()


def load_balancing_loss(
    router_probs: torch.Tensor,
    expert_indices: torch.Tensor,
) -> torch.Tensor:
    """Switch-Transformer load-balancing loss, averaged over GQA groups.

    Args:
        router_probs: Softmax probabilities, shape ``(B, S, G, M)``.
        expert_indices: Top-k expert ids, shape ``(B, S, G, k)``.

    Returns:
        Scalar auxiliary loss. Uniform top-1 routing yields a value near 1.
    """
    _, _, _, num_experts = router_probs.shape
    ones = torch.ones_like(expert_indices, dtype=router_probs.dtype)
    mask = torch.zeros_like(router_probs)
    mask.scatter_add_(-1, expert_indices, ones)
    # Fraction of tokens dispatched to each expert, per group.
    expert_load = mask.mean(dim=(0, 1))
    mean_prob = router_probs.mean(dim=(0, 1))
    return (num_experts * (expert_load * mean_prob).sum(dim=-1)).mean()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class GQEAttention(nn.Module):
    """Grouped Query Experts attention layer.

    Within each GQA group a router selects ``top_k`` query-head experts per
    token. All KV heads stay dense, so the KV cache matches GQA. The default
    ``variant="gqe"`` concatenates:

    1. Hard-selected expert outputs (unscaled)
    2. A renormalized weighted-sum slot (router learning signal)
    3. An always-on shared query head

    before the output projection ``W_O``, matching Section 3 of the paper.

    Args:
        dim: Model hidden size ``d``.
        num_query_experts: Total routed query-head experts ``N``.
        num_kv_heads: Number of GQA groups / KV heads ``G``.
        top_k: Experts activated per group ``k``.
        head_dim: Per-head dimension. Defaults to ``dim // num_query_experts``.
        variant: ``"gqe"``, ``"hard_concat"``, or ``"weighted_concat"``.
        dropout: Attention dropout probability.
        bias: Whether Q/K/V/O projections use bias.
        is_causal: Apply a causal attention mask when no explicit mask is given
            and query/key lengths match (prefill).
        use_rope: Apply rotary position embeddings to queries and keys.
        rope_theta: RoPE base frequency.
        max_seq_len: RoPE cache length.
        eps: Numerical stability constant for probability renormalization.
    """

    def __init__(
        self,
        dim: int,
        num_query_experts: int = 16,
        num_kv_heads: int = 8,
        top_k: int = 1,
        head_dim: Optional[int] = None,
        variant: str = "gqe",
        dropout: float = 0.0,
        bias: bool = False,
        is_causal: bool = True,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        if num_query_experts % num_kv_heads != 0:
            raise ValueError(
                f"num_query_experts ({num_query_experts}) must be divisible by "
                f"num_kv_heads ({num_kv_heads})."
            )
        experts_per_group = num_query_experts // num_kv_heads
        if not 1 <= top_k <= experts_per_group:
            raise ValueError(
                f"top_k ({top_k}) must be in [1, {experts_per_group}]."
            )
        variant = variant.lower()
        if variant not in {"gqe", "hard_concat", "weighted_concat"}:
            raise ValueError(
                "variant must be 'gqe', 'hard_concat', or 'weighted_concat'."
            )
        if head_dim is None:
            if dim % num_query_experts != 0:
                raise ValueError(
                    f"dim ({dim}) must be divisible by num_query_experts "
                    f"({num_query_experts}) when head_dim is omitted."
                )
            head_dim = dim // num_query_experts
        if use_rope and head_dim % 2 != 0:
            raise ValueError("head_dim must be even when use_rope=True.")

        self.dim = dim
        self.num_query_experts = num_query_experts
        self.num_kv_heads = num_kv_heads
        self.experts_per_group = experts_per_group
        self.top_k = top_k
        self.head_dim = head_dim
        self.variant = variant
        self.dropout = dropout
        self.is_causal = is_causal
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.eps = eps

        self.renormalized_slot = variant == "gqe"
        self.shared_head = variant == "gqe"
        self.weight_selected_slots = variant == "weighted_concat"

        self.q_proj = nn.Linear(dim, num_query_experts * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=bias)
        self.router = nn.Linear(
            dim, num_kv_heads * experts_per_group, bias=False
        )
        if self.shared_head:
            self.q_shared = nn.Linear(dim, head_dim, bias=bias)
        else:
            self.q_shared = None
        self.o_proj = nn.Linear(self.num_output_slots * head_dim, dim, bias=bias)

        if use_rope:
            self.register_buffer(
                "rope_cos",
                torch.empty(0),
                persistent=False,
            )
            self.register_buffer(
                "rope_sin",
                torch.empty(0),
                persistent=False,
            )
            self._build_rope_cache(max_seq_len)

        self.last_aux_loss: Optional[torch.Tensor] = None
        self.last_router_probs: Optional[torch.Tensor] = None
        self.last_expert_indices: Optional[torch.Tensor] = None

        self._reset_parameters()

    @property
    def num_output_slots(self) -> int:
        """Number of ``d_h``-sized slots concatenated before ``W_O``."""
        slots = self.top_k * self.num_kv_heads
        if self.renormalized_slot:
            slots += 1
        if self.shared_head:
            slots += 1
        return slots

    @property
    def active_query_heads(self) -> int:
        """Query-attention computations per token, including the shared head."""
        return self.top_k * self.num_kv_heads + int(self.shared_head)

    def _reset_parameters(self) -> None:
        for module in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if self.q_shared is not None:
            nn.init.xavier_uniform_(self.q_shared.weight)
            if self.q_shared.bias is not None:
                nn.init.zeros_(self.q_shared.bias)
        # Near-uniform routing at initialization.
        nn.init.zeros_(self.router.weight)

    def _build_rope_cache(self, seq_len: int, device=None, dtype=None) -> None:
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32)
                / self.head_dim
            )
        )
        positions = torch.arange(seq_len, device=inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.rope_cos = emb.cos().to(dtype=dtype or torch.float32)
        self.rope_sin = emb.sin().to(dtype=dtype or torch.float32)
        self.max_seq_len = seq_len

    def _rope_tensors(
        self, seq_len: int, offset: int, device, dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        total = offset + seq_len
        if self.rope_cos.numel() == 0 or total > self.rope_cos.size(0):
            self._build_rope_cache(max(total, self.max_seq_len), device=device, dtype=dtype)
        if self.rope_cos.device != device or self.rope_cos.dtype != dtype:
            self.rope_cos = self.rope_cos.to(device=device, dtype=dtype)
            self.rope_sin = self.rope_sin.to(device=device, dtype=dtype)
        cos = self.rope_cos[offset:total]
        sin = self.rope_sin[offset:total]
        return cos, sin

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (B, S, ..., D), cos/sin: (S, D)
        shape = [1, cos.size(0)] + [1] * (x.dim() - 3) + [cos.size(-1)]
        cos = cos.view(*shape)
        sin = sin.view(*shape)
        return x * cos + _rotate_half(x) * sin

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        query_len: int,
        key_len: int,
    ) -> torch.Tensor:
        dropout_p = self.dropout if self.training else 0.0
        causal = self.is_causal and attention_mask is None and query_len == key_len
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=dropout_p,
            is_causal=causal,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        return_output: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], GQEOutput]:
        """
        Args:
            hidden_states: ``(batch, seq, dim)``.
            attention_mask: Optional SDPA mask broadcastable to
                ``(batch, heads, q_len, k_len)``.
            past_key_value: Cached ``(key, value)`` with shape
                ``(batch, kv_seq, G, head_dim)`` each.
            use_cache: If True, also return updated KV tensors.
            return_output: If True, return a :class:`GQEOutput`.

        Returns:
            Hidden states of shape ``(batch, seq, dim)``, or a cache / output
            wrapper when requested.
        """
        batch, seq_len, _ = hidden_states.shape
        g = self.num_kv_heads
        m = self.experts_per_group
        k = self.top_k
        d = self.head_dim

        query = self.q_proj(hidden_states).view(batch, seq_len, g, m, d)
        key = self.k_proj(hidden_states).view(batch, seq_len, g, d)
        value = self.v_proj(hidden_states).view(batch, seq_len, g, d)

        past_len = 0 if past_key_value is None else past_key_value[0].size(1)
        if self.use_rope:
            q_cos, q_sin = self._rope_tensors(
                seq_len, past_len, hidden_states.device, query.dtype
            )
            query = self._apply_rope(query, q_cos, q_sin)
            key = self._apply_rope(key, q_cos, q_sin)

        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=1)
            value = torch.cat([past_key_value[1], value], dim=1)
        kv_len = key.size(1)

        router_logits = self.router(hidden_states).view(batch, seq_len, g, m)
        router_probs = F.softmax(router_logits, dim=-1)
        topk_probs, expert_indices = torch.topk(router_probs, k, dim=-1)

        gather_index = expert_indices.unsqueeze(-1).expand(batch, seq_len, g, k, d)
        query_selected = torch.gather(query, dim=3, index=gather_index)

        # (B, G*k, S, D) queries against per-group KV repeated k times.
        query_attn = query_selected.reshape(batch, seq_len, g * k, d).transpose(1, 2)
        key_attn = (
            key.unsqueeze(3)
            .expand(batch, kv_len, g, k, d)
            .reshape(batch, kv_len, g * k, d)
            .transpose(1, 2)
        )
        value_attn = (
            value.unsqueeze(3)
            .expand(batch, kv_len, g, k, d)
            .reshape(batch, kv_len, g * k, d)
            .transpose(1, 2)
        )
        selected_out = self._attend(
            query_attn, key_attn, value_attn, attention_mask, seq_len, kv_len
        )
        selected_out = selected_out.transpose(1, 2).reshape(batch, seq_len, g, k, d)

        if self.weight_selected_slots:
            slots = (selected_out * topk_probs.unsqueeze(-1)).reshape(
                batch, seq_len, g * k * d
            )
        else:
            slots = selected_out.reshape(batch, seq_len, g * k * d)

        parts = [slots]
        if self.renormalized_slot:
            weights = topk_probs / topk_probs.sum(dim=(-1, -2), keepdim=True).clamp_min(
                self.eps
            )
            weighted = (selected_out * weights.unsqueeze(-1)).sum(dim=(2, 3))
            parts.append(weighted)
        if self.shared_head:
            query_shared = self.q_shared(hidden_states).view(batch, seq_len, d)
            if self.use_rope:
                query_shared = self._apply_rope(query_shared, q_cos, q_sin)
            key_shared = key.mean(dim=2)
            value_shared = value.mean(dim=2)
            shared_out = self._attend(
                query_shared.unsqueeze(1),
                key_shared.unsqueeze(1),
                value_shared.unsqueeze(1),
                attention_mask,
                seq_len,
                kv_len,
            ).squeeze(1)
            parts.append(shared_out)

        hidden = self.o_proj(torch.cat(parts, dim=-1))
        aux_loss = load_balancing_loss(router_probs, expert_indices)
        self.last_aux_loss = aux_loss
        self.last_router_probs = router_probs.detach()
        self.last_expert_indices = expert_indices.detach()
        present = (key, value) if use_cache else None

        if return_output:
            return GQEOutput(
                hidden_states=hidden,
                aux_loss=aux_loss,
                past_key_value=present,
                router_probs=router_probs,
                expert_indices=expert_indices,
            )
        if use_cache:
            return hidden, present
        return hidden
