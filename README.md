# GQE: Grouped Query Experts

Official PyTorch implementation of our paper:

**[Grouped Query Experts: Mixture-of-Experts on GQA Self-Attention](https://arxiv.org/abs/2606.20945)**

[Vishesh Tripathi](https://github.com/FrontiersMindAI), [Abhay Kumar](https://www.linkedin.com/in/akanyaani/)

FrontiersMind

[Paper](https://arxiv.org/abs/2606.20945)

---

## 🚀 Installation

You can install this package using `pip`:

### Basic Installation

```bash
pip install git+https://github.com/FrontiersMindAI/GQE.git
```

### With PyTorch Lightning Support

```bash
pip install "git+https://github.com/FrontiersMindAI/GQE.git#egg=gqe[lightning]"
```

## 🧠 Algorithm Overview

Grouped Query Experts (GQE) is a mixture-of-experts layer on top of grouped-query attention (GQA). Within each GQA group, a router selects **k query-head experts per token** while all key–value (KV) heads remain dense and unchanged. This preserves GQA’s KV-cache benefits and reduces only the active query-head computation.

The default configuration matches the paper’s main setting:

- Softmax over experts **inside each GQA group**, then top-k selection
- Hard concatenation of the selected expert outputs (unscaled)
- A **renormalized weighted-sum slot** so the language-modeling loss trains the router
- An **always-on shared query head** that anchors training
- A Switch-style **load-balancing auxiliary loss**

At the 250M-parameter scale with a 30B-token budget, GQE matches the all-active GQA baseline while activating half of the routed query-head experts, computing 9 of 16 query-attention heads once the shared head is included, and reaching **1.7–1.8×** prefill speedup at long context lengths.

![GQE](./gqe.png)

## 📚 Usage

### Basic Usage

```python
import torch
from gqe import GQEAttention, collect_aux_loss

attn = GQEAttention(
    dim=768,
    num_query_experts=16,  # N query-head experts
    num_kv_heads=8,        # G GQA groups / KV heads
    top_k=1,               # k experts per group
)

x = torch.randn(2, 128, 768)
y = attn(x)  # (2, 128, 768)

# Add the load-balancing loss after the language-modeling loss
loss = lm_loss + 0.01 * collect_aux_loss(attn)
loss.backward()
```

### Training Loop

```python
from gqe import GQE, collect_aux_loss

model = YourModel()  # contains one or more GQEAttention layers
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for batch in dataloader:
    optimizer.zero_grad()
    logits = model(batch)
    lm_loss = loss_fn(logits, labels)
    loss = lm_loss + 0.01 * collect_aux_loss(model)
    loss.backward()
    optimizer.step()
```

### PyTorch Lightning (with optional dependency)

```python
from lightning import Trainer
from gqe import GQELightningCallback

trainer = Trainer(
    callbacks=[
        GQELightningCallback(aux_loss_coef=0.01)
    ]
)

trainer.fit(model, dataloader)
```

The callback injects the GQE auxiliary loss before `loss.backward()` and logs `gqe_aux_loss`. If you already add `collect_aux_loss` in `training_step`, set `inject_aux_loss=False` to avoid applying it twice.

---

## ⚙️ Implementation Details

GQE is a drop-in attention module. It is compatible with standard PyTorch training loops, and the optional Lightning callback uses Lightning’s callback system so no training-step rewrite is required.

The KV path is dense GQA: keys and values are always computed for every group, so the cache layout matches the underlying GQA model. Only selected query experts (plus the shared head) run attention. The shared head is an always-on query that attends against the mean of the dense GQA KV heads, adding one query-attention computation without enlarging the KV cache.

In the 16-query / 8-KV, `k=1` setting this produces 8 hard expert slots + 1 renormalized slot + 1 shared slot, so `W_O` maps `10 · d_h → d` rather than `16 · d_h → d`.

---

## 🔬 Testing & Development

GQE comes with a test suite covering routing, output construction, causal masking, KV cache, and the Lightning callback.

### Running Tests

```bash
./run_tests.sh
```

```bash
./run_tests.sh lightning   # callback tests
./run_tests.sh all
```

---

## 🧪 Usage

### PyTorch

```python
from gqe import GQEAttention, collect_aux_loss

attn = GQEAttention(
    dim=768,
    num_query_experts=16,
    num_kv_heads=8,
    top_k=1,
    variant="gqe",
    is_causal=True,
    use_rope=True,
)

y = attn(x)
loss = lm_loss + 0.01 * collect_aux_loss(attn)
```

### PyTorch Lightning

```python
from gqe import GQELightningCallback

gqe_cb = GQELightningCallback(aux_loss_coef=0.01)

trainer = L.Trainer(
    callbacks=[gqe_cb]
)
```

---

## 🔍 GQE Parameters

| Argument | Description | Default |
|---|---|---|
| `dim` | Hidden size `d`. | required |
| `num_query_experts` | Total routed query-head experts `N`. Must be divisible by `num_kv_heads`. | `16` |
| `num_kv_heads` | GQA groups / dense KV heads `G`. | `8` |
| `top_k` | Experts activated per group `k`. | `1` |
| `head_dim` | Per-head dimension `d_h`. Defaults to `dim // num_query_experts`. | `None` |
| `variant` | Output construction. `"gqe"` uses hard concat + renormalized slot + shared head. `"hard_concat"` and `"weighted_concat"` match the paper ablations. | `"gqe"` |
| `dropout` | Attention dropout. | `0.0` |
| `bias` | Bias on Q/K/V/O projections. | `False` |
| `is_causal` | Causal mask for language-model prefill. | `True` |
| `use_rope` | Rotary position embeddings on queries and keys. | `False` |
| `rope_theta` | RoPE base frequency. | `10000.0` |
| `max_seq_len` | RoPE cache length. | `2048` |

### Callback Parameters

| Argument | Description | Default |
|---|---|---|
| `aux_loss_coef` | Multiplier for the load-balancing auxiliary loss. | `0.01` |
| `inject_aux_loss` | If `True`, backpropagate the aux loss in `on_before_backward`. | `True` |
| `log_stats` | Log `gqe_aux_loss` through Lightning. | `True` |

---

## Ablations

The paper shows that sparse query-head routing matches dense GQA only with renormalized scoring and a shared head. The `variant` flag reproduces Table 2:

```python
GQEAttention(..., variant="weighted_concat")  # no renormalized slot
GQEAttention(..., variant="hard_concat")      # hard concat only
GQEAttention(..., variant="gqe")              # full method
```

---

## Citation

```bibtex
@misc{tripathi2026gqe,
      title={Grouped Query Experts: Mixture-of-Experts on GQA Self-Attention},
      author={Vishesh Tripathi and Abhay Kumar},
      year={2026},
      eprint={2606.20945},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.20945},
}
```

---

## 📜 License

Apache-2.0 license
