## Grouped Query Experts

Self-attention is a major bottleneck in Transformers at long context lengths because its compute grows quadratically with sequence length. We propose Grouped Query Experts (GQE), a mixture-of-experts layer built on grouped-query attention (GQA). For each token, a router selects only k query-head experts within each GQA group, while keeping all key–value (KV) heads dense and unchanged. This preserves GQA’s KV-cache benefits while reducing query-head computation. At the 250M-parameter scale, GQE matches standard GQA accuracy on a fixed 30B-token training budget while activating only half of the query-head experts, achieving 1.7–1.8× faster prefill at long context lengths.

![GQE](./gqe.png)

``` bibtex
@article{tripathi2026gqe,
  title={Grouped Value Attention: Efficient KV Caching via On-Demand Key Reconstruction},
  author={Vishesh Tripathi, Abhay Kumar},
  year={2026}
}
```
