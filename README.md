## Grouped Query Experts

Self-attention is central to Transformer performance and is often the most expensive part of the Transformer at long context lengths, because its pairwise token interactions scale quadratically with sequence length. Standard dense attention also applies the same set of attention
heads to every token regardless of token difficulty or information content. This uniform activation can waste compute, especially as sequences grow longer and attention cost increases rapidly. We propose Grouped Query Experts (GQE), a mixture-of-experts layer on top of grouped-query
attention (GQA): within each GQA group, a router selects k query-head experts per token while all key–value (KV) heads remain dense and unchanged. Thus GQE keeps the KV-cache benefits of GQA and reduces only the active query-head computation. On a fixed 30B-token budget at
the 250M-parameter scale, GQE matches the all-active GQA baseline in downstream accuracy while activating half of the routed query-head experts per token, and achieves 1.7–1.8× prefill speedup at long context length.


![GQE](./gqe.png)

``` bibtex
@article{tripathi2026gqe,
  title={Grouped Value Attention: Efficient KV Caching via On-Demand Key Reconstruction},
  author={Vishesh Tripathi, Abhay Kumar},
  year={2026}
}
```
