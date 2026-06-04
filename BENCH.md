# Memory Plant — retrieval benchmark

## What this measures (and what it does NOT)

This is a **retrieval-quality** micro-benchmark: given a query, does Memory
Plant's `search` return the relevant passage in the top-k? It measures the
layer MP actually owns — embedding-based retrieval + cosine + filters.

It is **NOT** [LongMemEval](https://github.com/xiaowu0162/LongMemEval). LongMemEval
also tests multi-session aggregation, temporal reasoning, knowledge updates, and
**answer generation** (LLM-as-judge). Those are the *agent's* job, not the
memory store's. So the numbers here are **not directly comparable** to a
LongMemEval "96.2%" — they cover a different (narrower) thing.

## Setup

- **Embedder:** `intfloat/multilingual-e5-small` (384-d) run via **MLX** — the
  Qwen on-device runtime family, no ONNX Runtime (strategy "A′").
  Reference: [`bindings/bench_retrieval_mlx.py`](bindings/bench_retrieval_mlx.py).
- **Corpus:** 33 short passages across 12 topics, ru-heavy (+ en/kz), with
  **hard negatives** (same-topic distractors) so the task is non-trivial.
- **Queries:** 28 labeled, paraphrased (semantic, not lexical), ru-heavy.
- **Harness:** [`src/bin/bench_retrieval.rs`](src/bin/bench_retrieval.rs) loads
  the vectors into `DocumentMemory` and computes Recall@k / MRR (k=10).
- **Vectors committed** as `src/testdata/bench_retrieval.json` → reproducible
  with no Python/MLX/network. Regression-guarded by the `retrieval_bench_floor_e5_mlx`
  test (asserts R@3 ≥ 0.95, MRR ≥ 0.95).

Reproduce:
```sh
# host vectors (needs the MLX venv):
python bindings/bench_retrieval_mlx.py /tmp/bench_retrieval.json
cargo run --bin bench_retrieval -- /tmp/bench_retrieval.json
```

## Results (2026-06-04)

### e5 (multilingual-e5-small via MLX) → Memory Plant

| Lang | n | R@1 | R@3 | R@5 | MRR |
|------|---|-----|-----|-----|-----|
| ru | 23 | 0.957 | 1.000 | 1.000 | 0.978 |
| en | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| kz | 2 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Overall** | **28** | **0.964** | **1.000** | **1.000** | **0.982** |

### MockEncoder byte-hash baseline (floor)

| Lang | n | R@1 | R@3 | R@5 | MRR |
|------|---|-----|-----|-----|-----|
| Overall | 28 | 0.036 | 0.143 | 0.250 | 0.138 |

The baseline ≈ chance — it proves the harness isn't trivially passing; the
e5 lift (R@1 0.04 → 0.96) is the real retrieval signal.

## Honest caveats

- **Small, curated set** (28 queries / 33 passages). Numbers are **indicative**,
  not authoritative. A standard large benchmark (e.g. **MIRACL-ru**, mMARCO-ru)
  would be more credible — that's the next step for a publishable number.
- Mostly **single-hop** (one relevant passage per query). Multi-hop / temporal /
  contradiction-resolution are not exercised here.
- The corpus is hand-built; hard negatives make it non-trivial, but it is not a
  held-out third-party dataset.

## Takeaway

Retrieval quality of the e5→MP pipeline is **strong on this set (R@3 = 1.0,
MRR = 0.98), including Russian** — the priority language. The honest next step
to claim a defensible, comparable number is to run **MIRACL-ru** (or a LongMemEval
ru-style set) at scale.
