# memory-plant-rs (Python bindings)

Native-speed Python bindings for the [memory-plant](../) Rust port.
Drop-in usage from Python, 10-50× faster than the pure-Python
reference on hot paths.

## Install

```bash
pip install maturin
maturin develop --release       # build + install into current venv
# OR
maturin build --release         # produce a .whl in target/wheels/
pip install target/wheels/*.whl
```

## Use

```python
import memory_plant_rs as mp

# Direct HLB store
am = mp.AdaptiveMemory(dim=1024, vocab_cap=256, seed=42)
am.store("user|works_as", "engineer")
am.store("user|lives_in", "tokyo")
assert am.retrieve("user|works_as") == "engineer"

# Provable algebraic forget (residual ≈ 0)
am.forget("user|works_as")
assert am.retrieve("user|works_as") is None

# Per-user wrapper with regex fact extraction
pm = mp.PersonalMemory("alice", dim=1024, vocab_cap=256, seed=42)
facts = pm.ingest("I work as engineer and live in tokyo")
print(facts)
# [{'subject': 'user', 'predicate': 'works_as', 'obj': 'engineer'},
#  {'subject': 'user', 'predicate': 'lives_in', 'obj': 'tokyo'}]

assert pm.recall("works_as") == "engineer"
print(pm.all_facts())
# {'user|works_as': 'engineer', 'user|lives_in': 'tokyo'}

# GDPR Article 17
pm.forget_all()
```

## Performance vs pure Python

Indicative numbers on Apple M-series CPU:

| Operation | Python ref | Rust binding | Speedup |
|---|---|---|---|
| HLB bind / unbind (d=1024) | ~50 μs | ~3 μs | ~17× |
| AdaptiveMemory.store (1 fact) | ~150 μs | ~10 μs | ~15× |
| AdaptiveMemory.retrieve (AMP) | ~500 μs | ~50 μs | ~10× |
| Bulk replay 1000 facts | ~3 s | ~200 ms | ~15× |

(Measured on the same logical operations — Rust eliminates the
interpreter / autograd overhead PyTorch adds.)

## What's exposed (Phase 7a)

- `AdaptiveMemory(dim, vocab_cap, shard_capacity=None, seed=42)`
  - `.store(key, value) -> int`
  - `.retrieve(key) -> str | None`
  - `.forget(key) -> bool`
  - `.dim`, `.shard_capacity`, `.n_shards`, `.total_facts`

- `PersonalMemory(user_id, dim=512, vocab_cap=4096, seed=42)`
  - `.ingest(message) -> list[dict]`
  - `.store_fact(predicate, value, subject="user")`
  - `.recall(predicate, subject=None) -> str | None`
  - `.all_facts() -> dict`
  - `.forget(predicate, subject=None) -> bool`
  - `.forget_all() -> int`
  - `.user_id`

Future additions: AnthropicExtractor wrapper, DocumentMemory (with
fastembed feature), AuditTrail, persistence helpers.
