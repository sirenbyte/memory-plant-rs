# memory-plant-rs — on-device memory engine

The **canonical Rust engine** for Memory Plant — the product track for a
privacy-first, on-device pocket-AI assistant. Extracted from
[`sirenbyte/MemoryPlant`](https://github.com/sirenbyte/MemoryPlant) (the Python
reference / live MCP server) **with full commit history** via `git subtree`.

Targets edge / on-device AI (UniFFI iOS + Android), server backends via FFI,
and WASM.

## Current capabilities

- **HLB core** — `bind`/`unbind`/`normalize`/cosine, ±1 roles, **provable
  algebraic forget** (GDPR), ~5–7 B/fact compression.
- **Facts** — `PersonalMemory` store/recall/ingest/forget, vocab mgmt.
- **Document RAG** — chunking, caller-embedded `add_document` + rich `search`
  (metadata / contains_text / min_score / doc_ids); multilingual e5
  (`--features fastembed`) or caller-supplied vectors (ORT-free on device).
- **Persistence** — plaintext JSON, **redb** KV, and **ChaCha20-Poly1305
  at-rest encryption** (`save_sealed`/`load_or_create_sealed`).
- **Vector index** — exact brute-force default, optional HNSW (`--features ann`).
- **Mobile bindings** — UniFFI Swift/Kotlin + iOS `xcframework`. See
  [`bindings/README.md`](bindings/README.md), [`CROSS_COMPILE.md`](CROSS_COMPILE.md).
- **Benchmark** — retrieval R@1=0.964 / MRR=0.982 on a curated ru-heavy set
  (see [`BENCH.md`](BENCH.md); honest scope — retrieval-only, not LongMemEval).

113 tests green (`cargo test`).

## Legacy roadmap (historical)

**Phase 0 — HLB core**: ✅ done. `bind`, `unbind`, `normalize`,
`cosine_similarity`, `random_hlb_role`, plus a superposition + AMP
recall test that passes at N=10 in dim=1024.

Roadmap:

| Phase | Scope | Status |
|---|---|---|
| 0 | HLB primitives + tests | ✅ |
| 1 | `AdaptiveMemory` shards + capacity rules | next |
| 2 | `PersonalMemory` (per-user facts, vocab mgmt, auto-upgrade dim) | |
| 3 | `AuditTrail` split-pattern layer | |
| 4 | Persistence (save/load state, serde-json + npy) | |
| 5 | MCP server (port of `mp_mcp.py`) | |
| 6 | Multi-target builds (Linux/Mac/Win/iOS/Android/WASM) | |
| 7 | Language bindings (pyo3, napi-rs, JNI/UniFFI) | |
| 8 | LongMemEval benchmark vs mem0/zep/letta | |

## Why a port

The Python reference at the repo root is feature-complete but limited
to PyTorch deployments (~1 GB install). The Rust impl targets:

- **Edge AI**: ~10 MB native binary, no GIL, no Python runtime
- **Cross-platform**: one Rust codebase → Linux / Mac / Win / iOS /
  Android / WASM via `cargo build --target`
- **Speedup for Python users**: 10-50× faster operations exposed back
  to Python via `pyo3` (drop-in replacement for the hot path)
- **JVM integration**: Memory Plant as a library inside Spring Boot
  apps via JNI / UniFFI

## Build + test

```bash
cd rust/
cargo test --release
```

## Architecture vs Python

Same math (HLB / HRR algebra, AMP refinement, shard scheme,
split-pattern audit). Same `(subject, predicate) → vocab_idx` storage
model. Same provable algebraic forget semantics. Differences:

- Float type is `f32` throughout (Python defaults to it via PyTorch,
  but the type isn't enforced — here it is in the API).
- Roles, values, and memory tensors are `ndarray::Array1<f32>`
  instead of `torch.Tensor`.
- Error handling is explicit `Result<T, HlbError>` instead of
  Python exceptions.
- Random number generator is `ChaCha8Rng` with explicit seeds
  (matches `torch.manual_seed` reproducibility model).

## License

Apache-2.0 — same intent as upstream, more permissive than the
Python reference's proprietary license. Allows commercial edge / OEM
deployments. License will be unified at v1.0.
