# Memory Plant — Rust port

Native-speed implementation of the Memory Plant associative-memory
layer. Targets edge / on-device AI, server backends via FFI bindings,
and WASM.

## Status

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
