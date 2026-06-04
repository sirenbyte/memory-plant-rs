# P3 — Rust on-device port: findings + starter code (5-agent panel, 2026-06-04)

Worktree `feat/mp-multilingual-upgrade`. Port is ~4.3k LoC, structurally ~80% there.
This session = panel spec + starter code + a CRITICAL correctness finding. Actual
implementation + mobile builds + device testing remain multi-week.

---

## 🔴 CRITICAL — HLB Python↔Rust parity is BROKEN (fix before P4 / on-device provable-forget)
The Rust HLB is an internally-valid but **different** implementation — a fact stored by
Python cannot be recalled/forgotten by Rust on the same bank. Three divergences:
1. **Role distribution:** Python = MiND bimodal-normal σ=1/√N (`memory_plant.py:77-90`); Rust = ±1 (`hlb.rs:129-137`, `adaptive.rs:64-74`).
2. **Unbind op:** Python = division `M/role` (`memory_plant.py:98-100`); Rust = multiply `M*role` (`hlb.rs:110-124`). Each correct only for its own role type (multiply-unbind on MiND roles → err 0.76).
3. **RNG/seed:** Python sha256[:8] big-endian → torch.Generator; Rust sha256 full → ChaCha8. Uncorrelated.
**Implication:** provable-forget holds *within* each impl, but is NOT portable. Phone-Rust and hub-Python can't share one HLB bank → breaks P4 sync + "provable forget on-device" trust.
**FIX (recommended):** persist Python's vocab + per-key role tensors, **load them verbatim in Rust** (regenerating from a different RNG can never be bit-exact) + change Rust `unbind` to division + MiND roles. Then add a live PyO3 parity test (needs `pip install maturin`).
*Verified by re-running Python primitives + Rust public API on identical inputs (not via the PyO3 bridge — maturin absent).*

## 🟠 CORRECTION — there is NO encryption-at-rest today
`persistence.rs` writes **plaintext pretty JSON** (`:96-97`). "ChaCha8" in the repo is the deterministic RNG for role/vocab (the zero-blob replay basis), NOT encryption. Earlier "ChaCha at-rest" claims were wrong. Real at-rest secrecy needs adding `chacha20poly1305` AEAD (see P3.4).

---

## P3.1 — Vector index (usearch + pure-Rust fallback)
`document.rs` is brute-force O(N) cosine (`:319-335`) — fine to ~10k chunks; no index seam. Add:
```rust
pub trait Index: Send + Sync {
    fn add(&mut self, id: u64, v: &[f32]);
    fn search(&self, q: &[f32], k: usize) -> Vec<(u64, f32)>;
    fn rebuild(&mut self, vs: &[(u64, Vec<f32>)]);   // forget path = full reset
    fn len(&self) -> usize;
}
```
- **usearch** (C++ core, ARM SIMD, mmap `view`, i8 quant) behind `--features usearch` — breaks "no C deps" + wasm; desktop/server only.
- **instant-distance** (pure-Rust HNSW) = portable ANN fallback. Keep brute-force default for N<10k (`ANN_THRESHOLD=10_000`). `forget` → `rebuild` with survivors. RaBitQ 1-bit packing (`rabitq_index.py`) = separate workstream behind same trait.
- ⚠️ verify usearch 2.x Rust API names before committing.

## P3.2 — On-device embeddings (fastembed → ort/ONNX)
Today embeddings come from Python; phone needs Rust. Existing `FastembedEncoder` uses English AllMiniLM. Switch to **fastembed 5 `EmbeddingModel::MultilingualE5Small`** (384-dim, ru/kz, auto-download+cache). Bake e5 prefixes: `embed_query`→"query: ", `embed_passage`→"passage: " (add `encode_query` to the `Encoder` trait; route `semantic_search`'s query through it). dim 384 = same as MiniLM (index unchanged) BUT e5 vectors incompatible → reindex on switch.
- Model 113MB → on-demand download to app Caches (`with_cache_dir`), not bundled.
- **Mobile ORT = the real pain:** iOS needs ORT compiled from source (CoreML); Android NNAPI from-source; ship CPU-first. Gate behind `--features e5` (breaks wasm).
- **ru/kz fallback:** IBM Granite-Embedding-Multilingual-R2 (sub-100M, explicitly trains Kazakh+Russian, Apache) — custom `ort` loader.

## P3.3 — UniFFI bindings (iOS Swift + Android Kotlin)
Proc-macros (no .udl), UniFFI 0.30. Wrap `MemoryService` in a `Mutex` (Send+Sync) `#[uniffi::Object]`; export ingest/recall/store_fact/store_triplet/forget/forget_all/export/save. `Fact`/`SearchHit`→`#[uniffi::Record]`, `HlbError`→`#[uniffi::Error]`. Generics/`Arc<dyn Extractor>`/`serde_json::Value` stay internal. New `uniffi-bindings/` crate (crate-type cdylib+staticlib+lib), `uniffi-bindgen` bin, `setup_scaffolding!()`. Build: per-target `cargo build` → `uniffi-bindgen generate --library` → xcframework (iOS) / AAR + JNA (Android). ⚠️ Swift-6/Xcode-26 isolation only partial (issue #2818); pin UniFFI (pre-1.0). Full starter wrapper + Swift/Kotlin usage in panel notes. `session_start/end` not in Rust core yet → thin wrappers later.

## P3.4 — Storage (redb) + WASM + build matrix
- **redb 3.1+** (pure-Rust, ACID, wasm-compiles) replaces JSON. Keep zero-blob replay: store `(key, vocab_idx)` rows keyed by `(shard, seq)` so iteration order = store order. Add **chacha20poly1305 AEAD** on value bytes for real at-rest encryption. Migration: `load_json → save_redb`, version both, keep JSON loader one release. (rusqlite rejected: bundles C/SQLCipher.)
- **WASM gating:** `cfg(not(wasm32))` on mcp_server/service/anthropic/openai (fs/http); fix `SystemTime::now` (`audit.rs:21,117`, `document.rs:394`) + `ChaCha8Rng::from_os_rng` (`hlb.rs:132`) via a `now_ms()` shim + `getrandom = {version="0.3", features=["wasm_js"]}`. ndarray+rustfft compile on wasm (no rayon/BLAS). Browser persistence = redb in-memory backend flushed to OPFS/IndexedDB via JS shim.
- **Build matrix** (Makefile): targets ios, ios-sim, android arm64/v7, wasm32, desktop; `cargo build --lib` cross, `cargo test` host-only; CI adds NDK toolchain.

---
## ✅ STARTED (2026-06-04): Rust e5 multilingual embeddings — DONE + tested
Decision: **Rust-only** product (no Python in product; mass-market users start empty → NO data migration; HLB-parity-with-Python is MOOT). Priority **ru + en** (kz = optional bonus).
- `FastembedEncoder::multilingual()` (MultilingualE5Small, 384-dim) + `embed_query`/`embed_passage` (e5 prefixes) added to `src/document.rs`, behind existing `--features fastembed`.
- Built clean (`--features fastembed`, fastembed 5.15 + ort 2.0, 45s; ORT downloaded prebuilt on macOS host — desktop build is painless, only MOBILE ORT is hard).
- Test `document::e5_tests::multilingual_ru_kz_ranks_correctly` PASSES (ru + en correct; kz bonus). Default build (93 tests) untouched.
- ⇒ on-device ru/en embeddings work in the native Rust engine. Next: wire `embed_query` into `semantic_search` (query prefix), then redb + at-rest encryption.

## Prioritized order (Rust-only, ru/en priority)
1. **🔴 HLB parity fix** (persist Python roles/vocab → load in Rust + division-unbind) + maturin PyO3 parity test. *Blocks provable-forget + P4.*
2. redb + at-rest encryption (also fixes the plaintext-JSON gap).
3. fastembed e5 multilingual (matches P0; on-device ru/kz).
4. instant-distance ANN (portable) ; usearch opt-in desktop.
5. UniFFI packaging (iOS/Android) — the longest pole (Swift-6 + ORT mobile builds).
## Effort: ~6–10 wk to ship-quality (UniFFI packaging + mobile ORT are the hard parts). Honest: not closeable in one session.
