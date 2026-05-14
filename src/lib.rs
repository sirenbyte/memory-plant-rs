//! Memory Plant — Rust port.
//!
//! A compressed associative-memory layer for AI agents. Stores
//! `(key → value)` facts using algebraic binding (HLB or HRR) so the
//! whole set of N facts fits in a single fixed-size tensor `M` with
//! amortized cost ~5-7 bytes per fact.
//!
//! This crate is the high-performance Rust implementation of the
//! Python reference at the root of the repo. It targets:
//!
//! - **Edge / on-device AI** — ~10 MB binary, no GIL, no PyTorch.
//! - **Server backends** — pyo3 / napi-rs / JNI bindings give 10-50×
//!   speedup to existing Python / Node / JVM applications.
//! - **WASM** — browser / cross-platform deployments.
//!
//! ## Architecture (Phase 0 of the port — only HLB core is here)
//!
//! ```text
//! lib.rs              — crate root, re-exports public API
//! ├── hlb.rs          — bind / unbind / normalize / cosine sim
//! └── (future)
//!     ├── adaptive.rs — AdaptiveMemory: shard-based scaling
//!     ├── personal.rs — PersonalMemory: per-user facts + vocab mgmt
//!     ├── audit.rs    — AuditTrail: split-pattern audit log
//!     └── mcp.rs      — MCP server (port of mp_mcp.py)
//! ```
//!
//! ## Mathematical reference
//!
//! HLB (Holographic Linear Binding) — element-wise multiplication
//! with ±1 bimodal roles. For role `r ∈ {-1, +1}^d`:
//! ```text
//!     bind(r, v)         = r ⊙ v
//!     unbind(M, r)       = M ⊙ r            ; because r ⊙ r = 1
//!     M = bind(r1, v1) + bind(r2, v2) + ...
//!     unbind(M, r1)      ≈ v1               ; with cross-talk noise
//! ```
//! Cross-talk noise after AMP refinement is σ = √(N-1)/√d for N facts
//! at dimension d. Safe capacity ~d/13 facts per shard at HLB recall ≥95%.
//!
//! See the Python reference's `memory_plant.py` for the full
//! mathematical exposition.

pub mod hlb;
pub mod vocab;
pub mod adaptive;

// Re-export the most-used items at crate root.
pub use hlb::{bind_hlb, unbind_hlb, normalize, cosine_similarity, random_hlb_role, HlbError};
pub use vocab::Vocab;
pub use adaptive::{AdaptiveMemory, role_from_key, safe_shard_capacity};
