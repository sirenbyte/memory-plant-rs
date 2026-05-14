//! HLB (Holographic Linear Binding) primitives.
//!
//! HLB stores `(role, value) → memory` associations through element-wise
//! multiplication with `±1` bipolar role vectors. This file is the
//! kernel of Memory Plant — everything higher (shards, personal
//! memory, audit log) sits on top of these few functions.
//!
//! ## Operations
//!
//! | op         | formula                       | cost |
//! |------------|-------------------------------|------|
//! | `bind`     | `r ⊙ v`                       | O(d) |
//! | `unbind`   | `M ⊙ r`  (since `r ⊙ r = 1`) | O(d) |
//! | `normalize`| `v / ‖v‖₂`                    | O(d) |
//! | `cosine`   | `(a · b) / (‖a‖ · ‖b‖)`       | O(d) |
//!
//! ## Equivalence to the Python reference
//!
//! The Python `hlb_bind` / `hlb_unbind` in `memory_plant.py` take
//! `torch.Tensor`. Here they take `ndarray::ArrayView1<f32>` so the
//! same numerical operations apply. Cross-validation tests verify
//! identical outputs on identical inputs.

use ndarray::{Array1, ArrayView1};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use thiserror::Error;

/// All errors this module can raise. Memory-Plant-internal call sites
/// should `?`-propagate; FFI layers translate to platform-native types.
#[derive(Debug, Error)]
pub enum HlbError {
    #[error("dim mismatch: lhs={lhs}, rhs={rhs}")]
    DimMismatch { lhs: usize, rhs: usize },
    #[error("zero-norm vector: cannot normalize")]
    ZeroNorm,
    #[error("vocab full: cap={cap}, attempted to register {attempted:?}")]
    VocabFull { cap: usize, attempted: String },
    #[error("key not found in any shard: {key:?}")]
    KeyNotFound { key: String },
}

/// L2-normalize a vector in place. Returns `ZeroNorm` if the input
/// has magnitude below `f32::EPSILON` (silent NaN propagation would
/// poison downstream similarity comparisons).
pub fn normalize(v: &mut Array1<f32>) -> Result<(), HlbError> {
    let norm = v.dot(v).sqrt();
    if norm < f32::EPSILON {
        return Err(HlbError::ZeroNorm);
    }
    v.mapv_inplace(|x| x / norm);
    Ok(())
}

/// Cosine similarity between two equal-length vectors. Returns a
/// value in `[-1, 1]`. Does NOT require the inputs to be normalized —
/// normalization is folded into the computation. Saves a `clone()`
/// at the call site, hot enough to matter on retrieve paths.
pub fn cosine_similarity(
    a: ArrayView1<f32>,
    b: ArrayView1<f32>,
) -> Result<f32, HlbError> {
    if a.len() != b.len() {
        return Err(HlbError::DimMismatch {
            lhs: a.len(),
            rhs: b.len(),
        });
    }
    let dot = a.dot(&b);
    let na = a.dot(&a).sqrt();
    let nb = b.dot(&b).sqrt();
    let denom = na * nb;
    if denom < f32::EPSILON {
        return Ok(0.0);
    }
    Ok(dot / denom)
}

/// HLB bind: element-wise multiplication of a `±1` role vector with a
/// value vector. Result has the same dimension; algebraically it's
/// the encoding of `(role, value)` such that `unbind_hlb(result, role)`
/// recovers `value` exactly.
///
/// In the superposition setting `M = bind(r1, v1) + bind(r2, v2) + ...`,
/// `unbind_hlb(M, r_i)` recovers `v_i` plus cross-talk noise from the
/// other facts. AMP refinement against a known vocab snaps noisy
/// vectors back to the nearest vocab entry.
#[inline]
pub fn bind_hlb(
    role: ArrayView1<f32>,
    value: ArrayView1<f32>,
) -> Result<Array1<f32>, HlbError> {
    if role.len() != value.len() {
        return Err(HlbError::DimMismatch {
            lhs: role.len(),
            rhs: value.len(),
        });
    }
    Ok(&role * &value)
}

/// HLB unbind: recover the value previously bound to `role` from
/// `memory`. Algebraically `memory * role`, which exploits the HLB
/// invariant `role * role = 1` for bipolar `±1` roles. For HRR this
/// would be circular correlation (FFT-based); HLB is the cheaper
/// element-wise version. The Rust impl is the same single line —
/// most of the speedup comes from avoiding Python interpreter
/// overhead on hot retrieve loops.
#[inline]
pub fn unbind_hlb(
    memory: ArrayView1<f32>,
    role: ArrayView1<f32>,
) -> Result<Array1<f32>, HlbError> {
    if memory.len() != role.len() {
        return Err(HlbError::DimMismatch {
            lhs: memory.len(),
            rhs: role.len(),
        });
    }
    // bind == unbind for HLB because role * role = 1 element-wise.
    // Same single matmul, kept as a distinct function for clarity
    // at the call site and so future ops (HRR, MAP-C) can specialise.
    Ok(&memory * &role)
}

/// Build a random bipolar `±1` role vector of dimension `dim`. Uses
/// `ChaCha8Rng` so callers can pin a seed for reproducible benchmarks.
/// Pass `seed=None` for OS entropy.
pub fn random_hlb_role(dim: usize, seed: Option<u64>) -> Array1<f32> {
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_os_rng(),
    };
    Array1::from_shape_fn(dim, |_| {
        if rng.random::<bool>() { 1.0 } else { -1.0 }
    })
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn normalize_unit_vector_unchanged() {
        let mut v = array![1.0_f32, 0.0, 0.0];
        normalize(&mut v).unwrap();
        assert_abs_diff_eq!(v[0], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn normalize_zero_errors() {
        let mut v = array![0.0_f32, 0.0, 0.0];
        let result = normalize(&mut v);
        assert!(matches!(result, Err(HlbError::ZeroNorm)));
    }

    #[test]
    fn cosine_identical_vectors_is_one() {
        let v = array![1.0_f32, 2.0, 3.0];
        let c = cosine_similarity(v.view(), v.view()).unwrap();
        assert_abs_diff_eq!(c, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = array![1.0_f32, 0.0];
        let b = array![0.0_f32, 1.0];
        let c = cosine_similarity(a.view(), b.view()).unwrap();
        assert_abs_diff_eq!(c, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn bind_unbind_roundtrip_exact_single_fact() {
        // The defining invariant: bind then unbind with the same role
        // recovers the value perfectly when the memory holds only one
        // fact. Cross-talk only appears in superposition.
        let role = random_hlb_role(512, Some(42));
        let value = random_hlb_role(512, Some(7)); // a different bipolar
        let memory = bind_hlb(role.view(), value.view()).unwrap();
        let recovered = unbind_hlb(memory.view(), role.view()).unwrap();
        for (got, want) in recovered.iter().zip(value.iter()) {
            assert_abs_diff_eq!(got, want, epsilon = 1e-6);
        }
    }

    #[test]
    fn superposition_then_amp_lookup() {
        // Real-world setup: vocab of K entries, store N=10 random
        // facts in a single memory tensor, then unbind one role and
        // verify the closest vocab entry is the one we stored.
        const DIM: usize = 1024;
        const K: usize = 64;
        const N: usize = 10;

        // Build vocab — random unit-norm vectors.
        let mut vocab: Vec<Array1<f32>> = (0..K)
            .map(|i| {
                let mut rng = ChaCha8Rng::seed_from_u64(1000 + i as u64);
                let mut v = Array1::from_shape_fn(DIM, |_| {
                    rng.random::<f32>() * 2.0 - 1.0
                });
                normalize(&mut v).unwrap();
                v
            })
            .collect();

        // Build N roles + remember which vocab idx each fact stores.
        let mut roles = Vec::with_capacity(N);
        let mut truth: Vec<usize> = Vec::with_capacity(N);
        let mut memory = Array1::<f32>::zeros(DIM);

        for i in 0..N {
            let role = random_hlb_role(DIM, Some(2000 + i as u64));
            let vocab_idx = (i * 7) % K; // some deterministic spread
            let bound = bind_hlb(role.view(), vocab[vocab_idx].view()).unwrap();
            memory = &memory + &bound;
            roles.push(role);
            truth.push(vocab_idx);
        }

        // Retrieve each fact: unbind by its role, find nearest vocab.
        let mut correct = 0;
        for i in 0..N {
            let unbound = unbind_hlb(memory.view(), roles[i].view()).unwrap();
            // AMP refinement: argmax cosine against vocab.
            let (best_idx, _) = vocab
                .iter()
                .enumerate()
                .map(|(idx, v)| {
                    (idx, cosine_similarity(unbound.view(), v.view()).unwrap())
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            if best_idx == truth[i] {
                correct += 1;
            }
        }

        // At N=10, DIM=1024, σ ≈ √9/√1024 ≈ 0.094 — well below the
        // 0.28 phase-transition boundary. Expect 100% recall.
        assert_eq!(correct, N, "expected 100% recall at safe load");

        // (Suppress unused-mut warning on vocab — it's only mut at init.)
        let _ = &mut vocab;
    }

    #[test]
    fn dim_mismatch_errors() {
        let a = array![1.0_f32, 2.0];
        let b = array![1.0_f32, 2.0, 3.0];
        assert!(bind_hlb(a.view(), b.view()).is_err());
        assert!(unbind_hlb(a.view(), b.view()).is_err());
        assert!(cosine_similarity(a.view(), b.view()).is_err());
    }

    #[test]
    fn role_is_bipolar() {
        // Every entry of a random_hlb_role must be exactly ±1, so
        // that role * role = 1 element-wise (the unbind invariant).
        let role = random_hlb_role(256, Some(99));
        for x in role.iter() {
            assert!(*x == 1.0 || *x == -1.0, "non-bipolar entry: {}", x);
        }
    }
}
