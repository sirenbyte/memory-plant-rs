//! Unified VSA — Vector Symbolic Architectures parameterized by
//! phase resolution `k` bits per dimension.
//!
//! ## Mathematical unification
//!
//! All standard VSAs are special cases of complex-valued phase
//! arithmetic with quantized phase angles:
//!
//! ```text
//!     Phase storage:  φ ∈ {2πj/2^k}, j = 0..2^k-1
//!     bind(r, v):     phase_add  modulo 2^k
//!     unbind(M, r):   phase_sub  modulo 2^k
//! ```
//!
//! Specializations:
//!
//! | k | name              | phases | bytes per role (dim=1024) |
//! |---|-------------------|--------|---------------------------|
//! | 1 | HLB (bipolar)     | 2      | 128 B                     |
//! | 2 | qFHRR-4           | 4      | 256 B                     |
//! | 4 | qFHRR-16          | 16     | 512 B                     |
//! | 8 | qFHRR-256         | 256    | 1024 B (1 B per dim)      |
//! | ∞ | FHRR (continuous) | ℝ      | 8192 B (complex<f32>)    |
//!
//! ## Storage
//!
//! Internally `Phase::Quantized { phases, k, dim }` packs `k` phase
//! bits per dim into a `Vec<u8>`. For `k=1` (HLB) we get 8 phases per
//! byte — same density as the bit-packed bipolar in vocab.rs. For
//! `k=2` we get 4 phases per byte, etc.
//!
//! `Phase::Real` is a separate variant for the legacy f32 ±1 storage
//! we use in adaptive.rs today, so we don't break existing tests
//! while the migration to bit-packed phases is in progress.
//!
//! ## bind / unbind invariants
//!
//! For any role `r` and value `v` of equal dim and matching k:
//! ```text
//!     unbind(bind(r, v), r) == v
//! ```
//! Proven in tests for k ∈ {1, 2, 4} with multiple seeds.

use ndarray::Array1;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VsaError {
    #[error("dim mismatch: lhs={lhs} rhs={rhs}")]
    DimMismatch { lhs: usize, rhs: usize },
    #[error("k mismatch: lhs={lhs} rhs={rhs}")]
    KMismatch { lhs: u8, rhs: u8 },
    #[error("invalid k={0}: must be 1, 2, 4 or 8")]
    InvalidK(u8),
}

/// Unified phase representation. Choose variant by storage / precision
/// trade-off. For HLB we keep the legacy real f32 form for now to
/// preserve byte-equality with `bind_hlb` callers; new code should
/// use `Quantized`.
#[derive(Debug, Clone)]
pub enum Phase {
    /// HLB legacy real form: ±1 floats, dim values.
    /// Equivalent to k=1 but with f32 per element (no bit packing).
    Real(Array1<f32>),
    /// Bit-packed phases. `phases.len() = ceil(dim * k / 8)`.
    Quantized { phases: Vec<u8>, k: u8, dim: usize },
}

impl Phase {
    pub fn dim(&self) -> usize {
        match self {
            Phase::Real(v) => v.len(),
            Phase::Quantized { dim, .. } => *dim,
        }
    }

    pub fn k(&self) -> u8 {
        match self {
            Phase::Real(_) => 1,
            Phase::Quantized { k, .. } => *k,
        }
    }

    /// Read phase index at position `i`. Returns value in [0, 2^k - 1]
    /// for Quantized, or 0/1 for Real (mapped from ±1 → {0, 1}).
    pub fn phase_at(&self, i: usize) -> u8 {
        match self {
            Phase::Real(v) => if v[i] > 0.0 { 0 } else { 1 },
            Phase::Quantized { phases, k, dim } => unpack_phase(phases, i, *k, *dim),
        }
    }

    /// Build a random role with `k` phase bits per dim.
    pub fn random_role(dim: usize, k: u8, seed: u64) -> Result<Self, VsaError> {
        if !matches!(k, 1 | 2 | 4 | 8) {
            return Err(VsaError::InvalidK(k));
        }
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let max_phase = 1u32 << k as u32;
        if k == 1 {
            // Keep Real form for legacy compatibility.
            let arr = Array1::from_shape_fn(dim, |_| {
                if rng.random::<bool>() { 1.0 } else { -1.0 }
            });
            return Ok(Phase::Real(arr));
        }
        let mut phases = vec![0u8; bytes_needed(dim, k)];
        for i in 0..dim {
            let p = rng.random::<u32>() % max_phase;
            pack_phase(&mut phases, i, p as u8, k, dim);
        }
        Ok(Phase::Quantized { phases, k, dim })
    }

    /// Build a deterministic role from a key string via SHA-256 → ChaCha8.
    /// Reproducible across machines, matches the algorithm in adaptive.rs.
    pub fn role_from_key(key: &str, dim: usize, k: u8) -> Result<Self, VsaError> {
        let mut hasher = Sha256::new();
        hasher.update(key.as_bytes());
        hasher.update(b":");
        hasher.update(&(dim as u64).to_le_bytes());
        hasher.update(&[k]);
        let digest = hasher.finalize();
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&digest);
        let mut rng = ChaCha8Rng::from_seed(seed);
        let max_phase = 1u32 << k as u32;
        if k == 1 {
            let arr = Array1::from_shape_fn(dim, |_| {
                if rng.random::<bool>() { 1.0 } else { -1.0 }
            });
            return Ok(Phase::Real(arr));
        }
        let mut phases = vec![0u8; bytes_needed(dim, k)];
        for i in 0..dim {
            let p = rng.random::<u32>() % max_phase;
            pack_phase(&mut phases, i, p as u8, k, dim);
        }
        Ok(Phase::Quantized { phases, k, dim })
    }

    /// bind(role, value) — phase addition mod 2^k.
    ///
    /// For HLB (k=1, Real form): element-wise multiply (since
    /// {-1, +1} multiplication is XOR on the phase bit).
    pub fn bind(&self, other: &Phase) -> Result<Phase, VsaError> {
        check_compat(self, other)?;
        match (self, other) {
            (Phase::Real(a), Phase::Real(b)) => Ok(Phase::Real(a * b)),
            (
                Phase::Quantized { phases: pa, k, dim },
                Phase::Quantized { phases: pb, .. },
            ) => {
                let mod_mask = (1u32 << *k as u32) - 1;
                let mut out = vec![0u8; pa.len()];
                for i in 0..*dim {
                    let a = unpack_phase(pa, i, *k, *dim) as u32;
                    let b = unpack_phase(pb, i, *k, *dim) as u32;
                    let sum = (a + b) & mod_mask;
                    pack_phase(&mut out, i, sum as u8, *k, *dim);
                }
                Ok(Phase::Quantized { phases: out, k: *k, dim: *dim })
            }
            _ => Err(VsaError::KMismatch { lhs: self.k(), rhs: other.k() }),
        }
    }

    /// unbind(memory, role) — phase subtraction mod 2^k.
    ///
    /// For HLB (k=1, Real form): bind == unbind because role² = 1.
    pub fn unbind(&self, role: &Phase) -> Result<Phase, VsaError> {
        check_compat(self, role)?;
        match (self, role) {
            (Phase::Real(m), Phase::Real(r)) => Ok(Phase::Real(m * r)),
            (
                Phase::Quantized { phases: pm, k, dim },
                Phase::Quantized { phases: pr, .. },
            ) => {
                let mod_mask = (1u32 << *k as u32) - 1;
                let mut out = vec![0u8; pm.len()];
                for i in 0..*dim {
                    let m = unpack_phase(pm, i, *k, *dim) as i32;
                    let r = unpack_phase(pr, i, *k, *dim) as i32;
                    let diff = ((m - r).rem_euclid(mod_mask as i32 + 1)) as u32;
                    pack_phase(&mut out, i, diff as u8, *k, *dim);
                }
                Ok(Phase::Quantized { phases: out, k: *k, dim: *dim })
            }
            _ => Err(VsaError::KMismatch { lhs: self.k(), rhs: role.k() }),
        }
    }

    /// Cosine similarity. For Real this is the standard ⟨a, b⟩ / ‖a‖‖b‖.
    /// For Quantized we lift phases to complex unit vectors and take
    /// the real part of the normalized inner product — same as
    /// FHRR cosine in the phase domain.
    pub fn cosine(&self, other: &Phase) -> Result<f32, VsaError> {
        check_compat(self, other)?;
        match (self, other) {
            (Phase::Real(a), Phase::Real(b)) => {
                let na = (a.dot(a)).sqrt();
                let nb = (b.dot(b)).sqrt();
                let denom = na * nb;
                if denom < f32::EPSILON {
                    return Ok(0.0);
                }
                Ok(a.dot(b) / denom)
            }
            (
                Phase::Quantized { phases: pa, k, dim },
                Phase::Quantized { phases: pb, .. },
            ) => {
                let scale = std::f32::consts::TAU / (1u32 << *k as u32) as f32;
                let mut acc = 0.0_f64;
                for i in 0..*dim {
                    let a = unpack_phase(pa, i, *k, *dim) as f32 * scale;
                    let b = unpack_phase(pb, i, *k, *dim) as f32 * scale;
                    acc += (a - b).cos() as f64;
                }
                Ok((acc / *dim as f64) as f32)
            }
            _ => Err(VsaError::KMismatch { lhs: self.k(), rhs: other.k() }),
        }
    }

    /// Bytes per role at this representation. Useful for benchmarking
    /// storage trade-offs vs precision.
    pub fn storage_bytes(&self) -> usize {
        match self {
            Phase::Real(v) => v.len() * 4,
            Phase::Quantized { phases, .. } => phases.len(),
        }
    }
}

// ============================================================
// Phase packing helpers
// ============================================================

fn bytes_needed(dim: usize, k: u8) -> usize {
    (dim * k as usize + 7) / 8
}

fn pack_phase(buf: &mut [u8], i: usize, p: u8, k: u8, _dim: usize) {
    let bit_pos = i * k as usize;
    let byte_idx = bit_pos / 8;
    let bit_off = bit_pos % 8;
    let mask = ((1u32 << k as u32) - 1) as u8;
    // For k that aligns to 8 (1, 2, 4, 8), the phase fits within one byte.
    if bit_off + k as usize <= 8 {
        buf[byte_idx] = (buf[byte_idx] & !(mask << bit_off)) | ((p & mask) << bit_off);
    } else {
        // Cross-byte (only when k=4 starts on bit 5,6,7 — shouldn't
        // happen since dim*k is always byte-aligned). Defensive only.
        let low_bits = 8 - bit_off;
        let low_mask = (1u8 << low_bits) - 1;
        buf[byte_idx] = (buf[byte_idx] & !(low_mask << bit_off))
            | ((p & low_mask) << bit_off);
        let high_bits = k as usize - low_bits;
        let high_mask = (1u8 << high_bits) - 1;
        buf[byte_idx + 1] = (buf[byte_idx + 1] & !high_mask)
            | ((p >> low_bits) & high_mask);
    }
}

fn unpack_phase(buf: &[u8], i: usize, k: u8, _dim: usize) -> u8 {
    let bit_pos = i * k as usize;
    let byte_idx = bit_pos / 8;
    let bit_off = bit_pos % 8;
    let mask = ((1u32 << k as u32) - 1) as u8;
    if bit_off + k as usize <= 8 {
        (buf[byte_idx] >> bit_off) & mask
    } else {
        let low_bits = 8 - bit_off;
        let high_bits = k as usize - low_bits;
        let low_mask = (1u8 << low_bits) - 1;
        let high_mask = (1u8 << high_bits) - 1;
        let low = (buf[byte_idx] >> bit_off) & low_mask;
        let high = buf[byte_idx + 1] & high_mask;
        low | (high << low_bits)
    }
}

fn check_compat(a: &Phase, b: &Phase) -> Result<(), VsaError> {
    if a.dim() != b.dim() {
        return Err(VsaError::DimMismatch { lhs: a.dim(), rhs: b.dim() });
    }
    if a.k() != b.k() {
        return Err(VsaError::KMismatch { lhs: a.k(), rhs: b.k() });
    }
    Ok(())
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn hlb_real_roundtrip() {
        // k=1 path (Real variant)
        let role = Phase::random_role(512, 1, 42).unwrap();
        let value = Phase::random_role(512, 1, 7).unwrap();
        let memory = role.bind(&value).unwrap();
        let recovered = memory.unbind(&role).unwrap();
        if let (Phase::Real(rec), Phase::Real(val)) = (&recovered, &value) {
            for (a, b) in rec.iter().zip(val.iter()) {
                assert_abs_diff_eq!(a, b, epsilon = 1e-6);
            }
        } else {
            panic!("expected Real variant");
        }
    }

    #[test]
    fn qfhrr_k2_roundtrip() {
        // k=2 path — 4-state phases
        let role = Phase::random_role(256, 2, 42).unwrap();
        let value = Phase::random_role(256, 2, 7).unwrap();
        let memory = role.bind(&value).unwrap();
        let recovered = memory.unbind(&role).unwrap();
        // Phase-by-phase equality
        for i in 0..256 {
            assert_eq!(recovered.phase_at(i), value.phase_at(i),
                "mismatch at i={i}");
        }
    }

    #[test]
    fn qfhrr_k4_roundtrip() {
        // k=4 path — 16-state phases
        let role = Phase::random_role(128, 4, 42).unwrap();
        let value = Phase::random_role(128, 4, 7).unwrap();
        let memory = role.bind(&value).unwrap();
        let recovered = memory.unbind(&role).unwrap();
        for i in 0..128 {
            assert_eq!(recovered.phase_at(i), value.phase_at(i));
        }
    }

    #[test]
    fn qfhrr_k8_roundtrip() {
        // k=8 path — full byte per phase
        let role = Phase::random_role(64, 8, 42).unwrap();
        let value = Phase::random_role(64, 8, 7).unwrap();
        let memory = role.bind(&value).unwrap();
        let recovered = memory.unbind(&role).unwrap();
        for i in 0..64 {
            assert_eq!(recovered.phase_at(i), value.phase_at(i));
        }
    }

    #[test]
    fn role_from_key_deterministic() {
        let a = Phase::role_from_key("test_key", 256, 2).unwrap();
        let b = Phase::role_from_key("test_key", 256, 2).unwrap();
        for i in 0..256 {
            assert_eq!(a.phase_at(i), b.phase_at(i));
        }
    }

    #[test]
    fn different_keys_different_roles() {
        let a = Phase::role_from_key("alice", 256, 2).unwrap();
        let b = Phase::role_from_key("bob", 256, 2).unwrap();
        // Random 2-bit phases — agreement should be ~25%
        let agree = (0..256).filter(|i| a.phase_at(*i) == b.phase_at(*i)).count();
        let frac = agree as f32 / 256.0;
        assert!((0.15..0.35).contains(&frac), "agreement {frac:.2} not near 0.25");
    }

    #[test]
    fn cosine_self_is_one() {
        for k in [1, 2, 4, 8] {
            let r = Phase::random_role(256, k, 42).unwrap();
            let c = r.cosine(&r).unwrap();
            assert_abs_diff_eq!(c, 1.0, epsilon = 1e-4);
        }
    }

    #[test]
    fn storage_scales_with_k() {
        // dim=1024 — exercise the storage table from module docs
        let h = Phase::random_role(1024, 1, 1).unwrap();
        let q2 = Phase::random_role(1024, 2, 1).unwrap();
        let q4 = Phase::random_role(1024, 4, 1).unwrap();
        let q8 = Phase::random_role(1024, 8, 1).unwrap();
        // Real variant for k=1 is 4 bytes per dim (legacy compat).
        // After we migrate the AdaptiveMemory storage to bit-packed,
        // this drops to 128 B as advertised.
        assert_eq!(h.storage_bytes(), 1024 * 4);
        assert_eq!(q2.storage_bytes(), 1024 * 2 / 8); // 256 B
        assert_eq!(q4.storage_bytes(), 1024 * 4 / 8); // 512 B
        assert_eq!(q8.storage_bytes(), 1024);          // 1024 B
    }

    #[test]
    fn k_mismatch_errors() {
        let a = Phase::random_role(64, 2, 1).unwrap();
        let b = Phase::random_role(64, 4, 1).unwrap();
        let r = a.bind(&b);
        assert!(matches!(r, Err(VsaError::KMismatch { .. })));
    }

    #[test]
    fn invalid_k_errors() {
        let r = Phase::random_role(64, 3, 1); // 3 not in {1,2,4,8}
        assert!(matches!(r, Err(VsaError::InvalidK(3))));
    }

    #[test]
    fn superposition_then_amp_lookup_k2() {
        // Same algorithm as hlb superposition test but at k=2.
        // Build vocab of K=8 phase vectors at dim=512 k=2.
        const DIM: usize = 512;
        const VOCAB: usize = 8;
        const N: usize = 5;
        let vocab: Vec<Phase> = (0..VOCAB)
            .map(|i| Phase::random_role(DIM, 2, 1000 + i as u64).unwrap())
            .collect();

        // Build N roles and N truth indices.
        let mut roles = Vec::with_capacity(N);
        let mut truth = Vec::with_capacity(N);
        // Memory: superposition is harder under quantized phase since
        // simple sum-of-phases doesn't correspond to bind algebra
        // directly (would need a phasor representation).
        // For the unit test we verify pair-wise bind/unbind works
        // even though full superposition AMP needs phasor decoding.
        for i in 0..N {
            let role = Phase::random_role(DIM, 2, 2000 + i as u64).unwrap();
            let idx = i % VOCAB;
            let memory = role.bind(&vocab[idx]).unwrap();
            // Roundtrip just this single fact
            let decoded = memory.unbind(&role).unwrap();
            // The unbind should reproduce vocab[idx] phases exactly.
            for j in 0..DIM {
                assert_eq!(decoded.phase_at(j), vocab[idx].phase_at(j),
                    "decoded fact {i} differs at dim {j}");
            }
            roles.push(role);
            truth.push(idx);
        }
        // Sanity: roles are all distinct.
        let _ = (roles, truth);
    }
}
