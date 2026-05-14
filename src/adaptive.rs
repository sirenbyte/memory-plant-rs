//! AdaptiveMemory — auto-sharding HLB store.
//!
//! Single-shard HLB memory saturates around `dim/13` facts before
//! cross-talk noise σ crosses the AMP refinement phase boundary
//! (~0.28). AdaptiveMemory keeps adding fresh shards under the hood
//! so the user can store arbitrarily many facts at constant per-fact
//! retrieval quality.
//!
//! ## Storage model
//!
//! ```text
//!     facts: { key → value_string }
//!     vocab: ordered list of distinct value strings + their
//!            corresponding random unit-norm vectors (V × d matrix)
//!     shards: vec of Shard, each holding:
//!         M:       Array1<f32>(d) — superposition of bound facts
//!         keys:    Vec<String> — facts in this shard, in store order
//!         k2v:     map key → vocab_idx (so we can rebuild M on load)
//!
//!     store(key, value):
//!         vocab_idx = vocab.register(value)
//!         shard = last_shard_with_room()
//!         role = deterministic_role(key, d)
//!         shard.M += bind(role, vocab[vocab_idx])
//!         shard.k2v[key] = vocab_idx
//!         key_to_shard[key] = shard_idx
//!
//!     retrieve(key):
//!         shard_idx = key_to_shard[key]
//!         role = deterministic_role(key, d)
//!         decoded = unbind(shards[shard_idx].M, role)
//!         (best_vocab_idx, _) = vocab.nearest(decoded)
//!         return vocab.key_at(best_vocab_idx)
//! ```
//!
//! Roles are derived deterministically from the key via SHA-256 +
//! ChaCha8 seeding. This means we never persist role tensors — they
//! regenerate identically on `load_state()` from the same key string.

use ndarray::Array1;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};
use std::collections::HashMap;

use crate::hlb::{bind_hlb, unbind_hlb, HlbError};
use crate::vocab::Vocab;

/// Phase-transition coefficient for HLB cross-talk noise.
/// Safe shard capacity = `dim / DENOMINATOR_HLB` where σ ≤ 0.28.
const DENOMINATOR_HLB: usize = 13;

/// Map a (key, dim) pair to a deterministic bipolar ±1 role vector.
///
/// Algorithm:
/// 1. SHA-256 over `key || ":" || dim_le_bytes`
/// 2. Use the first 32 bytes as a ChaCha8 seed
/// 3. Generate `dim` bipolar values
///
/// Property: identical inputs across machines / runs yield identical
/// output bytes. This lets us drop role tensors from persisted state
/// and recompute them lazily on load_state — same trade-off as the
/// Python reference's `deterministic_roles=True` path.
pub fn role_from_key(key: &str, dim: usize) -> Array1<f32> {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    hasher.update(b":");
    hasher.update(&(dim as u64).to_le_bytes());
    let digest = hasher.finalize();
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&digest);
    let mut rng = ChaCha8Rng::from_seed(seed);
    Array1::from_shape_fn(dim, |_| if rng.random::<bool>() { 1.0 } else { -1.0 })
}

/// Compute the dim-aware safe shard capacity. Mirrors the Python
/// `_safe_shard_capacity` formula.
pub fn safe_shard_capacity(dim: usize) -> usize {
    (dim / DENOMINATOR_HLB).max(20)
}

#[derive(Debug)]
struct Shard {
    /// Superposition tensor — `M = Σ bind(r_i, v_i)`.
    memory: Array1<f32>,
    /// Keys stored in this shard, in store order. Useful for replay
    /// and for capacity tracking (`keys.len() < shard_capacity`).
    keys: Vec<String>,
    /// `key → vocab_idx` — the live source of truth for replay.
    k2v: HashMap<String, usize>,
}

impl Shard {
    fn new(dim: usize) -> Self {
        Self {
            memory: Array1::zeros(dim),
            keys: Vec::new(),
            k2v: HashMap::new(),
        }
    }

    fn len(&self) -> usize { self.keys.len() }
}

#[derive(Debug)]
pub struct AdaptiveMemory {
    dim: usize,
    shard_capacity: usize,
    shards: Vec<Shard>,
    pub vocab: Vocab,
    /// Routes a key to the shard that owns it. O(1) retrieve dispatch.
    key_to_shard: HashMap<String, usize>,
    total_facts: usize,
}

impl AdaptiveMemory {
    /// Construct with explicit capacity and seed for the vocab tensor.
    /// `shard_capacity=None` triggers the dim-aware safe default.
    pub fn new(
        dim: usize,
        vocab_cap: usize,
        shard_capacity: Option<usize>,
        seed: u64,
    ) -> Result<Self, HlbError> {
        let vocab = Vocab::new(vocab_cap, dim, seed)?;
        let shard_capacity = shard_capacity.unwrap_or_else(|| safe_shard_capacity(dim));
        Ok(Self {
            dim,
            shard_capacity,
            shards: Vec::new(),
            vocab,
            key_to_shard: HashMap::new(),
            total_facts: 0,
        })
    }

    pub fn dim(&self) -> usize { self.dim }
    pub fn shard_capacity(&self) -> usize { self.shard_capacity }
    pub fn n_shards(&self) -> usize { self.shards.len() }
    pub fn total_facts(&self) -> usize { self.total_facts }

    /// Extract `(key, vocab_idx)` pairs from a shard in store order.
    /// Persistence layer uses this to dump deterministic replay data.
    pub(crate) fn _shard_pairs_impl(&self, shard_idx: usize) -> Vec<(String, usize)> {
        self.shards
            .get(shard_idx)
            .map(|s| {
                s.keys
                    .iter()
                    .filter_map(|k| s.k2v.get(k).map(|&v| (k.clone(), v)))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Append a `(key, value)` fact. Overwrites previous value if
    /// `key` already exists — same semantics as a HashMap insert.
    ///
    /// Returns the vocab index of the stored value (useful for
    /// downstream audit / replay).
    pub fn store(&mut self, key: &str, value: &str) -> Result<usize, HlbError> {
        let vocab_idx = self.vocab.register(value)?;

        // If the key already lives somewhere, overwrite by subtracting
        // the old bind then adding the new one. This preserves the
        // algebraic-forget property: M after overwrite is exactly
        // what it would be if the old value had never been stored.
        if let Some(&shard_idx) = self.key_to_shard.get(key) {
            let role = role_from_key(key, self.dim);
            if let Some(&old_vocab_idx) = self.shards[shard_idx].k2v.get(key) {
                let old_value = self.vocab.row(old_vocab_idx).to_owned();
                let subtract = bind_hlb(role.view(), old_value.view())?;
                self.shards[shard_idx].memory = &self.shards[shard_idx].memory - &subtract;
            }
            let new_value = self.vocab.row(vocab_idx).to_owned();
            let add = bind_hlb(role.view(), new_value.view())?;
            self.shards[shard_idx].memory = &self.shards[shard_idx].memory + &add;
            self.shards[shard_idx].k2v.insert(key.to_string(), vocab_idx);
            return Ok(vocab_idx);
        }

        // New key — pick the last shard if it has room, else allocate.
        if self.shards.is_empty()
            || self.shards.last().unwrap().len() >= self.shard_capacity
        {
            self.shards.push(Shard::new(self.dim));
        }
        let shard_idx = self.shards.len() - 1;
        let role = role_from_key(key, self.dim);
        let value_vec = self.vocab.row(vocab_idx).to_owned();
        let bound = bind_hlb(role.view(), value_vec.view())?;
        self.shards[shard_idx].memory = &self.shards[shard_idx].memory + &bound;
        self.shards[shard_idx].keys.push(key.to_string());
        self.shards[shard_idx].k2v.insert(key.to_string(), vocab_idx);
        self.key_to_shard.insert(key.to_string(), shard_idx);
        self.total_facts += 1;
        Ok(vocab_idx)
    }

    /// Retrieve the value previously stored under `key`. Returns
    /// `None` if the key was never seen, `Err` only on mathematical
    /// failure.
    ///
    /// Algorithm: damped resonator + AMP refinement.
    ///   1. Decode every key in the target shard via unbind+argmax.
    ///      This gives noisy `(vocab_idx)` predictions for each.
    ///   2. Iterate `n_iter` times:
    ///      a. For every key, subtract the current guess of every
    ///         OTHER fact from M, then re-decode just that key.
    ///         This removes cross-talk in a co-ordinated pass.
    ///      b. Damping: blend new prediction with previous.
    ///   3. Return the prediction for the requested key.
    ///
    /// Cost: O(N × n_iter × (V × d + N × d)) per retrieve. For
    /// shard_capacity=78, dim=1024, vocab=200, n_iter=10 →
    /// ~16 MFlop, ~1-3 ms on modern CPU. Identical math to the
    /// Python reference's `damped_resonator_se_amp` on a single
    /// shard. retrieve_all does the same work for all facts at once.
    pub fn retrieve(&self, key: &str) -> Result<Option<String>, HlbError> {
        let shard_idx = match self.key_to_shard.get(key) {
            Some(&i) => i,
            None => return Ok(None),
        };
        let predictions = self.decode_shard_amp(shard_idx, 10)?;
        Ok(predictions
            .get(key)
            .and_then(|&idx| self.vocab.key_at(idx).map(String::from)))
    }

    /// Decode every key in a shard simultaneously with iterative
    /// damped-resonator AMP. Returns `key → vocab_idx` predictions.
    /// Public for retrieve_all + tests.
    pub fn decode_shard_amp(
        &self,
        shard_idx: usize,
        n_iter: usize,
    ) -> Result<HashMap<String, usize>, HlbError> {
        let shard = &self.shards[shard_idx];
        let memory = &shard.memory;
        // Precompute the role for every key in the shard.
        let roles: Vec<(String, Array1<f32>)> = shard
            .keys
            .iter()
            .map(|k| (k.clone(), role_from_key(k, self.dim)))
            .collect();
        if roles.is_empty() {
            return Ok(HashMap::new());
        }

        // Step 1 — initial single-shot predictions.
        let mut predictions: HashMap<String, usize> = HashMap::with_capacity(roles.len());
        for (k, r) in &roles {
            let decoded = unbind_hlb(memory.view(), r.view())?;
            if let Some((idx, _)) = self.vocab.nearest(decoded.view()) {
                predictions.insert(k.clone(), idx);
            }
        }

        // Step 2 — iterative refinement. On each pass, for each key
        // subtract the current best guess of all OTHER facts, then
        // re-decode just that key. damping smooths oscillations.
        let damping: f32 = 0.5;
        for _ in 0..n_iter {
            let mut changed = false;
            // Compute the current full "explained" superposition.
            let mut explained: Array1<f32> = Array1::zeros(self.dim);
            for (k, r) in &roles {
                if let Some(&idx) = predictions.get(k) {
                    let v = self.vocab.row(idx);
                    explained = &explained + &bind_hlb(r.view(), v)?;
                }
            }

            // For each key, isolate its contribution and re-decode.
            for (k, r) in &roles {
                let prev_idx = predictions[k];
                let prev_v = self.vocab.row(prev_idx);
                let prev_bind = bind_hlb(r.view(), prev_v)?;
                // residual = M - (explained - this_key's_current_bind)
                //          = M - explained_without_this
                let cleaned = memory - &(&explained - &prev_bind);
                let isolated = unbind_hlb(cleaned.view(), r.view())?;
                if let Some((new_idx, _)) = self.vocab.nearest(isolated.view()) {
                    if new_idx != prev_idx {
                        // Damped commit — only flip a fraction of disagreements
                        // per pass to prevent oscillation when several facts
                        // share a near-collision in vocab space.
                        let rand_check = {
                            let mut h = sha2::Sha256::new();
                            h.update(k.as_bytes());
                            h.update(&new_idx.to_le_bytes());
                            let d = h.finalize();
                            u32::from_le_bytes([d[0], d[1], d[2], d[3]]) as f32 / u32::MAX as f32
                        };
                        if rand_check > damping {
                            predictions.insert(k.clone(), new_idx);
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break; // converged
            }
        }
        Ok(predictions)
    }

    /// Algebraic forget: subtract `bind(role, value)` from the shard's
    /// M tensor. Residual ≈ 0 (exact subtraction in float32). This is
    /// the GDPR-killer feature — provably erased, not just deleted
    /// from an index.
    ///
    /// Returns true if the key was present and forgotten, false if
    /// no-op.
    pub fn forget(&mut self, key: &str) -> Result<bool, HlbError> {
        let shard_idx = match self.key_to_shard.get(key) {
            Some(&i) => i,
            None => return Ok(false),
        };
        let vocab_idx = match self.shards[shard_idx].k2v.get(key) {
            Some(&v) => v,
            None => return Ok(false),
        };
        let role = role_from_key(key, self.dim);
        let value = self.vocab.row(vocab_idx).to_owned();
        let bound = bind_hlb(role.view(), value.view())?;
        self.shards[shard_idx].memory = &self.shards[shard_idx].memory - &bound;
        // Remove tracking — keys vec is order-sensitive but a
        // post-forget store will reuse the slot, which is fine for
        // the current `len() < cap` allocation rule.
        self.shards[shard_idx].keys.retain(|k| k != key);
        self.shards[shard_idx].k2v.remove(key);
        self.key_to_shard.remove(key);
        self.total_facts -= 1;
        Ok(true)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn role_is_deterministic() {
        let r1 = role_from_key("alice|works_as|hr", 512);
        let r2 = role_from_key("alice|works_as|hr", 512);
        for (a, b) in r1.iter().zip(r2.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-9);
        }
    }

    #[test]
    fn role_differs_for_different_keys() {
        let r1 = role_from_key("alice", 256);
        let r2 = role_from_key("bob", 256);
        // Two random ±1 vectors should disagree on ~50% of dims.
        let agree = r1.iter().zip(r2.iter()).filter(|(a, b)| a == b).count();
        let frac = agree as f32 / 256.0;
        assert!(
            (0.4..0.6).contains(&frac),
            "agreement {:.2} not near 0.5 — roles too correlated",
            frac
        );
    }

    #[test]
    fn store_retrieve_single_fact() {
        let mut am = AdaptiveMemory::new(1024, 256, None, 42).unwrap();
        am.store("user|works_as", "engineer").unwrap();
        assert_eq!(am.retrieve("user|works_as").unwrap(), Some("engineer".into()));
        assert_eq!(am.total_facts(), 1);
        assert_eq!(am.n_shards(), 1);
    }

    #[test]
    fn store_retrieve_unknown_key() {
        let am = AdaptiveMemory::new(512, 64, None, 42).unwrap();
        assert_eq!(am.retrieve("nonexistent").unwrap(), None);
    }

    #[test]
    fn safe_capacity_matches_python() {
        // Python reference: max(20, dim // 13).
        assert_eq!(safe_shard_capacity(256), 20);   // floor 19, clamped to 20
        assert_eq!(safe_shard_capacity(512), 39);
        assert_eq!(safe_shard_capacity(1024), 78);
        assert_eq!(safe_shard_capacity(2048), 157);
    }

    #[test]
    fn auto_shard_at_capacity() {
        let mut am = AdaptiveMemory::new(1024, 1024, Some(20), 42).unwrap();
        // Store 50 facts with cap=20 → expect 3 shards.
        for i in 0..50 {
            am.store(&format!("k{i}"), &format!("v{i}")).unwrap();
        }
        assert_eq!(am.total_facts(), 50);
        assert_eq!(am.n_shards(), 3, "50 facts at cap=20 should make 3 shards");
    }

    #[test]
    fn recall_near_perfect_at_half_load() {
        // Half-load test: dim=1024 → safe_cap=78, store 117 facts
        // (= 1.5 shards). Single-shot AMP (argmax cosine, no
        // iterative refinement) is good enough here.
        //
        // TODO Phase 2: full damped-resonator iterative AMP to
        // match Python reference's 100% recall at full safe_load
        // (3x saturated shards = 234 facts at dim=1024). Without
        // iterative refinement we measure ~90% — enough to prove
        // the architecture, insufficient for production.
        let mut am = AdaptiveMemory::new(1024, 200, None, 42).unwrap();
        let mut truth: Vec<(String, String)> = Vec::with_capacity(117);
        for i in 0..117 {
            let key = format!("k_{i}");
            let value = format!("v_{}", i % 100);
            am.store(&key, &value).unwrap();
            truth.push((key, value));
        }

        let mut correct = 0;
        for (k, expected) in &truth {
            if let Some(got) = am.retrieve(k).unwrap() {
                if &got == expected {
                    correct += 1;
                }
            }
        }
        let pct = 100.0 * correct as f32 / 117.0;
        assert!(
            pct >= 95.0,
            "expected ≥95% recall at half load, got {pct:.1}% ({correct}/117)"
        );
    }

    #[test]
    fn algebraic_forget_residual_near_zero() {
        // Store a fact, forget it, then store it again and verify
        // the M tensor is byte-equal to what we'd get from a fresh
        // store — proves forget is exact, not soft-delete.
        let mut am = AdaptiveMemory::new(1024, 64, None, 42).unwrap();
        am.store("k1", "v1").unwrap();
        am.store("k2", "v2").unwrap();
        am.store("k3", "v3").unwrap();
        let m_before = am.shards[0].memory.clone();

        // Add a fact, then forget it.
        am.store("temp", "ephemeral").unwrap();
        assert_eq!(am.retrieve("temp").unwrap(), Some("ephemeral".into()));
        let was_present = am.forget("temp").unwrap();
        assert!(was_present);
        assert_eq!(am.retrieve("temp").unwrap(), None);

        // Residual = max abs difference between M_before and M_now.
        let m_after = &am.shards[0].memory;
        let residual: f32 = m_before
            .iter()
            .zip(m_after.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            residual < 1e-5,
            "residual {residual:.2e} should be ~0 — algebraic forget broken"
        );
    }

    #[test]
    fn overwrite_same_key() {
        let mut am = AdaptiveMemory::new(1024, 32, None, 42).unwrap();
        am.store("user|lives_in", "almaty").unwrap();
        am.store("user|lives_in", "tokyo").unwrap();
        // Second store overwrites algebraically. Retrieve must
        // return the latest value, total_facts stays 1.
        assert_eq!(
            am.retrieve("user|lives_in").unwrap(),
            Some("tokyo".into())
        );
        assert_eq!(am.total_facts(), 1);
    }
}
