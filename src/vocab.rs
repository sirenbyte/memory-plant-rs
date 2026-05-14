//! Vocabulary — the set of values that can be stored.
//!
//! Memory Plant stores `(key, value)` pairs where every distinct
//! `value` is an entry in a pre-allocated vocab tensor. New values
//! are appended up to `cap`; AMP retrieval projects a noisy unbound
//! vector onto the nearest vocab vector.
//!
//! Storage layout: `tensor: Array2<f32>` of shape `(cap, dim)`. The
//! first `len()` rows are populated; trailing rows are reserved space
//! that grows on `register()`. This matches the Python reference's
//! `_vocab_tensor` slab + `_vocab_keys` list pattern.

use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use std::collections::HashMap;

use crate::hlb::{normalize, HlbError};

#[derive(Debug)]
pub struct Vocab {
    pub tensor: Array2<f32>,
    keys: Vec<String>,
    index: HashMap<String, usize>,
    cap: usize,
    dim: usize,
}

impl Vocab {
    /// Build a vocab with random unit-norm rows. Seedable for tests.
    pub fn new(cap: usize, dim: usize, seed: u64) -> Result<Self, HlbError> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut tensor = Array2::<f32>::zeros((cap, dim));
        for i in 0..cap {
            let mut row = Array1::from_shape_fn(dim, |_| {
                // standard normal via Box-Muller wouldn't add value
                // for our purposes — uniform [-1, 1] then normalize
                // gives identical statistical behavior for AMP.
                rng.random::<f32>() * 2.0 - 1.0
            });
            normalize(&mut row)?;
            tensor.row_mut(i).assign(&row);
        }
        Ok(Self {
            tensor,
            keys: Vec::with_capacity(cap),
            index: HashMap::with_capacity(cap),
            cap,
            dim,
        })
    }

    pub fn dim(&self) -> usize { self.dim }
    pub fn cap(&self) -> usize { self.cap }
    pub fn len(&self) -> usize { self.keys.len() }
    pub fn is_empty(&self) -> bool { self.keys.is_empty() }

    /// Look up the index of a previously-registered value. None if
    /// unknown (caller should `register()` first).
    pub fn get(&self, value: &str) -> Option<usize> {
        self.index.get(value).copied()
    }

    /// Get the row vector for a vocab index.
    pub fn row(&self, idx: usize) -> ArrayView1<'_, f32> {
        self.tensor.row(idx)
    }

    /// Get the value string at an index (for AMP retrieve → string).
    pub fn key_at(&self, idx: usize) -> Option<&str> {
        self.keys.get(idx).map(String::as_str)
    }

    /// Append a new value to the vocab. Returns its index, or
    /// `VocabFull` if at capacity. Idempotent: if `value` is already
    /// registered, returns its existing index without growing.
    pub fn register(&mut self, value: &str) -> Result<usize, HlbError> {
        if let Some(&idx) = self.index.get(value) {
            return Ok(idx);
        }
        if self.keys.len() >= self.cap {
            return Err(HlbError::VocabFull {
                cap: self.cap,
                attempted: value.into(),
            });
        }
        let idx = self.keys.len();
        self.index.insert(value.to_string(), idx);
        self.keys.push(value.to_string());
        Ok(idx)
    }

    /// AMP refinement: cosine-similarity argmax over the live vocab
    /// rows (rows beyond `len()` are still random noise and would
    /// distort the argmax). Returns `(idx, score)`. O(V × d).
    pub fn nearest(&self, query: ArrayView1<f32>) -> Option<(usize, f32)> {
        if self.is_empty() {
            return None;
        }
        // Slice off the unused rows so they don't pollute argmax.
        let live = self.tensor.slice(ndarray::s![..self.keys.len(), ..]);
        let q_norm = query.dot(&query).sqrt();
        if q_norm < f32::EPSILON {
            return Some((0, 0.0));
        }
        let mut best_idx = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        // ndarray's row iterator + dot is cache-friendly. For huge
        // vocabs SIMD via matmul would help but adds a dep; keep
        // this minimal-deps path for Phase 1.
        for (i, row) in live.rows().into_iter().enumerate() {
            let r_norm = row.dot(&row).sqrt();
            if r_norm < f32::EPSILON {
                continue;
            }
            let score = row.dot(&query) / (q_norm * r_norm);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }
        Some((best_idx, best_score))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_idempotent() {
        let mut v = Vocab::new(8, 128, 42).unwrap();
        let i1 = v.register("apple").unwrap();
        let i2 = v.register("apple").unwrap();
        assert_eq!(i1, i2);
        assert_eq!(v.len(), 1);
    }

    #[test]
    fn register_appends_distinct() {
        let mut v = Vocab::new(8, 128, 42).unwrap();
        v.register("apple").unwrap();
        v.register("banana").unwrap();
        v.register("cherry").unwrap();
        assert_eq!(v.len(), 3);
        assert_eq!(v.get("banana"), Some(1));
    }

    #[test]
    fn register_full_errors() {
        let mut v = Vocab::new(2, 64, 42).unwrap();
        v.register("a").unwrap();
        v.register("b").unwrap();
        let result = v.register("c");
        assert!(matches!(result, Err(HlbError::VocabFull { .. })));
    }

    #[test]
    fn nearest_recovers_self() {
        let mut v = Vocab::new(4, 256, 42).unwrap();
        v.register("alpha").unwrap();
        v.register("beta").unwrap();
        v.register("gamma").unwrap();
        // Query with one of the actual vocab rows — must be self.
        let row = v.row(1).to_owned();
        let (idx, score) = v.nearest(row.view()).unwrap();
        assert_eq!(idx, 1);
        assert!(score > 0.99, "self-cosine should be ~1, got {score}");
    }

    #[test]
    fn nearest_empty_is_none() {
        let v = Vocab::new(4, 64, 42).unwrap();
        let q = ndarray::Array1::<f32>::zeros(64);
        assert!(v.nearest(q.view()).is_none());
    }
}
