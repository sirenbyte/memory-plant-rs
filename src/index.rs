//! P3: pluggable vector-index seam for on-device semantic search.
//!
//! `BruteForceIndex` (exact, always available) is the right default at mp's scale
//! — the Python bench shows brute-force is encoder-dominated (~46µs/query at 2.5k).
//! For larger on-device corpora, `HnswIndex` (pure-Rust `instant-distance`, behind
//! `--features ann`) gives sub-linear ANN with no C/C++ deps (wasm-clean).
//! `document.rs` can adopt this trait to swap exact↔ANN behind one seam; the
//! algebraic-forget path maps to `rebuild` (mp rebuilds its index on forget).

/// Vector index over (id, embedding) pairs. Cosine similarity; higher = closer.
pub trait VectorIndex: Send + Sync {
    fn add(&mut self, id: u64, vector: Vec<f32>);
    /// Top-k by cosine similarity, descending: (id, similarity).
    fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)>;
    /// Reset and rebuild from scratch (the forget / compaction path).
    fn rebuild(&mut self, items: Vec<(u64, Vec<f32>)>);
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

fn brute_topk(items: &[(u64, Vec<f32>)], query: &[f32], k: usize) -> Vec<(u64, f32)> {
    let mut scored: Vec<(u64, f32)> =
        items.iter().map(|(id, v)| (*id, cosine(query, v))).collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(k);
    scored
}

/// Exact O(N) index — the default for mp's scale.
#[derive(Default)]
pub struct BruteForceIndex {
    items: Vec<(u64, Vec<f32>)>,
}

impl BruteForceIndex {
    pub fn new() -> Self {
        Self::default()
    }
}

impl VectorIndex for BruteForceIndex {
    fn add(&mut self, id: u64, vector: Vec<f32>) {
        self.items.push((id, vector));
    }
    fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        brute_topk(&self.items, query, k)
    }
    fn rebuild(&mut self, items: Vec<(u64, Vec<f32>)>) {
        self.items = items;
    }
    fn len(&self) -> usize {
        self.items.len()
    }
}

// ============================================================
// Pure-Rust HNSW via instant-distance (feature `ann`)
// ============================================================
#[cfg(feature = "ann")]
mod hnsw {
    use super::{brute_topk, VectorIndex};
    use instant_distance::{Builder, HnswMap, Point, Search};

    #[derive(Clone)]
    pub struct Emb(pub Vec<f32>);

    impl Point for Emb {
        fn distance(&self, other: &Self) -> f32 {
            // cosine distance = 1 - cosine similarity
            let dot: f32 = self.0.iter().zip(&other.0).map(|(a, b)| a * b).sum();
            let na: f32 = self.0.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = other.0.iter().map(|x| x * x).sum::<f32>().sqrt();
            if na == 0.0 || nb == 0.0 { 1.0 } else { 1.0 - dot / (na * nb) }
        }
    }

    /// HNSW index. instant-distance is build-once, so `add` buffers + marks dirty;
    /// the graph is (re)built by `rebuild`. While dirty/unbuilt, `search` falls back
    /// to an exact scan so results are always correct.
    #[derive(Default)]
    pub struct HnswIndex {
        items: Vec<(u64, Vec<f32>)>,
        map: Option<HnswMap<Emb, u64>>,
        dirty: bool,
    }

    impl HnswIndex {
        pub fn new() -> Self {
            Self::default()
        }
    }

    impl VectorIndex for HnswIndex {
        fn add(&mut self, id: u64, vector: Vec<f32>) {
            self.items.push((id, vector));
            self.dirty = true;
        }
        fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
            match &self.map {
                Some(map) if !self.dirty => {
                    let q = Emb(query.to_vec());
                    let mut s = Search::default();
                    map.search(&q, &mut s)
                        .take(k)
                        .map(|item| (*item.value, 1.0 - item.distance))
                        .collect()
                }
                _ => brute_topk(&self.items, query, k), // unbuilt/dirty → exact fallback
            }
        }
        fn rebuild(&mut self, items: Vec<(u64, Vec<f32>)>) {
            self.items = items;
            let pts: Vec<Emb> = self.items.iter().map(|(_, v)| Emb(v.clone())).collect();
            let vals: Vec<u64> = self.items.iter().map(|(id, _)| *id).collect();
            self.map = if pts.is_empty() {
                None
            } else {
                Some(Builder::default().build(pts, vals))
            };
            self.dirty = false;
        }
        fn len(&self) -> usize {
            self.items.len()
        }
    }
}

#[cfg(feature = "ann")]
pub use hnsw::HnswIndex;

#[cfg(test)]
mod tests {
    use super::*;

    fn data() -> Vec<(u64, Vec<f32>)> {
        vec![
            (1, vec![1.0, 0.0, 0.0]),
            (2, vec![0.0, 1.0, 0.0]),
            (3, vec![0.9, 0.1, 0.0]),
        ]
    }

    #[test]
    fn brute_force_exact_topk() {
        let mut idx = BruteForceIndex::new();
        for (id, v) in data() {
            idx.add(id, v);
        }
        let hits = idx.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(idx.len(), 3);
        assert_eq!(hits[0].0, 1); // exact match ranks first
        assert_eq!(hits[1].0, 3); // then the near vector
    }

    #[test]
    fn brute_force_rebuild_drops_old() {
        let mut idx = BruteForceIndex::new();
        idx.rebuild(data());
        idx.rebuild(vec![(2, vec![0.0, 1.0, 0.0])]); // forget path
        let hits = idx.search(&[1.0, 0.0, 0.0], 5);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].0, 2);
    }

    #[cfg(feature = "ann")]
    #[test]
    fn hnsw_finds_exact_match() {
        let mut h = HnswIndex::new();
        h.rebuild(data());
        let hits = h.search(&[1.0, 0.0, 0.0], 1);
        assert_eq!(hits[0].0, 1);
    }
}
