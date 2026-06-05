//! Python bindings for the memory-plant Rust core via pyo3.
//!
//! Exposes:
//!   - PyAdaptiveMemory — direct HLB store/retrieve/forget
//!   - PyPersonalMemory — per-user wrapper with regex extraction +
//!                       ingest/recall/forget_all/all_facts API
//!
//! Build (from this directory):
//!   maturin develop --release    # installs into current venv
//!   maturin build --release      # produces .whl in target/wheels/
//!
//! Use from Python:
//!   import memory_plant_rs as mp
//!   am = mp.AdaptiveMemory(dim=1024, vocab_cap=256, seed=42)
//!   am.store("user|works_as", "engineer")
//!   am.retrieve("user|works_as")   # -> "engineer"
//!
//! Performance: hot paths (HLB bind/unbind, AMP refinement, FAISS-
//! style nearest-neighbor) run as native Rust code with no GIL
//! contention. Typically 10-50× faster than the pure-Python
//! reference at the same recall.

use memory_plant::{
    chunk_text, AdaptiveMemory, DocumentMemory, Fact as MpFact, MockEncoder, PersonalMemory,
    RegexExtractor, SearchFilter, SearchHit,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::wrap_pyfunction;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;

/// Convert a memory-plant HlbError into a Python exception.
fn hlb_err(e: memory_plant::HlbError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

// ============================================================
// AdaptiveMemory
// ============================================================

#[pyclass(name = "AdaptiveMemory")]
struct PyAdaptiveMemory {
    inner: AdaptiveMemory,
}

#[pymethods]
impl PyAdaptiveMemory {
    /// Build a new AdaptiveMemory.
    ///
    /// Args:
    ///   dim: HLB dimensionality. 512 is production default,
    ///        1024 gives bigger safe-vocab capacity.
    ///   vocab_cap: maximum distinct values across all stored facts.
    ///   shard_capacity: optional override; None uses dim/13 (safe).
    ///   seed: rng seed for the vocab tensor (reproducible runs).
    #[new]
    #[pyo3(signature = (dim = 512, vocab_cap = 4096, shard_capacity = None, seed = 42))]
    fn new(
        dim: usize,
        vocab_cap: usize,
        shard_capacity: Option<usize>,
        seed: u64,
    ) -> PyResult<Self> {
        let inner = AdaptiveMemory::new(dim, vocab_cap, shard_capacity, seed).map_err(hlb_err)?;
        Ok(Self { inner })
    }

    /// Append a (key, value) fact. Overwrites algebraically on repeat
    /// keys. Returns the vocab index.
    fn store(&mut self, key: &str, value: &str) -> PyResult<usize> {
        self.inner.store(key, value).map_err(hlb_err)
    }

    /// Look up the value previously stored under `key`. Returns None
    /// if no such key.
    fn retrieve(&self, key: &str) -> PyResult<Option<String>> {
        self.inner.retrieve(key).map_err(hlb_err)
    }

    /// Algebraic forget — provable, residual ≈ 0. Returns True if
    /// key was present.
    fn forget(&mut self, key: &str) -> PyResult<bool> {
        self.inner.forget(key).map_err(hlb_err)
    }

    #[getter]
    fn dim(&self) -> usize { self.inner.dim() }

    #[getter]
    fn shard_capacity(&self) -> usize { self.inner.shard_capacity() }

    #[getter]
    fn n_shards(&self) -> usize { self.inner.n_shards() }

    #[getter]
    fn total_facts(&self) -> usize { self.inner.total_facts() }

    fn __repr__(&self) -> String {
        format!(
            "AdaptiveMemory(dim={}, shard_capacity={}, n_shards={}, total_facts={})",
            self.inner.dim(),
            self.inner.shard_capacity(),
            self.inner.n_shards(),
            self.inner.total_facts(),
        )
    }
}

// ============================================================
// PersonalMemory
// ============================================================

#[pyclass(name = "PersonalMemory")]
struct PyPersonalMemory {
    inner: PersonalMemory,
}

#[pymethods]
impl PyPersonalMemory {
    /// PersonalMemory(user_id, dim=512, vocab_cap=4096, seed=42)
    ///
    /// Uses the offline RegexExtractor by default. Anthropic / sampling
    /// extractors are a future addition (will require additional
    /// constructors).
    #[new]
    #[pyo3(signature = (user_id, dim = 512, vocab_cap = 4096, seed = 42))]
    fn new(user_id: String, dim: usize, vocab_cap: usize, seed: u64) -> PyResult<Self> {
        let mp = AdaptiveMemory::new(dim, vocab_cap, None, seed).map_err(hlb_err)?;
        let inner = PersonalMemory::new(user_id, mp, Arc::new(RegexExtractor::new()));
        Ok(Self { inner })
    }

    /// Extract facts from natural-language `message` and store them.
    /// Returns a list of dicts: `[{subject, predicate, obj}, ...]`.
    fn ingest(&mut self, py: Python<'_>, message: &str) -> PyResult<Vec<Py<PyDict>>> {
        let facts = self.inner.ingest(message).map_err(hlb_err)?;
        facts
            .iter()
            .map(|f| fact_to_dict(py, f))
            .collect()
    }

    /// Store a fact directly (no LLM extraction).
    #[pyo3(signature = (predicate, value, subject = "user".to_string()))]
    fn store_fact(&mut self, predicate: &str, value: &str, subject: String) -> PyResult<()> {
        let f = MpFact::new(subject, predicate, value, "direct");
        self.inner.store_fact(&f).map_err(hlb_err)
    }

    /// recall(predicate, subject=None) -> str | None
    #[pyo3(signature = (predicate, subject = None))]
    fn recall(&self, predicate: &str, subject: Option<&str>) -> PyResult<Option<String>> {
        self.inner.recall(predicate, subject).map_err(hlb_err)
    }

    /// Snapshot of every stored fact: {key_suffix: value}.
    fn all_facts(&self) -> PyResult<HashMap<String, String>> {
        self.inner.all_facts().map_err(hlb_err)
    }

    /// Algebraic forget of a single fact.
    #[pyo3(signature = (predicate, subject = None))]
    fn forget(&mut self, predicate: &str, subject: Option<&str>) -> PyResult<bool> {
        self.inner.forget(predicate, subject).map_err(hlb_err)
    }

    /// GDPR Article 17 — total user erasure. Returns count of forgotten facts.
    fn forget_all(&mut self) -> PyResult<usize> {
        self.inner.forget_all().map_err(hlb_err)
    }

    #[getter]
    fn user_id(&self) -> String { self.inner.user_id.clone() }

    fn __repr__(&self) -> String {
        format!(
            "PersonalMemory(user_id={:?}, total_facts={})",
            self.inner.user_id,
            self.inner.mp.total_facts(),
        )
    }
}

fn fact_to_dict(py: Python<'_>, f: &MpFact) -> PyResult<Py<PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("subject", &f.subject)?;
    d.set_item("predicate", &f.predicate)?;
    d.set_item("obj", &f.obj)?;
    Ok(d.unbind())
}

// ============================================================
// DocumentMemory  (RAG: precomputed embeddings + semantic search)
// ============================================================

/// Document/RAG store. The Qwen side computes embeddings (e5 via MLX) and
/// passes them in; this store only does the PRECOMPUTED path + cosine search,
/// so its internal encoder is a no-op `MockEncoder` and is never invoked.
#[pyclass(name = "DocumentMemory")]
struct PyDocumentMemory {
    inner: DocumentMemory<MockEncoder>,
    n_chunks: usize,
}

#[pymethods]
impl PyDocumentMemory {
    /// DocumentMemory(dim=384). dim must match the embedder (multilingual-e5-small = 384).
    #[new]
    #[pyo3(signature = (dim = 384))]
    fn new(dim: usize) -> Self {
        Self { inner: DocumentMemory::new(MockEncoder::new(dim)), n_chunks: 0 }
    }

    /// Store a document: pre-chunked text + matching precomputed embeddings
    /// (chunks[i] <-> embeddings[i]). metadata: optional {str: str}. Returns #chunks.
    #[pyo3(signature = (doc_id, chunks, embeddings, metadata = None))]
    fn add_document(
        &mut self,
        doc_id: String,
        chunks: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        metadata: Option<HashMap<String, String>>,
    ) -> usize {
        let md: HashMap<String, JsonValue> = metadata
            .unwrap_or_default()
            .into_iter()
            .map(|(k, v)| (k, JsonValue::String(v)))
            .collect();
        let added = self
            .inner
            .add_document_with_embeddings(doc_id, chunks, embeddings, md);
        self.n_chunks += added;
        added
    }

    /// Semantic search with a PRECOMPUTED query embedding. Returns a list of
    /// dicts {chunk_id, doc_id, score, text}, best first.
    #[pyo3(signature = (query_emb, k = 5, min_score = None, doc_ids = None))]
    fn search(
        &self,
        py: Python<'_>,
        query_emb: Vec<f32>,
        k: usize,
        min_score: Option<f32>,
        doc_ids: Option<Vec<String>>,
    ) -> PyResult<Vec<Py<PyDict>>> {
        let filter = SearchFilter { min_score, doc_ids, ..Default::default() };
        self.inner
            .search(&query_emb, k, &filter)
            .iter()
            .map(|h| hit_to_dict(py, h))
            .collect()
    }

    /// Persist to disk; reload with DocumentMemory.load(path, dim).
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save_state(path)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Load a previously saved store. dim must match what it was built with.
    #[staticmethod]
    #[pyo3(signature = (path, dim = 384))]
    fn load(path: &str, dim: usize) -> PyResult<Self> {
        let inner = DocumentMemory::load_state(path, MockEncoder::new(dim))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner, n_chunks: 0 })
    }

    #[getter]
    fn n_chunks(&self) -> usize { self.n_chunks }

    fn __repr__(&self) -> String {
        format!("DocumentMemory(n_chunks={})", self.n_chunks)
    }
}

fn hit_to_dict(py: Python<'_>, h: &SearchHit) -> PyResult<Py<PyDict>> {
    let d = PyDict::new_bound(py);
    d.set_item("chunk_id", &h.chunk_id)?;
    d.set_item("doc_id", &h.doc_id)?;
    d.set_item("score", h.score)?;
    d.set_item("text", &h.text)?;
    Ok(d.unbind())
}

/// chunk_text(text, chunk_size=200, chunk_overlap=20) -> list[str]
/// Paragraph/sentence-aware chunking (pure Rust). Pairs with embed(passage).
#[pyfunction]
#[pyo3(name = "chunk_text", signature = (text, chunk_size = 200, chunk_overlap = 20))]
fn chunk_text_py(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    chunk_text(text, chunk_size, chunk_overlap)
}

// ============================================================
// Module entry point
// ============================================================

#[pymodule]
fn memory_plant_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAdaptiveMemory>()?;
    m.add_class::<PyPersonalMemory>()?;
    m.add_class::<PyDocumentMemory>()?;
    m.add_function(wrap_pyfunction!(chunk_text_py, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
