//! DocumentMemory — chunked text with embedding search.
//!
//! Mirrors Python's `vector_memory.DocumentMemory`. Long-form text
//! (audit context, conversation transcripts, document RAG) gets
//! chunked, embedded, and indexed for semantic search.
//!
//! ## Design
//!
//! ```text
//!     add_document(doc_id, text, metadata):
//!         chunks = chunk_text(text, size=200 words, overlap=20)
//!         embeddings = encoder.encode(chunks)
//!         for each (chunk_text, embedding):
//!             store under chunk_id = "{doc_id}#chunk_{i}"
//!
//!     semantic_search(query, k, filter?):
//!         q_emb = encoder.encode([query])[0]
//!         scored = [(chunk_id, cosine(q_emb, emb), text) for chunk in chunks]
//!         return top_k after filter
//! ```
//!
//! ## Encoder abstraction
//!
//! `DocumentMemory<E: Encoder>` is generic over the embedding backend.
//! This decouples the indexing logic from the model — production uses
//! fastembed-rs or ort (ONNX Runtime) via feature flags; tests use a
//! `MockEncoder` that hashes input bytes into a deterministic vector
//! so we can roundtrip without downloading models.

use crate::hlb::cosine_similarity;
use crate::persistence::PersistError;
use ndarray::ArrayView1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Anything that can turn `[String]` into `Vec<Vec<f32>>` of fixed
/// dimensionality. Send + Sync so DocumentMemory is share-able.
pub trait Encoder: Send + Sync {
    /// Encode passages/documents (the storage path).
    fn encode(&self, texts: &[String]) -> Vec<Vec<f32>>;
    /// Encode search queries. Default = same as `encode` (back-compat); e5-style
    /// encoders override to apply the "query: " prefix (vs "passage: " for encode).
    fn encode_query(&self, texts: &[String]) -> Vec<Vec<f32>> {
        self.encode(texts)
    }
    fn dim(&self) -> usize;
}

/// Mock encoder for tests — hashes each text byte into a fixed-dim
/// vector. Deterministic + reproducible without external model files.
pub struct MockEncoder {
    pub dim: usize,
}

impl MockEncoder {
    pub fn new(dim: usize) -> Self { Self { dim } }
}

impl Encoder for MockEncoder {
    fn encode(&self, texts: &[String]) -> Vec<Vec<f32>> {
        texts
            .iter()
            .map(|t| {
                let mut v = vec![0.0f32; self.dim];
                for (i, b) in t.bytes().enumerate() {
                    v[i % self.dim] += (b as f32) - 127.5;
                }
                // L2 normalize
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > f32::EPSILON {
                    v.iter_mut().for_each(|x| *x /= norm);
                }
                v
            })
            .collect()
    }
    fn dim(&self) -> usize { self.dim }
}

// ============================================================
// Production encoder via fastembed-rs (feature-gated)
// ============================================================

/// Pure-Rust sentence-transformer encoder. Downloads AllMiniLM-L6-v2
/// (~30 MB, 384 dims) on first use into a cache directory.
///
/// Enabled by `--features fastembed`. Without it, only MockEncoder
/// exists (suitable for tests but not production semantic search).
///
/// The wrapping layer adapts fastembed's batched-error API to the
/// trait's "return empty vec on failure" contract.
#[cfg(feature = "fastembed")]
pub struct FastembedEncoder {
    model: std::sync::Mutex<fastembed::TextEmbedding>,
    dim: usize,
    e5: bool,   // prepend e5 "query: "/"passage: " prefixes (multilingual ru/kz)
}

#[cfg(feature = "fastembed")]
impl FastembedEncoder {
    /// Initialise with the default AllMiniLM-L6-v2 model. 384-dim,
    /// L2-normalized embeddings out of the box.
    pub fn new() -> Result<Self, fastembed::Error> {
        use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                .with_show_download_progress(false),
        )?;
        Ok(Self {
            model: std::sync::Mutex::new(model),
            dim: 384,
            e5: false,
        })
    }

    /// Use a specific fastembed model (different size / language).
    pub fn with_model(
        model_kind: fastembed::EmbeddingModel,
        dim: usize,
    ) -> Result<Self, fastembed::Error> {
        use fastembed::{InitOptions, TextEmbedding};
        let model = TextEmbedding::try_new(
            InitOptions::new(model_kind).with_show_download_progress(false),
        )?;
        Ok(Self { model: std::sync::Mutex::new(model), dim, e5: false })
    }

    /// Multilingual e5-small (Russian + Kazakh, on-device). 384-dim.
    /// Use embed_query/embed_passage so the e5 prefixes are applied.
    pub fn multilingual() -> Result<Self, fastembed::Error> {
        use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::MultilingualE5Small)
                .with_show_download_progress(false),
        )?;
        Ok(Self { model: std::sync::Mutex::new(model), dim: 384, e5: true })
    }

    /// Raw model call, no prefix. encode()/encode_query() add the e5 prefixes.
    fn encode_raw(&self, texts: Vec<String>) -> Vec<Vec<f32>> {
        match self.model.lock() {
            Ok(mut m) => m.embed(texts, None).unwrap_or_default(),
            Err(_) => Vec::new(),
        }
    }

    /// Embed a single search query (e5 "query: " prefix when multilingual).
    pub fn embed_query(&self, text: &str) -> Vec<f32> {
        self.encode_query(&[text.to_string()]).into_iter().next().unwrap_or_default()
    }

    /// Embed a single stored passage (e5 "passage: " prefix when multilingual).
    pub fn embed_passage(&self, text: &str) -> Vec<f32> {
        self.encode(&[text.to_string()]).into_iter().next().unwrap_or_default()
    }
}

#[cfg(feature = "fastembed")]
impl Encoder for FastembedEncoder {
    fn encode(&self, texts: &[String]) -> Vec<Vec<f32>> {           // passage path
        let t: Vec<String> = if self.e5 {
            texts.iter().map(|s| format!("passage: {s}")).collect()
        } else {
            texts.to_vec()
        };
        self.encode_raw(t)
    }
    fn encode_query(&self, texts: &[String]) -> Vec<Vec<f32>> {     // query path
        let t: Vec<String> = if self.e5 {
            texts.iter().map(|s| format!("query: {s}")).collect()
        } else {
            texts.to_vec()
        };
        self.encode_raw(t)
    }
    fn dim(&self) -> usize { self.dim }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentEntry {
    pub doc_id: String,
    pub metadata: HashMap<String, serde_json::Value>,
    pub n_chunks: usize,
    pub added_ts: f64,
}

/// Stored chunk. Text is gzip-compressed in memory when long enough
/// to benefit (gzip header overhead is ~20 bytes, so short chunks
/// would round-trip larger). Embedding stays uncompressed because
/// FAISS-style cosine search needs raw floats per query.
///
/// Use `Chunk::text()` to materialize the original string on demand.
/// Hot search paths only touch `embedding`; text is decompressed
/// exactly once per returned hit (top-k), not per candidate.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub id: String,
    pub doc_id: String,
    pub idx: usize,
    /// Either raw UTF-8 bytes (`text_compressed = false`) or
    /// gzip-deflate(text) bytes (`text_compressed = true`).
    text_bytes: Vec<u8>,
    text_compressed: bool,
    pub embedding: Vec<f32>,
}

/// Below this length, gzip's ~20-byte header makes compression a net
/// loss. Above it, typical text compresses 2-4× — saves real memory
/// at 10K+ chunks scale.
const COMPRESS_THRESHOLD: usize = 128;

impl Chunk {
    /// Construct, compressing the text if it crosses the threshold.
    fn new(id: String, doc_id: String, idx: usize, text: &str, embedding: Vec<f32>) -> Self {
        let (bytes, compressed) = compress_text(text);
        Self { id, doc_id, idx, text_bytes: bytes, text_compressed: compressed, embedding }
    }

    /// Decompress and return the original chunk text. Called only on
    /// the small set of search hits (typically k=5-20), not on every
    /// candidate. ~5-10 μs per call for ~500-char chunks on modern CPU.
    pub fn text(&self) -> String {
        if !self.text_compressed {
            return String::from_utf8_lossy(&self.text_bytes).into_owned();
        }
        use flate2::read::GzDecoder;
        use std::io::Read;
        let mut dec = GzDecoder::new(self.text_bytes.as_slice());
        let mut out = String::new();
        let _ = dec.read_to_string(&mut out);
        out
    }

    /// Raw stored byte length — useful for memory accounting and
    /// the compression-ratio test.
    pub fn stored_bytes(&self) -> usize { self.text_bytes.len() }

    /// Whether this chunk's text is held in gzip form.
    pub fn is_compressed(&self) -> bool { self.text_compressed }
}

fn compress_text(text: &str) -> (Vec<u8>, bool) {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    if text.len() < COMPRESS_THRESHOLD {
        return (text.as_bytes().to_vec(), false);
    }
    let mut enc = GzEncoder::new(Vec::with_capacity(text.len() / 2), Compression::default());
    if enc.write_all(text.as_bytes()).is_err() {
        return (text.as_bytes().to_vec(), false);
    }
    match enc.finish() {
        Ok(bytes) if bytes.len() < text.len() => (bytes, true),
        _ => (text.as_bytes().to_vec(), false),
    }
}

pub struct DocumentMemory<E: Encoder> {
    encoder: E,
    documents: HashMap<String, DocumentEntry>,
    chunks: Vec<Chunk>,
    chunk_size: usize,
    chunk_overlap: usize,
    /// Optional ANN index (derived from `chunks`). Used only for no-filter
    /// queries; None = exact brute-force scan (the default).
    ann: Option<Box<dyn crate::index::VectorIndex>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub chunk_id: String,
    pub doc_id: String,
    pub score: f32,
    pub text: String,
    /// Owning document's metadata (cloned). Lets callers filter/route on the
    /// hit without a second lookup.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Rich, out-of-box retrieval filter (parity with vector-DB libs like
/// agentmemory). All set conditions are ANDed. An empty/`Default` filter
/// matches everything. Applied on the EXACT scan path (the ANN fast-path is
/// filter-blind and is bypassed whenever any condition is set).
#[derive(Debug, Clone, Default)]
pub struct SearchFilter {
    /// Every entry must equal the owning document's metadata value (exact).
    pub metadata: HashMap<String, serde_json::Value>,
    /// Case-insensitive substring that must appear in the chunk text.
    pub contains_text: Option<String>,
    /// Minimum cosine score in [-1, 1]; hits below are dropped
    /// (equivalent to a `max_distance = 1 - min_score` threshold).
    pub min_score: Option<f32>,
    /// Restrict the search to these document ids (None = all documents).
    pub doc_ids: Option<Vec<String>>,
}

impl SearchFilter {
    /// True when no condition is set — lets `search` take the ANN fast-path.
    pub fn is_empty(&self) -> bool {
        self.metadata.is_empty()
            && self.contains_text.is_none()
            && self.min_score.is_none()
            && self.doc_ids.is_none()
    }

    fn doc_matches(&self, entry: &DocumentEntry) -> bool {
        if let Some(ids) = &self.doc_ids {
            if !ids.iter().any(|id| id == &entry.doc_id) {
                return false;
            }
        }
        self.metadata
            .iter()
            .all(|(k, v)| entry.metadata.get(k) == Some(v))
    }
}

impl<E: Encoder> DocumentMemory<E> {
    pub fn new(encoder: E) -> Self {
        Self {
            encoder,
            documents: HashMap::new(),
            chunks: Vec::new(),
            chunk_size: 200,
            chunk_overlap: 20,
            ann: None,
        }
    }

    pub fn with_chunking(mut self, size: usize, overlap: usize) -> Self {
        self.chunk_size = size;
        self.chunk_overlap = overlap.min(size.saturating_sub(1));
        self
    }

    /// Enable an opt-in ANN index (e.g. `HnswIndex` for large corpora) used for
    /// NO-FILTER queries; filtered queries always use the exact scan. The index
    /// is derived from `chunks` (rebuilt on add/forget) so it can't desync.
    pub fn with_ann(mut self, index: Box<dyn crate::index::VectorIndex>) -> Self {
        self.ann = Some(index);
        self.rebuild_ann();
        self
    }

    /// Rebuild the ANN index from the current chunks (no-op if disabled).
    fn rebuild_ann(&mut self) {
        if self.ann.is_none() {
            return;
        }
        let items: Vec<(u64, Vec<f32>)> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (i as u64, c.embedding.clone()))
            .collect();
        self.ann.as_mut().unwrap().rebuild(items);
    }

    pub fn n_documents(&self) -> usize { self.documents.len() }
    pub fn n_chunks(&self) -> usize { self.chunks.len() }

    /// Index `text` under `doc_id`. Re-adding a doc_id replaces its
    /// chunks (forget + re-add). Returns number of chunks indexed.
    pub fn add_document(
        &mut self,
        doc_id: impl Into<String>,
        text: &str,
        metadata: HashMap<String, serde_json::Value>,
    ) -> usize {
        let doc_id = doc_id.into();
        // Re-add semantics — drop existing chunks for this doc first.
        if self.documents.contains_key(&doc_id) {
            self.forget_document(&doc_id);
        }
        let chunks = chunk_text(text, self.chunk_size, self.chunk_overlap);
        if chunks.is_empty() {
            return 0;
        }
        let embeddings = self.encoder.encode(&chunks);
        let n = chunks.len();
        for (i, (txt, emb)) in chunks.iter().zip(embeddings.iter()).enumerate() {
            let preview: String = txt.chars().take(500).collect();
            self.chunks.push(Chunk::new(
                format!("{doc_id}#chunk_{i}"),
                doc_id.clone(),
                i,
                &preview,
                emb.clone(),
            ));
        }
        self.documents.insert(
            doc_id.clone(),
            DocumentEntry {
                doc_id,
                metadata,
                n_chunks: n,
                added_ts: now(),
            },
        );
        self.rebuild_ann();
        n
    }

    /// Drop every chunk owned by `doc_id`. Returns true if anything
    /// was removed.
    pub fn forget_document(&mut self, doc_id: &str) -> bool {
        let before = self.chunks.len();
        self.chunks.retain(|c| c.doc_id != doc_id);
        self.documents.remove(doc_id);
        let changed = self.chunks.len() != before;
        self.rebuild_ann(); // positions shifted → rebuild derived index
        changed
    }

    /// Build a SearchHit for a chunk, attaching its document's metadata.
    fn make_hit(&self, c: &Chunk, score: f32) -> SearchHit {
        SearchHit {
            chunk_id: c.id.clone(),
            doc_id: c.doc_id.clone(),
            score,
            text: c.text(),
            metadata: self
                .documents
                .get(&c.doc_id)
                .map(|e| e.metadata.clone())
                .unwrap_or_default(),
        }
    }

    /// Top-k cosine-similar chunks. Optional `filter` runs on each
    /// candidate's owning document metadata — None passes through all.
    pub fn semantic_search<F>(
        &self,
        query: &str,
        k: usize,
        filter: Option<F>,
    ) -> Vec<SearchHit>
    where
        F: Fn(&DocumentEntry) -> bool,
    {
        if self.chunks.is_empty() {
            return Vec::new();
        }
        let q_emb = self.encoder.encode_query(&[query.to_string()]);  // e5 "query: " prefix
        if q_emb.is_empty() {
            return Vec::new();
        }
        let q = &q_emb[0];
        // ANN fast-path: enabled index + no metadata filter (the index is
        // filter-blind; filtered queries fall through to the exact scan below).
        if filter.is_none() {
            if let Some(idx) = &self.ann {
                return idx
                    .search(q, k)
                    .into_iter()
                    .filter_map(|(pos, score)| {
                        self.chunks.get(pos as usize).map(|c| self.make_hit(c, score))
                    })
                    .collect();
            }
        }
        let q_view = ArrayView1::from(q.as_slice());

        let mut scored: Vec<(f32, &Chunk)> = self
            .chunks
            .iter()
            .filter(|c| {
                if let Some(f) = &filter {
                    self.documents.get(&c.doc_id).map_or(false, f)
                } else {
                    true
                }
            })
            .filter_map(|c| {
                let cv = ArrayView1::from(c.embedding.as_slice());
                cosine_similarity(q_view, cv).ok().map(|s| (s, c))
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        // Decompress text only for the final top-k. Embeddings did
        // all the heavy lifting up to this point on compressed-text
        // chunks; the actual decompress is amortized across at most
        // `k` hits (typically 5-20).
        scored
            .into_iter()
            .take(k)
            .map(|(score, c)| self.make_hit(c, score))
            .collect()
    }

    /// Add a document whose chunks are ALREADY embedded by the caller — no
    /// internal encoder/ORT is used. This is the on-device path: split a file
    /// with `chunk_text`, embed each chunk with the app's own model (or the
    /// device LLM), then store here. `chunks` and `embeddings` must be the
    /// same length. Re-adding a `doc_id` replaces it. Returns chunks indexed.
    pub fn add_document_with_embeddings(
        &mut self,
        doc_id: impl Into<String>,
        chunks: Vec<String>,
        embeddings: Vec<Vec<f32>>,
        metadata: HashMap<String, serde_json::Value>,
    ) -> usize {
        let doc_id = doc_id.into();
        if chunks.len() != embeddings.len() || chunks.is_empty() {
            return 0;
        }
        if self.documents.contains_key(&doc_id) {
            self.forget_document(&doc_id);
        }
        let n = chunks.len();
        for (i, (txt, emb)) in chunks.iter().zip(embeddings.into_iter()).enumerate() {
            let preview: String = txt.chars().take(500).collect();
            self.chunks.push(Chunk::new(
                format!("{doc_id}#chunk_{i}"),
                doc_id.clone(),
                i,
                &preview,
                emb,
            ));
        }
        self.documents.insert(
            doc_id.clone(),
            DocumentEntry { doc_id, metadata, n_chunks: n, added_ts: now() },
        );
        self.rebuild_ann();
        n
    }

    /// Rich retrieval over a CALLER-SUPPLIED query embedding (no encoder/ORT).
    /// Applies `filter` (metadata eq / contains_text / min_score / doc_ids).
    /// Uses the ANN fast-path only when the filter is empty.
    pub fn search(&self, query_emb: &[f32], k: usize, filter: &SearchFilter) -> Vec<SearchHit> {
        if self.chunks.is_empty() || query_emb.is_empty() {
            return Vec::new();
        }
        // ANN fast-path: no filter conditions only (the index is filter-blind).
        if filter.is_empty() {
            if let Some(idx) = &self.ann {
                return idx
                    .search(query_emb, k)
                    .into_iter()
                    .filter_map(|(pos, score)| {
                        self.chunks.get(pos as usize).map(|c| self.make_hit(c, score))
                    })
                    .collect();
            }
        }
        let needle = filter.contains_text.as_ref().map(|s| s.to_lowercase());
        let q_view = ArrayView1::from(query_emb);
        let mut scored: Vec<(f32, &Chunk)> = self
            .chunks
            .iter()
            .filter(|c| {
                // document-level filters (metadata / doc_ids) — cheap
                self.documents.get(&c.doc_id).map_or(false, |e| filter.doc_matches(e))
            })
            .filter(|c| {
                // chunk-level substring filter — decompress only when requested
                needle.as_ref().map_or(true, |n| c.text().to_lowercase().contains(n))
            })
            .filter_map(|c| {
                let cv = ArrayView1::from(c.embedding.as_slice());
                cosine_similarity(q_view, cv).ok().map(|s| (s, c))
            })
            .filter(|(s, _)| filter.min_score.map_or(true, |m| *s >= m))
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(score, c)| self.make_hit(c, score)).collect()
    }

    pub fn document(&self, doc_id: &str) -> Option<&DocumentEntry> {
        self.documents.get(doc_id)
    }

    /// All document ids currently stored.
    pub fn document_ids(&self) -> Vec<String> {
        self.documents.keys().cloned().collect()
    }
}

// ============================================================
// DocumentMemory persistence (plaintext JSON + sealed AEAD)
// ============================================================

const DOC_SCHEMA_VERSION: u32 = 1;

#[derive(Serialize, Deserialize)]
struct ChunkSnapshot {
    id: String,
    doc_id: String,
    idx: usize,
    text: String, // decompressed; recompressed on load via Chunk::new
    embedding: Vec<f32>,
}

#[derive(Serialize, Deserialize)]
struct DocumentSnapshot {
    schema_version: u32,
    chunk_size: usize,
    chunk_overlap: usize,
    documents: Vec<DocumentEntry>,
    chunks: Vec<ChunkSnapshot>,
}

impl<E: Encoder> DocumentMemory<E> {
    fn build_snapshot(&self) -> DocumentSnapshot {
        DocumentSnapshot {
            schema_version: DOC_SCHEMA_VERSION,
            chunk_size: self.chunk_size,
            chunk_overlap: self.chunk_overlap,
            documents: self.documents.values().cloned().collect(),
            chunks: self
                .chunks
                .iter()
                .map(|c| ChunkSnapshot {
                    id: c.id.clone(),
                    doc_id: c.doc_id.clone(),
                    idx: c.idx,
                    text: c.text(),
                    embedding: c.embedding.clone(),
                })
                .collect(),
        }
    }

    fn from_snapshot(snap: DocumentSnapshot, encoder: E) -> Result<Self, PersistError> {
        if snap.schema_version != DOC_SCHEMA_VERSION {
            return Err(PersistError::Corrupt(format!(
                "document schema {} != {}",
                snap.schema_version, DOC_SCHEMA_VERSION
            )));
        }
        let mut dm = DocumentMemory::new(encoder).with_chunking(snap.chunk_size, snap.chunk_overlap);
        for c in snap.chunks {
            dm.chunks.push(Chunk::new(c.id, c.doc_id, c.idx, &c.text, c.embedding));
        }
        for d in snap.documents {
            dm.documents.insert(d.doc_id.clone(), d);
        }
        dm.rebuild_ann();
        Ok(dm)
    }

    /// Save the document store to `path/documents.json` (plaintext).
    pub fn save_state(&self, path: impl AsRef<Path>) -> Result<(), PersistError> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        let json = serde_json::to_vec(&self.build_snapshot())?;
        std::fs::write(path.join("documents.json"), json)?;
        Ok(())
    }

    /// Restore from `path/documents.json`. Returns an empty store (not an
    /// error) if the file is absent, so callers can open-or-create.
    pub fn load_state(path: impl AsRef<Path>, encoder: E) -> Result<Self, PersistError> {
        let file = path.as_ref().join("documents.json");
        if !file.exists() {
            return Ok(DocumentMemory::new(encoder));
        }
        let raw = std::fs::read(file)?;
        let snap: DocumentSnapshot = serde_json::from_slice(&raw)?;
        Self::from_snapshot(snap, encoder)
    }

    /// Encrypted-at-rest save (ChaCha20-Poly1305) → `path/documents.json.enc`.
    /// File contents are sensitive, so this is the recommended on-device path.
    pub fn save_state_sealed(&self, path: impl AsRef<Path>, key: &[u8; 32]) -> Result<(), PersistError> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        let bytes = serde_json::to_vec(&self.build_snapshot())?;
        let sealed = crate::crypto::seal(&bytes, key);
        std::fs::write(path.join("documents.json.enc"), sealed)?;
        Ok(())
    }

    /// Restore from `path/documents.json.enc` (empty store if absent).
    pub fn load_state_sealed(path: impl AsRef<Path>, key: &[u8; 32], encoder: E) -> Result<Self, PersistError> {
        let file = path.as_ref().join("documents.json.enc");
        if !file.exists() {
            return Ok(DocumentMemory::new(encoder));
        }
        let sealed = std::fs::read(file)?;
        let raw = crate::crypto::open(&sealed, key)
            .map_err(|e| PersistError::Corrupt(e.to_string()))?;
        let snap: DocumentSnapshot = serde_json::from_slice(&raw)?;
        Self::from_snapshot(snap, encoder)
    }
}

/// Word-window chunking with overlap. Splits paragraphs first (blank-
/// line boundaries) so short standalone paragraphs survive intact.
/// Matches the Python reference's `_chunk_text`.
pub fn chunk_text(text: &str, chunk_size: usize, chunk_overlap: usize) -> Vec<String> {
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();
    let paragraphs = if paragraphs.is_empty() && !text.trim().is_empty() {
        vec![text.trim()]
    } else {
        paragraphs
    };

    let mut chunks = Vec::new();
    for para in paragraphs {
        let words: Vec<&str> = para.split_whitespace().collect();
        if words.len() <= chunk_size {
            chunks.push(words.join(" "));
            continue;
        }
        let step = chunk_size.saturating_sub(chunk_overlap).max(1);
        let mut start = 0;
        while start < words.len() {
            let end = (start + chunk_size).min(words.len());
            chunks.push(words[start..end].join(" "));
            if end == words.len() {
                break;
            }
            start += step;
        }
    }
    chunks
}

fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make() -> DocumentMemory<MockEncoder> {
        DocumentMemory::new(MockEncoder::new(32)).with_chunking(10, 2)
    }

    #[test]
    fn chunk_text_short() {
        let c = chunk_text("just a few words", 10, 2);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn chunk_text_long_with_overlap() {
        let text: String = (0..50)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let c = chunk_text(&text, 10, 2);
        // 50 words, chunk=10, step=8 → chunks at 0..10, 8..18, 16..26, 24..34, 32..42, 40..50
        assert_eq!(c.len(), 6);
    }

    #[test]
    fn add_document_indexes_chunks() {
        let mut dm = make();
        let n = dm.add_document(
            "doc1",
            "this is a test document with several words about embeddings",
            HashMap::new(),
        );
        assert_eq!(n, 1); // short text → 1 chunk
        assert_eq!(dm.n_documents(), 1);
        assert_eq!(dm.n_chunks(), 1);
    }

    #[test]
    fn add_document_overwrites() {
        let mut dm = make();
        dm.add_document("doc1", "first version", HashMap::new());
        dm.add_document("doc1", "completely different content here", HashMap::new());
        assert_eq!(dm.n_documents(), 1);
        // No leftover chunks from the first version.
        assert!(
            dm.chunks
                .iter()
                .all(|c| {
                    let t = c.text();
                    t.contains("different") || t.contains("content")
                })
        );
    }

    #[test]
    fn short_text_stored_uncompressed() {
        let mut dm = make();
        dm.add_document("a", "short text", HashMap::new());
        let c = &dm.chunks[0];
        assert!(!c.is_compressed(), "short text shouldn't trigger gzip");
        assert_eq!(c.text(), "short text");
    }

    #[test]
    fn long_text_stored_compressed_and_smaller() {
        // Build a long but repetitive paragraph so gzip can squeeze it.
        let long: String = "the quick brown fox jumps over the lazy dog. ".repeat(40);
        let original_len = long.len();
        let mut dm = DocumentMemory::new(MockEncoder::new(32))
            .with_chunking(10_000, 0);  // ensure 1 chunk
        dm.add_document("a", &long, HashMap::new());
        let c = &dm.chunks[0];
        assert!(c.is_compressed(), "long repetitive text should compress");
        assert!(
            c.stored_bytes() < original_len.min(500),
            "gzip stored={} should beat preview-capped raw={}",
            c.stored_bytes(), original_len.min(500),
        );
        // Roundtrip — text() returns the same 500-char preview.
        let recovered = c.text();
        assert!(recovered.starts_with("the quick brown fox"));
    }

    #[test]
    fn search_works_on_compressed_chunks() {
        let mut dm = DocumentMemory::new(MockEncoder::new(32))
            .with_chunking(10_000, 0);
        let long: String = "alpha beta gamma delta epsilon ".repeat(30);
        dm.add_document("a", &long, HashMap::new());
        let hits = dm.semantic_search::<fn(&DocumentEntry) -> bool>(
            "alpha beta", 3, None,
        );
        assert!(!hits.is_empty());
        // The returned hit text should be the decompressed preview.
        assert!(hits[0].text.contains("alpha"));
    }

    #[test]
    fn semantic_search_returns_relevant() {
        let mut dm = make();
        dm.add_document("a", "the cat sat on the mat", HashMap::new());
        dm.add_document("b", "this is about programming languages", HashMap::new());
        dm.add_document("c", "another unrelated note", HashMap::new());
        let hits = dm.semantic_search::<fn(&DocumentEntry) -> bool>(
            "cat sat on mat",
            5,
            None,
        );
        assert!(!hits.is_empty());
        // With our mock encoder (byte-hashed) exact query should rank doc 'a' first.
        assert_eq!(hits[0].doc_id, "a");
    }

    #[test]
    fn semantic_search_respects_filter() {
        let mut dm = make();
        let mut meta_a: HashMap<String, serde_json::Value> = HashMap::new();
        meta_a.insert("kind".into(), serde_json::json!("note"));
        let mut meta_b: HashMap<String, serde_json::Value> = HashMap::new();
        meta_b.insert("kind".into(), serde_json::json!("audit"));
        dm.add_document("a", "some note text", meta_a);
        dm.add_document("b", "some audit text", meta_b);
        let hits = dm.semantic_search(
            "any query",
            5,
            Some(|d: &DocumentEntry| {
                d.metadata.get("kind").map(|v| v == "audit").unwrap_or(false)
            }),
        );
        assert!(hits.iter().all(|h| h.doc_id == "b"));
    }

    #[test]
    fn forget_document_drops_chunks() {
        let mut dm = make();
        dm.add_document("a", "first doc", HashMap::new());
        dm.add_document("b", "second doc", HashMap::new());
        let n_before = dm.n_chunks();
        assert!(dm.forget_document("a"));
        assert_eq!(dm.n_documents(), 1);
        assert!(dm.n_chunks() < n_before);
        assert!(dm.chunks.iter().all(|c| c.doc_id != "a"));
    }

    #[test]
    fn forget_unknown_returns_false() {
        let mut dm = make();
        assert!(!dm.forget_document("does-not-exist"));
    }

    #[test]
    fn empty_search_yields_empty() {
        let dm = make();
        let hits = dm.semantic_search::<fn(&DocumentEntry) -> bool>("anything", 5, None);
        assert!(hits.is_empty());
    }

    fn make_emb() -> DocumentMemory<MockEncoder> {
        // dim=8 mock; we add chunks with explicit embeddings via the injection API.
        DocumentMemory::new(MockEncoder::new(8))
    }

    #[test]
    fn injection_add_and_search_with_filters() {
        let mut dm = make_emb();
        let mut meta_note: HashMap<String, serde_json::Value> = HashMap::new();
        meta_note.insert("kind".into(), serde_json::json!("note"));
        let mut meta_audit: HashMap<String, serde_json::Value> = HashMap::new();
        meta_audit.insert("kind".into(), serde_json::json!("audit"));

        dm.add_document_with_embeddings(
            "a", vec!["alpha shopping list milk".into()],
            vec![vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], meta_note);
        dm.add_document_with_embeddings(
            "b", vec!["beta audit log entry".into()],
            vec![vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], meta_audit);

        // Plain vector search: query close to doc 'a'.
        let q = vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let hits = dm.search(&q, 5, &SearchFilter::default());
        assert_eq!(hits[0].doc_id, "a");
        assert_eq!(hits[0].metadata.get("kind"), Some(&serde_json::json!("note")));

        // metadata filter → only audit docs.
        let mut f = SearchFilter::default();
        f.metadata.insert("kind".into(), serde_json::json!("audit"));
        let hits = dm.search(&q, 5, &f);
        assert!(hits.iter().all(|h| h.doc_id == "b"));

        // contains_text filter (case-insensitive substring on chunk text).
        let mut f2 = SearchFilter::default();
        f2.contains_text = Some("SHOPPING".into());
        let hits = dm.search(&q, 5, &f2);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].doc_id, "a");

        // min_score threshold drops weak matches.
        let mut f3 = SearchFilter::default();
        f3.min_score = Some(0.95);
        let hits = dm.search(&vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 5, &f3);
        assert!(hits.iter().all(|h| h.score >= 0.95));
        assert_eq!(hits[0].doc_id, "b");

        // doc_ids restriction.
        let mut f4 = SearchFilter::default();
        f4.doc_ids = Some(vec!["b".into()]);
        let hits = dm.search(&q, 5, &f4);
        assert!(hits.iter().all(|h| h.doc_id == "b"));
    }

    #[test]
    fn document_persistence_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut dm = make_emb();
        dm.add_document_with_embeddings(
            "big", vec!["chunk one text".into(), "chunk two text".into()],
            vec![vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                 vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            HashMap::new());
        dm.save_state(dir.path()).unwrap();

        let dm2 = DocumentMemory::load_state(dir.path(), MockEncoder::new(8)).unwrap();
        assert_eq!(dm2.n_documents(), 1);
        assert_eq!(dm2.n_chunks(), 2);
        let hits = dm2.search(&vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1, &SearchFilter::default());
        assert_eq!(hits[0].doc_id, "big");
        assert!(hits[0].text.contains("chunk one"));
    }

    #[test]
    fn document_sealed_roundtrip_and_no_plaintext() {
        let dir = tempfile::tempdir().unwrap();
        let key = [3u8; 32];
        let mut dm = make_emb();
        dm.add_document_with_embeddings(
            "secret", vec!["confidential file contents".into()],
            vec![vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], HashMap::new());
        dm.save_state_sealed(dir.path(), &key).unwrap();

        assert!(!dir.path().join("documents.json").exists(), "no plaintext doc dump");
        assert!(dir.path().join("documents.json.enc").exists());

        let ok = DocumentMemory::load_state_sealed(dir.path(), &key, MockEncoder::new(8)).unwrap();
        assert_eq!(ok.n_documents(), 1);
        // Wrong key → error.
        assert!(DocumentMemory::load_state_sealed(dir.path(), &[9u8; 32], MockEncoder::new(8)).is_err());
    }

    #[test]
    fn e5_mlx_vectors_rank_correctly_in_memory_plant() {
        // A′ proof: REAL multilingual-e5-small vectors produced via MLX (the
        // Qwen on-device runtime family — NOT ONNX Runtime; see
        // bindings/embed_e5_mlx.py). End-to-end: e5(MLX) → Memory Plant
        // add/search → the ru query ranks the ru document first. Baked into a
        // committed fixture so the suite re-runs with no Python/MLX/network.
        let raw = include_str!("testdata/e5_mlx_vectors.json");
        let v: serde_json::Value = serde_json::from_str(raw).unwrap();
        let dim = v["dim"].as_u64().unwrap() as usize;
        let to_vec = |val: &serde_json::Value| -> Vec<f32> {
            val.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
        };

        let mut dm = DocumentMemory::new(MockEncoder::new(dim));
        for doc in v["docs"].as_array().unwrap() {
            dm.add_document_with_embeddings(
                doc["id"].as_str().unwrap().to_string(),
                vec![doc["text"].as_str().unwrap().to_string()],
                vec![to_vec(&doc["emb"])],
                HashMap::new(),
            );
        }
        assert_eq!(dm.n_documents(), 4);

        let q = to_vec(&v["query"]["emb"]);
        let expected = v["query"]["expected_top"].as_str().unwrap();
        let hits = dm.search(&q, 4, &SearchFilter::default());
        assert!(!hits.is_empty());
        assert_eq!(hits[0].doc_id, expected, "e5(MLX) ru query must rank the ru doc first");
        assert!(hits[0].score > 0.8, "expected a strong ru match, got {}", hits[0].score);
        // ru beats the en distractor by a clear margin.
        let en = hits.iter().find(|h| h.doc_id == "en_cell").unwrap();
        assert!(hits[0].score - en.score > 0.15, "ru should clearly beat en");
    }

    #[test]
    fn ann_fast_path_matches_exact_no_filter() {
        use crate::index::BruteForceIndex;
        let mut exact = make();
        exact.add_document("a", "alpha bravo charlie delta", HashMap::new());
        exact.add_document("b", "xi omicron pi rho", HashMap::new());
        let mut base = make();
        base.add_document("a", "alpha bravo charlie delta", HashMap::new());
        base.add_document("b", "xi omicron pi rho", HashMap::new());
        let ann = base.with_ann(Box::new(BruteForceIndex::new()));

        let q = "alpha bravo charlie delta";
        let he = exact.semantic_search::<fn(&DocumentEntry) -> bool>(q, 5, None);
        let ha = ann.semantic_search::<fn(&DocumentEntry) -> bool>(q, 5, None);
        assert!(!he.is_empty() && !ha.is_empty());
        // ANN path (exact BruteForceIndex) must match the inline exact top hit.
        assert_eq!(ha[0].chunk_id, he[0].chunk_id, "ANN top must match exact (no filter)");
    }
}

#[cfg(all(test, feature = "fastembed"))]
mod e5_tests {
    use super::*;

    fn cos(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
    }

    /// P3.2: multilingual e5 fixes ru/kz dense retrieval on-device (mirror of the
    /// Python P0 test). Downloads MultilingualE5Small on first run.
    #[test]
    fn multilingual_ru_kz_ranks_correctly() {
        let enc = FastembedEncoder::multilingual().expect("e5 model load");
        let ru_author = enc.embed_passage("Роман «Война и мир» написал Лев Толстой.");
        let kz_capital = enc.embed_passage("Қазақстанның астанасы — Астана қаласы.");
        let en_distract = enc.embed_passage("The mitochondria is the powerhouse of the cell.");
        let q_ru = enc.embed_query("Кто написал роман Война и мир?");
        let q_kz = enc.embed_query("Қазақстанның астанасы қай қала?");
        assert!(cos(&q_ru, &ru_author) > cos(&q_ru, &en_distract), "ru>en");
        assert!(cos(&q_ru, &ru_author) > cos(&q_ru, &kz_capital), "ru>kz");
        assert!(cos(&q_kz, &kz_capital) > cos(&q_kz, &en_distract), "kz>en");
        assert!(cos(&q_kz, &kz_capital) > cos(&q_kz, &ru_author), "kz>ru");
    }

    /// P3 step 1: end-to-end semantic_search with e5 query-prefix wiring —
    /// a ru query retrieves the ru document over an en distractor.
    #[test]
    fn semantic_search_ru_with_e5_query_prefix() {
        use std::collections::HashMap;
        let enc = FastembedEncoder::multilingual().expect("e5 model load");
        let mut dm = DocumentMemory::new(enc);
        dm.add_document("ru", "Лев Толстой написал роман «Война и мир».", HashMap::new());
        dm.add_document("en", "Photosynthesis converts light into chemical energy.", HashMap::new());
        let hits = dm.semantic_search::<fn(&DocumentEntry) -> bool>(
            "Кто автор романа Война и мир?", 5, None);
        assert!(!hits.is_empty(), "expected hits");
        assert_eq!(hits[0].doc_id, "ru", "ru query should retrieve ru doc first");
    }
}
