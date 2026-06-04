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
use ndarray::ArrayView1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Anything that can turn `[String]` into `Vec<Vec<f32>>` of fixed
/// dimensionality. Send + Sync so DocumentMemory is share-able.
pub trait Encoder: Send + Sync {
    fn encode(&self, texts: &[String]) -> Vec<Vec<f32>>;
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

    fn embed_one(&self, text: String) -> Vec<f32> {
        self.encode(&[text]).into_iter().next().unwrap_or_default()
    }

    /// Embed a search query (e5 models: "query: " prefix).
    pub fn embed_query(&self, text: &str) -> Vec<f32> {
        let t = if self.e5 { format!("query: {text}") } else { text.to_string() };
        self.embed_one(t)
    }

    /// Embed a stored passage (e5 models: "passage: " prefix).
    pub fn embed_passage(&self, text: &str) -> Vec<f32> {
        let t = if self.e5 { format!("passage: {text}") } else { text.to_string() };
        self.embed_one(t)
    }
}

#[cfg(feature = "fastembed")]
impl Encoder for FastembedEncoder {
    fn encode(&self, texts: &[String]) -> Vec<Vec<f32>> {
        let texts_owned: Vec<String> = texts.iter().cloned().collect();
        match self.model.lock() {
            Ok(mut m) => m.embed(texts_owned, None).unwrap_or_default(),
            Err(_) => Vec::new(),
        }
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
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub chunk_id: String,
    pub doc_id: String,
    pub score: f32,
    pub text: String,
}

impl<E: Encoder> DocumentMemory<E> {
    pub fn new(encoder: E) -> Self {
        Self {
            encoder,
            documents: HashMap::new(),
            chunks: Vec::new(),
            chunk_size: 200,
            chunk_overlap: 20,
        }
    }

    pub fn with_chunking(mut self, size: usize, overlap: usize) -> Self {
        self.chunk_size = size;
        self.chunk_overlap = overlap.min(size.saturating_sub(1));
        self
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
        n
    }

    /// Drop every chunk owned by `doc_id`. Returns true if anything
    /// was removed.
    pub fn forget_document(&mut self, doc_id: &str) -> bool {
        let before = self.chunks.len();
        self.chunks.retain(|c| c.doc_id != doc_id);
        self.documents.remove(doc_id);
        self.chunks.len() != before
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
        let q_emb = self.encoder.encode(&[query.to_string()]);
        if q_emb.is_empty() {
            return Vec::new();
        }
        let q = &q_emb[0];
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
            .map(|(score, c)| SearchHit {
                chunk_id: c.id.clone(),
                doc_id: c.doc_id.clone(),
                score,
                text: c.text(),
            })
            .collect()
    }

    pub fn document(&self, doc_id: &str) -> Option<&DocumentEntry> {
        self.documents.get(doc_id)
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
}
