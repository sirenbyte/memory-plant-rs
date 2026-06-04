//! Retrieval micro-benchmark for Memory Plant.
//!
//! Loads a labeled multilingual corpus + queries (e5 vectors produced via MLX —
//! see ../../bindings/embed via edge-lora-test/bench_retrieval_mlx.py), runs them
//! through DocumentMemory `search`, and reports Recall@k / MRR — overall and per
//! language. Also runs a MockEncoder (byte-hash) baseline over the SAME texts so
//! the e5 lift is visible (proves the harness isn't trivially passing).
//!
//! HONEST SCOPE: measures the RETRIEVAL layer (the part MP owns), not LongMemEval
//! (no multi-session/temporal-reasoning/answer-generation).
//!
//! Usage: cargo run --release --bin bench_retrieval -- /tmp/bench_retrieval.json

use std::collections::BTreeMap;
use std::collections::HashMap;

use memory_plant::document::DocumentEntry;
use memory_plant::{DocumentMemory, MockEncoder, SearchFilter};

const K: usize = 10;

fn to_vec(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

/// rank (1-based) of the first relevant id in `ranked`, or 0 if absent in top-K.
fn first_relevant_rank(ranked: &[String], relevant: &[String]) -> usize {
    for (i, id) in ranked.iter().enumerate() {
        if relevant.iter().any(|r| r == id) {
            return i + 1;
        }
    }
    0
}

#[derive(Default, Clone)]
struct Agg {
    n: usize,
    r1: usize,
    r3: usize,
    r5: usize,
    mrr: f64,
}
impl Agg {
    fn add(&mut self, rank: usize) {
        self.n += 1;
        if rank == 1 { self.r1 += 1; }
        if rank >= 1 && rank <= 3 { self.r3 += 1; }
        if rank >= 1 && rank <= 5 { self.r5 += 1; }
        if rank >= 1 { self.mrr += 1.0 / rank as f64; }
    }
    fn row(&self, label: &str) -> String {
        let p = |x: usize| if self.n == 0 { 0.0 } else { x as f64 / self.n as f64 };
        format!(
            "{:<14} n={:<3} R@1={:.3}  R@3={:.3}  R@5={:.3}  MRR={:.3}",
            label, self.n, p(self.r1), p(self.r3), p(self.r5),
            if self.n == 0 { 0.0 } else { self.mrr / self.n as f64 },
        )
    }
}

fn report(title: &str, per_lang: &BTreeMap<String, Agg>, overall: &Agg) {
    println!("\n=== {title} ===");
    for (lang, a) in per_lang {
        println!("  {}", a.row(&format!("lang:{lang}")));
    }
    println!("  {}", overall.row("OVERALL"));
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/tmp/bench_retrieval.json".into());
    let raw = std::fs::read_to_string(&path).expect("read bench json");
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let dim = v["dim"].as_u64().unwrap() as usize;
    let corpus = v["corpus"].as_array().unwrap();
    let queries = v["queries"].as_array().unwrap();
    println!("bench: model={} dim={dim} corpus={} queries={}",
        v["model"].as_str().unwrap_or("?"), corpus.len(), queries.len());

    // ---- e5 (real MLX vectors via the injection path) ----
    let mut e5 = DocumentMemory::new(MockEncoder::new(dim));
    for d in corpus {
        e5.add_document_with_embeddings(
            d["id"].as_str().unwrap().to_string(),
            vec![d["text"].as_str().unwrap().to_string()],
            vec![to_vec(&d["emb"])],
            HashMap::new(),
        );
    }
    // ---- mock baseline (byte-hash encoder over the same texts) ----
    let mut mock = DocumentMemory::new(MockEncoder::new(dim));
    for d in corpus {
        mock.add_document(d["id"].as_str().unwrap().to_string(),
            d["text"].as_str().unwrap(), HashMap::new());
    }

    let (mut e5_all, mut mock_all) = (Agg::default(), Agg::default());
    let mut e5_lang: BTreeMap<String, Agg> = BTreeMap::new();
    let mut mock_lang: BTreeMap<String, Agg> = BTreeMap::new();

    for q in queries {
        let lang = q["lang"].as_str().unwrap().to_string();
        let relevant: Vec<String> =
            q["relevant"].as_array().unwrap().iter().map(|x| x.as_str().unwrap().to_string()).collect();

        // e5: search by the precomputed query vector.
        let q_emb = to_vec(&q["emb"]);
        let e5_ranked: Vec<String> = e5.search(&q_emb, K, &SearchFilter::default())
            .into_iter().map(|h| h.doc_id).collect();
        let r = first_relevant_rank(&e5_ranked, &relevant);
        e5_all.add(r);
        e5_lang.entry(lang.clone()).or_default().add(r);

        // mock: search by the query TEXT (encoder hashes it).
        let m_ranked: Vec<String> = mock
            .semantic_search::<fn(&DocumentEntry) -> bool>(q["text"].as_str().unwrap(), K, None)
            .into_iter().map(|h| h.doc_id).collect();
        let rm = first_relevant_rank(&m_ranked, &relevant);
        mock_all.add(rm);
        mock_lang.entry(lang).or_default().add(rm);
    }

    report("e5 (multilingual-e5-small via MLX) → Memory Plant", &e5_lang, &e5_all);
    report("MockEncoder byte-hash baseline (floor)", &mock_lang, &mock_all);
    println!();
}
