//! Local end-to-end demo of the on-device memory engine.
//!
//!   cargo run --example demo
//!
//! Exercises the exact API the phone calls (the UniFFI `MemoryPlant` object):
//! facts (store/recall/ingest/provable-forget) + document RAG (caller-supplied
//! embeddings → on device these come from e5/Qwen; here we use tiny vectors).
use memory_plant::ffi::MemoryPlant;
use std::collections::HashMap;

fn main() {
    println!("== Memory Plant — local demo ==\n");

    // In-memory engine. For durable + encrypted: MemoryPlant::load_or_create_sealed(path, key, …)
    let mp = MemoryPlant::new(512, 4096, "default".into());

    // --- Facts: store + exact recall ---
    mp.store_fact("works_as".into(), "engineer".into()).unwrap();
    mp.store_fact("lives_in".into(), "Almaty".into()).unwrap();
    println!("recall works_as -> {:?}", mp.recall_fact("works_as".into()).unwrap());
    println!("recall lives_in -> {:?}", mp.recall_fact("lives_in".into()).unwrap());

    // --- Ingest free text (offline RegexExtractor) ---
    let facts = mp.ingest_message("My name is Abzal. I live in Astana.".into()).unwrap();
    println!("\ningested {} fact(s) from text:", facts.len());
    for f in &facts {
        println!("  {} = {}", f.predicate, f.obj);
    }

    // --- Provable forget (GDPR) ---
    let removed = mp.forget_fact("works_as".into()).unwrap();
    println!(
        "\nforget works_as -> removed={}, recall now -> {:?}",
        removed,
        mp.recall_fact("works_as".into()).unwrap()
    );
    println!("total facts: {}", mp.total_facts());

    // --- Document RAG: store chunks with embeddings, semantic search ---
    // On device the vectors come from e5/Qwen; here we hand-craft 4-d vectors.
    mp.add_document(
        "note_milk".into(),
        vec!["buy milk and bread".into()],
        vec![vec![1.0, 0.0, 0.0, 0.0]],
        HashMap::new(),
    ).unwrap();
    mp.add_document(
        "note_meeting".into(),
        vec!["meeting with Bob at 5pm".into()],
        vec![vec![0.0, 1.0, 0.0, 0.0]],
        HashMap::new(),
    ).unwrap();

    let hits = mp.search(vec![0.95, 0.05, 0.0, 0.0], 3, HashMap::new(), None, None, None);
    println!("\ndoc search (query ≈ groceries), {} doc(s) indexed:", mp.n_documents());
    for h in &hits {
        println!("  {:.3}  {}  \"{}\"", h.score, h.doc_id, h.text);
    }

    println!("\n== done — engine runs locally ==");
}
