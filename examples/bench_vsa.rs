//! VSA(k) family empirical benchmark.
//!
//! For each (dim, N facts, k) configuration:
//!   1. Build a random vocab of K distinct values
//!   2. Store N random (key → value) facts in a single shard
//!   3. Retrieve each key, run AMP refinement, check accuracy
//!   4. Repeat 3 seeds, average
//!
//! Compares HLB (k=1 via existing AdaptiveMemory) against qFHRR k ∈
//! {2, 4, 8} implemented inline as a complex-domain memory tensor.
//!
//! Run:
//!   cargo run --release --example bench_vsa

use memory_plant::AdaptiveMemory;
use num_complex::Complex;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use sha2::{Digest, Sha256};
use std::time::Instant;

// ============================================================
// HLB benchmark (using existing AdaptiveMemory)
// ============================================================

fn bench_hlb(dim: usize, n_facts: usize, vocab_size: usize, seed: u64) -> (f32, usize, u128) {
    let mut am = AdaptiveMemory::new(dim, vocab_size, Some(usize::MAX / 2), seed).unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut truth: Vec<(String, String)> = Vec::with_capacity(n_facts);
    for i in 0..n_facts {
        let key = format!("k_{i}");
        let val_idx = rng.random_range(0..vocab_size);
        let value = format!("v_{val_idx}");
        am.store(&key, &value).unwrap();
        truth.push((key, value));
    }
    let t0 = Instant::now();
    let mut correct = 0;
    for (k, expected) in &truth {
        if let Some(got) = am.retrieve(k).unwrap() {
            if &got == expected { correct += 1; }
        }
    }
    let elapsed_us = t0.elapsed().as_micros();
    // Storage = M tensor (single Vec<f32> of dim) per shard + key strings.
    // Single shard here (we forced shard_capacity huge), so:
    let storage_bytes = dim * 4 + n_facts * 16; // M + keys
    (100.0 * correct as f32 / n_facts as f32, storage_bytes / n_facts, elapsed_us)
}

// ============================================================
// qFHRR benchmark (inline complex-domain memory)
// ============================================================

/// Map key+dim+k → deterministic complex-phase role.
fn role_qfhrr(key: &str, dim: usize, k: u8) -> Vec<Complex<f32>> {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    hasher.update(&(dim as u64).to_le_bytes());
    hasher.update(&[k]);
    let digest = hasher.finalize();
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&digest);
    let mut rng = ChaCha8Rng::from_seed(seed);
    let n_phases = 1u32 << k as u32;
    let tau = std::f32::consts::TAU;
    (0..dim)
        .map(|_| {
            let j = rng.random_range(0..n_phases);
            let theta = tau * j as f32 / n_phases as f32;
            Complex::new(theta.cos(), theta.sin())
        })
        .collect()
}

/// Build a vocab of K random complex-phase unit vectors.
fn build_qfhrr_vocab(k: u8, vocab_size: usize, dim: usize, seed: u64) -> Vec<Vec<Complex<f32>>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_phases = 1u32 << k as u32;
    let tau = std::f32::consts::TAU;
    (0..vocab_size)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    let j = rng.random_range(0..n_phases);
                    let theta = tau * j as f32 / n_phases as f32;
                    Complex::new(theta.cos(), theta.sin())
                })
                .collect()
        })
        .collect()
}

fn complex_cosine(a: &[Complex<f32>], b: &[Complex<f32>]) -> f32 {
    let dot: Complex<f32> = a.iter().zip(b).map(|(x, y)| x * y.conj()).sum();
    let na: f32 = a.iter().map(|c| c.norm_sqr()).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|c| c.norm_sqr()).sum::<f32>().sqrt();
    let denom = na * nb;
    if denom < f32::EPSILON { 0.0 } else { dot.re / denom }
}

fn bench_qfhrr(
    dim: usize,
    n_facts: usize,
    vocab_size: usize,
    k: u8,
    seed: u64,
) -> (f32, usize, u128) {
    let vocab = build_qfhrr_vocab(k, vocab_size, dim, seed);
    let mut memory: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); dim];
    let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(1));
    let mut truth: Vec<(String, usize)> = Vec::with_capacity(n_facts);
    let mut roles: Vec<Vec<Complex<f32>>> = Vec::with_capacity(n_facts);

    for i in 0..n_facts {
        let key = format!("k_{i}");
        let val_idx = rng.random_range(0..vocab_size);
        let role = role_qfhrr(&key, dim, k);
        for (j, slot) in memory.iter_mut().enumerate() {
            *slot += role[j] * vocab[val_idx][j];
        }
        truth.push((key, val_idx));
        roles.push(role);
    }

    let t0 = Instant::now();
    let mut correct = 0;
    for (i, (_, expected_idx)) in truth.iter().enumerate() {
        let role = &roles[i];
        let decoded: Vec<Complex<f32>> = (0..dim)
            .map(|j| memory[j] * role[j].conj())
            .collect();
        let mut best_idx = 0;
        let mut best = f32::NEG_INFINITY;
        for (vi, v) in vocab.iter().enumerate() {
            let sim = complex_cosine(&decoded, v);
            if sim > best { best = sim; best_idx = vi; }
        }
        if best_idx == *expected_idx { correct += 1; }
    }
    let elapsed_us = t0.elapsed().as_micros();

    // Storage = M tensor (Vec<Complex<f32>> of dim, 8 bytes each) +
    // packed-phase roles (which we'd derive on demand if persisted).
    // For consumer storage cost we report memory only — that's what
    // gets persisted; roles re-derive from SHA-256 of key.
    let storage_bytes = dim * 8 + n_facts * 16;
    (100.0 * correct as f32 / n_facts as f32, storage_bytes / n_facts, elapsed_us)
}

// ============================================================
// Driver
// ============================================================

fn average3<F: Fn(u64) -> (f32, usize, u128)>(f: F) -> (f32, usize, u128) {
    let (a1, s1, t1) = f(42);
    let (a2, s2, t2) = f(123);
    let (a3, s3, t3) = f(999);
    ((a1 + a2 + a3) / 3.0, (s1 + s2 + s3) / 3, (t1 + t2 + t3) / 3)
}

fn main() {
    println!("=== VSA(k) family benchmark — recall + storage + retrieve latency ===");
    println!("Avg over 3 seeds. Vocab size K=64 random unit-phase values.");
    println!();
    println!("{:>6} {:>7} {:>5} {:>8} {:>10} {:>10}",
        "dim", "N", "k", "acc%", "B/fact", "μs/get");
    println!("{}", "-".repeat(60));

    let configs = [
        (256, 30),
        (256, 60),
        (512, 50),
        (512, 100),
        (1024, 80),
        (1024, 200),
        (1024, 400),
        (2048, 200),
    ];

    for (dim, n) in configs {
        let vocab_size = 64;
        // HLB (k=1) via existing AdaptiveMemory
        let (acc, sb, us) = average3(|s| bench_hlb(dim, n, vocab_size, s));
        println!("{:>6} {:>7} {:>5} {:>7.1}% {:>10} {:>9}μs",
            dim, n, 1, acc, sb, us / n as u128);

        for k in [2u8, 4, 8] {
            let (acc, sb, us) = average3(|s| bench_qfhrr(dim, n, vocab_size, k, s));
            println!("{:>6} {:>7} {:>5} {:>7.1}% {:>10} {:>9}μs",
                dim, n, k, acc, sb, us / n as u128);
        }
        println!();
    }

    println!("=== Storage / capacity / accuracy summary ===");
    println!();
    println!("HLB (k=1) — real bipolar, M tensor is dim*4 B");
    println!("qFHRR (k=2,4,8) — complex M tensor is dim*8 B (double)");
    println!();
    println!("Cross-talk noise σ = √(N-1)/√d (HLB) or √(N-1)/√d × phase_diversity (qFHRR)");
    println!("Phase transition where recall drops sharply: σ ≈ 0.28 for HLB.");
    println!();
    println!("Notes:");
    println!("- HLB column uses AdaptiveMemory with single shard (capacity forced huge)");
    println!("- qFHRR runs inline complex-domain memory — no shard auto-split,");
    println!("  no iterative AMP yet (this is single-shot bench).");
    println!("- Both use exact role re-derivation from key via SHA-256 → ChaCha8.");
}
