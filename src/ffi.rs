//! UniFFI bindings (P3): expose the core memory engine to Swift (iOS) and
//! Kotlin (Android). `MemoryService` is wrapped in a `Mutex` (it's Send+Sync —
//! `Extractor: Send + Sync`). The heavy/optional features (fastembed/ort, ann)
//! are NOT part of this FFI surface, so the bindings build is light and the
//! core (HLB store/recall/forget, persistence, crypto) cross-compiles cleanly.
//!
//! Generate bindings after `cargo build`:
//! ```sh
//! cargo run --bin uniffi-bindgen -- generate \
//!     --library target/debug/libmemory_plant.dylib --language swift  --out-dir bindings/swift
//! cargo run --bin uniffi-bindgen -- generate \
//!     --library target/debug/libmemory_plant.dylib --language kotlin --out-dir bindings/kotlin
//! ```
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::extractor::{Extractor, RegexExtractor};
use crate::fact::Fact;
use crate::service::MemoryService;

#[derive(uniffi::Record)]
pub struct FactDto {
    pub subject: String,
    pub predicate: String,
    pub obj: String,
    pub source: String,
}

impl From<Fact> for FactDto {
    fn from(f: Fact) -> Self {
        Self {
            subject: f.subject,
            predicate: f.predicate,
            obj: f.obj,
            source: f.source,
        }
    }
}

#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum MpError {
    #[error("{msg}")]
    Memory { msg: String },
}

impl MpError {
    fn from_err<E: std::fmt::Display>(e: E) -> Self {
        MpError::Memory { msg: e.to_string() }
    }
}

/// On-device personal memory for a single user. Thread-safe.
#[derive(uniffi::Object)]
pub struct MemoryPlant {
    inner: Mutex<MemoryService>,
    user: String,
}

#[uniffi::export]
impl MemoryPlant {
    /// New, EMPTY in-memory engine (never touches disk). Use `open` for a
    /// durable engine. Sane defaults: dim 512, vocab_cap 4096.
    #[uniffi::constructor]
    pub fn new(dim: u32, vocab_cap: u32, user: String) -> Arc<Self> {
        let factory = || -> Arc<dyn Extractor> { Arc::new(RegexExtractor::new()) };
        let svc = MemoryService::new(factory, dim as usize, vocab_cap as usize);
        Arc::new(Self { inner: Mutex::new(svc), user })
    }

    /// Open a DURABLE engine: load the state previously written by `save` at
    /// `path`, or start fresh there if none exists yet (load-or-create). This
    /// closes the cross-session round-trip — `loadOrCreate(path) → … → save(path)`
    /// survives a process restart, so on-device memory is persistent.
    ///
    /// (Named `load_or_create`, not `open`, because `open` is a reserved
    /// keyword in both Swift and Kotlin and would force backtick-escaping at
    /// every call site.)
    ///
    /// When existing state is found, its persisted `dim`/`vocab_cap` win and
    /// the args here are ignored; they apply only to a fresh create.
    #[uniffi::constructor]
    pub fn load_or_create(path: String, dim: u32, vocab_cap: u32, user: String) -> Result<Arc<Self>, MpError> {
        let factory = || -> Arc<dyn Extractor> { Arc::new(RegexExtractor::new()) };
        let has_state = std::path::Path::new(&path).join("service.json").exists();
        let svc = if has_state {
            MemoryService::load_state(&path, factory).map_err(MpError::from_err)?
        } else {
            MemoryService::new(factory, dim as usize, vocab_cap as usize)
        };
        Ok(Arc::new(Self { inner: Mutex::new(svc), user }))
    }

    /// Durable + ENCRYPTED engine: decrypt and load the sealed state at `path`
    /// (written by `saveSealed`), or start fresh if none exists there. `key`
    /// MUST be exactly 32 bytes and MUST match the key used to seal — a wrong
    /// key fails AEAD authentication and returns an error (no silent fallback).
    /// This is the recommended constructor for a privacy-first product.
    #[uniffi::constructor]
    pub fn load_or_create_sealed(
        path: String,
        key: Vec<u8>,
        dim: u32,
        vocab_cap: u32,
        user: String,
    ) -> Result<Arc<Self>, MpError> {
        let k = key32(&key)?;
        let factory = || -> Arc<dyn Extractor> { Arc::new(RegexExtractor::new()) };
        let has_state = std::path::Path::new(&path).join("service.json.enc").exists();
        let svc = if has_state {
            MemoryService::load_state_sealed(&path, &k, factory).map_err(MpError::from_err)?
        } else {
            MemoryService::new(factory, dim as usize, vocab_cap as usize)
        };
        Ok(Arc::new(Self { inner: Mutex::new(svc), user }))
    }

    pub fn store_fact(&self, predicate: String, value: String) -> Result<(), MpError> {
        let mut svc = self.inner.lock().unwrap();
        let pm = svc.user(&self.user).map_err(MpError::from_err)?;
        pm.store_fact(&Fact::new("user", &predicate, &value, "uniffi"))
            .map_err(MpError::from_err)
    }

    pub fn recall_fact(&self, predicate: String) -> Result<Option<String>, MpError> {
        let mut svc = self.inner.lock().unwrap();
        let pm = svc.user(&self.user).map_err(MpError::from_err)?;
        pm.recall(&predicate, None).map_err(MpError::from_err)
    }

    pub fn ingest_message(&self, message: String) -> Result<Vec<FactDto>, MpError> {
        let mut svc = self.inner.lock().unwrap();
        let pm = svc.user(&self.user).map_err(MpError::from_err)?;
        let facts = pm.ingest(&message).map_err(MpError::from_err)?;
        Ok(facts.into_iter().map(FactDto::from).collect())
    }

    pub fn forget_fact(&self, predicate: String) -> Result<bool, MpError> {
        let mut svc = self.inner.lock().unwrap();
        let pm = svc.user(&self.user).map_err(MpError::from_err)?;
        pm.forget(&predicate, None).map_err(MpError::from_err)
    }

    /// All stored facts: `{ "{subject}|{predicate}" -> value }`.
    pub fn export_user(&self) -> Result<HashMap<String, String>, MpError> {
        let mut svc = self.inner.lock().unwrap();
        let pm = svc.user(&self.user).map_err(MpError::from_err)?;
        pm.all_facts().map_err(MpError::from_err)
    }

    /// GDPR Article 17 — drop this user entirely (algebraic, residual ≈ 0).
    pub fn forget_user(&self) -> bool {
        self.inner.lock().unwrap().remove_user(&self.user)
    }

    pub fn total_facts(&self) -> u64 {
        self.inner.lock().unwrap().total_facts() as u64
    }

    /// Persist all users under `path` as a **plaintext** JSON tree. For
    /// privacy-first on-device storage use `saveSealed` instead.
    pub fn save(&self, path: String) -> Result<(), MpError> {
        self.inner.lock().unwrap().save_state(&path).map_err(MpError::from_err)
    }

    /// Encrypted-at-rest save (ChaCha20-Poly1305 AEAD): the whole on-disk
    /// footprint — values, keys, schema and service metadata — is sealed; no
    /// plaintext touches disk. `key` MUST be exactly 32 bytes; derive/store it
    /// in the iOS Keychain or Android Keystore. Pairs with `loadOrCreateSealed`.
    pub fn save_sealed(&self, path: String, key: Vec<u8>) -> Result<(), MpError> {
        let k = key32(&key)?;
        self.inner.lock().unwrap().save_state_sealed(&path, &k).map_err(MpError::from_err)
    }
}

/// Validate a caller-supplied key is exactly 32 bytes (ChaCha20-Poly1305).
fn key32(key: &[u8]) -> Result<[u8; 32], MpError> {
    <[u8; 32]>::try_from(key).map_err(|_| MpError::Memory {
        msg: format!("key must be exactly 32 bytes, got {}", key.len()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_store_recall_forget() {
        let mp = MemoryPlant::new(512, 256, "default".into());
        mp.store_fact("works_as".into(), "engineer".into()).unwrap();
        assert_eq!(mp.recall_fact("works_as".into()).unwrap(), Some("engineer".into()));
        assert!(mp.forget_fact("works_as".into()).unwrap());
        assert_eq!(mp.recall_fact("works_as".into()).unwrap(), None);
    }

    #[test]
    fn ffi_persistence_survives_restart() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap().to_string();
        // Session 1: open-fresh, store, save.
        {
            let mp = MemoryPlant::load_or_create(path.clone(), 512, 256, "default".into()).unwrap();
            mp.store_fact("lives_in".into(), "Almaty".into()).unwrap();
            mp.save(path.clone()).unwrap();
        }
        // Session 2 (simulated restart): re-open from disk → fact is still there.
        // (The engine normalises vocab values to lower-case, so "Almaty" → "almaty".)
        let mp2 = MemoryPlant::load_or_create(path.clone(), 512, 256, "default".into()).unwrap();
        assert_eq!(mp2.recall_fact("lives_in".into()).unwrap(), Some("almaty".into()));
        // Forget persists too.
        assert!(mp2.forget_fact("lives_in".into()).unwrap());
        mp2.save(path.clone()).unwrap();
        let mp3 = MemoryPlant::load_or_create(path, 512, 256, "default".into()).unwrap();
        assert_eq!(mp3.recall_fact("lives_in".into()).unwrap(), None);
    }

    #[test]
    fn ffi_sealed_persistence_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_str().unwrap().to_string();
        let key = vec![7u8; 32];
        // Session 1: open sealed-fresh, store a secret, save sealed.
        {
            let mp = MemoryPlant::load_or_create_sealed(
                path.clone(), key.clone(), 512, 256, "default".into()).unwrap();
            mp.store_fact("secret".into(), "value42".into()).unwrap();
            mp.save_sealed(path.clone(), key.clone()).unwrap();
        }
        // No plaintext on disk — only the sealed files exist.
        let p = std::path::Path::new(&path);
        assert!(!p.join("service.json").exists(), "plaintext service.json must NOT exist");
        assert!(p.join("service.json.enc").exists(), "sealed service.json.enc must exist");
        assert!(!p.join("users/default/adaptive.json").exists(), "plaintext adaptive must NOT exist");

        // Session 2 (restart) with the RIGHT key → secret restored.
        let mp2 = MemoryPlant::load_or_create_sealed(
            path.clone(), key.clone(), 512, 256, "default".into()).unwrap();
        assert_eq!(mp2.recall_fact("secret".into()).unwrap(), Some("value42".into()));

        // WRONG key → AEAD auth fails (no silent fallback).
        let wrong = vec![9u8; 32];
        assert!(MemoryPlant::load_or_create_sealed(
            path.clone(), wrong, 512, 256, "default".into()).is_err());

        // Bad key LENGTH → explicit error.
        assert!(mp2.save_sealed(path, vec![1u8; 16]).is_err());
    }
}
