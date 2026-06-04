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
    /// New in-memory engine. Sane defaults: dim 512, vocab_cap 4096.
    #[uniffi::constructor]
    pub fn new(dim: u32, vocab_cap: u32, user: String) -> Arc<Self> {
        let factory = || -> Arc<dyn Extractor> { Arc::new(RegexExtractor::new()) };
        let svc = MemoryService::new(factory, dim as usize, vocab_cap as usize);
        Arc::new(Self { inner: Mutex::new(svc), user })
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

    /// Persist all users under `path` (plaintext JSON tree; use the redb/sealed
    /// paths in persistence.rs for encrypted on-device storage).
    pub fn save(&self, path: String) -> Result<(), MpError> {
        self.inner.lock().unwrap().save_state(&path).map_err(MpError::from_err)
    }
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
}
