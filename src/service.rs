//! MemoryService — process-wide multi-tenant wrapper.
//!
//! Mirrors Python's `unified.MemoryService` (minus shared-vocab
//! optimization, which is a Phase 6+ concern). Each user gets their
//! own PersonalMemory + AuditTrail; the service holds them in a
//! HashMap keyed by user_id.
//!
//! Persistence: a service maps to a directory. Each user becomes a
//! subdirectory. The whole tree is JSON — see persistence.rs.

use crate::audit::AuditTrail;
use crate::extractor::Extractor;
use crate::adaptive::AdaptiveMemory;
use crate::hlb::HlbError;
use crate::personal::PersonalMemory;
use crate::persistence::PersistError;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub struct MemoryService {
    users: HashMap<String, PersonalMemory>,
    audits: HashMap<String, AuditTrail>,
    extractor_factory: Box<dyn Fn() -> Arc<dyn Extractor> + Send + Sync>,
    dim: usize,
    vocab_cap: usize,
    vocab_seed: u64,
}

impl MemoryService {
    pub fn new<F>(
        extractor_factory: F,
        dim: usize,
        vocab_cap: usize,
    ) -> Self
    where
        F: Fn() -> Arc<dyn Extractor> + Send + Sync + 'static,
    {
        Self {
            users: HashMap::new(),
            audits: HashMap::new(),
            extractor_factory: Box::new(extractor_factory),
            dim,
            vocab_cap,
            vocab_seed: 42,
        }
    }

    pub fn dim(&self) -> usize { self.dim }
    pub fn vocab_cap(&self) -> usize { self.vocab_cap }
    pub fn n_users(&self) -> usize { self.users.len() }

    /// Get-or-create the user's PersonalMemory.
    pub fn user(&mut self, uid: &str) -> Result<&mut PersonalMemory, HlbError> {
        if !self.users.contains_key(uid) {
            let am = AdaptiveMemory::new(
                self.dim,
                self.vocab_cap,
                None,
                self.vocab_seed,
            )?;
            let pm = PersonalMemory::new(uid, am, (self.extractor_factory)());
            self.users.insert(uid.to_string(), pm);
            self.audits.insert(uid.to_string(), AuditTrail::default());
        }
        Ok(self.users.get_mut(uid).unwrap())
    }

    pub fn audit(&mut self, uid: &str) -> Option<&mut AuditTrail> {
        self.audits.get_mut(uid)
    }

    /// GDPR Article 17 — drop a user entirely.
    pub fn remove_user(&mut self, uid: &str) -> bool {
        self.audits.remove(uid);
        self.users.remove(uid).is_some()
    }

    /// Snapshot user-ids list (for stats / iteration).
    pub fn user_ids(&self) -> Vec<String> {
        self.users.keys().cloned().collect()
    }

    /// Total facts across every user.
    pub fn total_facts(&self) -> usize {
        self.users.values().map(|p| p.mp.total_facts()).sum()
    }

    /// Save every user's state under `path/users/{uid}/`. Audit gets
    /// saved alongside.
    pub fn save_state(&self, path: impl AsRef<Path>) -> Result<(), PersistError> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        let users_dir = path.join("users");
        std::fs::create_dir_all(&users_dir)?;
        for (uid, pm) in &self.users {
            let user_dir = users_dir.join(safe_id(uid));
            pm.save_state(&user_dir)?;
            if let Some(audit) = self.audits.get(uid) {
                audit.save_state(&user_dir)?;
            }
        }
        // Service metadata.
        let meta = serde_json::json!({
            "dim": self.dim,
            "vocab_cap": self.vocab_cap,
            "vocab_seed": self.vocab_seed,
            "user_ids": self.user_ids(),
        });
        std::fs::write(
            path.join("service.json"),
            serde_json::to_string_pretty(&meta)?,
        )?;
        Ok(())
    }

    pub fn load_state<F>(
        path: impl AsRef<Path>,
        extractor_factory: F,
    ) -> Result<Self, PersistError>
    where
        F: Fn() -> Arc<dyn Extractor> + Send + Sync + 'static,
    {
        let path = path.as_ref();
        let meta_raw = std::fs::read_to_string(path.join("service.json"))?;
        let meta: serde_json::Value = serde_json::from_str(&meta_raw)?;
        let dim = meta["dim"].as_u64().ok_or_else(|| {
            PersistError::Corrupt("missing dim".into())
        })? as usize;
        let vocab_cap = meta["vocab_cap"].as_u64().ok_or_else(|| {
            PersistError::Corrupt("missing vocab_cap".into())
        })? as usize;
        let user_ids: Vec<String> = meta["user_ids"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let mut svc = MemoryService::new(extractor_factory, dim, vocab_cap);
        let users_dir = path.join("users");
        for uid in user_ids {
            let user_dir = users_dir.join(safe_id(&uid));
            if !user_dir.exists() {
                continue;
            }
            let pm = PersonalMemory::load_state(&user_dir, (svc.extractor_factory)())?;
            let audit = AuditTrail::load_state(&user_dir).unwrap_or_default();
            svc.users.insert(uid.clone(), pm);
            svc.audits.insert(uid, audit);
        }
        Ok(svc)
    }

    /// Encrypted-at-rest save of every user (ChaCha20-Poly1305 AEAD). Seals
    /// each user's bank + schema AND the service metadata; **no plaintext**
    /// is written. The audit trail is intentionally not persisted in sealed
    /// mode (avoids a plaintext side-channel). `key` must be 32 bytes — derive
    /// it from the OS keychain (iOS) / Keystore (Android).
    pub fn save_state_sealed(
        &self,
        path: impl AsRef<Path>,
        key: &[u8; 32],
    ) -> Result<(), PersistError> {
        let path = path.as_ref();
        std::fs::create_dir_all(path)?;
        let users_dir = path.join("users");
        std::fs::create_dir_all(&users_dir)?;
        for (uid, pm) in &self.users {
            pm.save_state_sealed(users_dir.join(safe_id(uid)), key)?;
        }
        let meta = serde_json::json!({
            "dim": self.dim,
            "vocab_cap": self.vocab_cap,
            "vocab_seed": self.vocab_seed,
            "user_ids": self.user_ids(),
        });
        let sealed = crate::crypto::seal(&serde_json::to_vec(&meta)?, key);
        std::fs::write(path.join("service.json.enc"), sealed)?;
        Ok(())
    }

    /// Restore from a tree written by `save_state_sealed`. The same `key` is
    /// required — a wrong key fails AEAD authentication (returns Corrupt).
    pub fn load_state_sealed<F>(
        path: impl AsRef<Path>,
        key: &[u8; 32],
        extractor_factory: F,
    ) -> Result<Self, PersistError>
    where
        F: Fn() -> Arc<dyn Extractor> + Send + Sync + 'static,
    {
        let path = path.as_ref();
        let sealed = std::fs::read(path.join("service.json.enc"))?;
        let raw = crate::crypto::open(&sealed, key)
            .map_err(|e| PersistError::Corrupt(e.to_string()))?;
        let meta: serde_json::Value = serde_json::from_slice(&raw)?;
        let dim = meta["dim"].as_u64().ok_or_else(|| {
            PersistError::Corrupt("missing dim".into())
        })? as usize;
        let vocab_cap = meta["vocab_cap"].as_u64().ok_or_else(|| {
            PersistError::Corrupt("missing vocab_cap".into())
        })? as usize;
        let user_ids: Vec<String> = meta["user_ids"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();

        let mut svc = MemoryService::new(extractor_factory, dim, vocab_cap);
        let users_dir = path.join("users");
        for uid in user_ids {
            let user_dir = users_dir.join(safe_id(&uid));
            if !user_dir.join("personal.json.enc").exists() {
                continue;
            }
            let pm = PersonalMemory::load_state_sealed(&user_dir, key, (svc.extractor_factory)())?;
            svc.users.insert(uid.clone(), pm);
            svc.audits.insert(uid, AuditTrail::default());
        }
        Ok(svc)
    }
}

fn safe_id(uid: &str) -> String {
    uid.chars()
        .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' || c == '.' { c } else { '_' })
        .collect()
}

/// Build a service at a stable location (env-driven) — used by the
/// MCP server binary. Returns (service, data_dir, default_user).
pub fn build_default_service() -> Result<(MemoryService, PathBuf, String), PersistError> {
    use crate::extractor::RegexExtractor;
    let data_dir: PathBuf = std::env::var("MP_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            dirs::home_dir()
                .map(|h| h.join(".memory-plant"))
                .unwrap_or_else(|| PathBuf::from("./.memory-plant"))
        });
    let default_user = std::env::var("MP_DEFAULT_USER").unwrap_or_else(|_| "default".into());
    let dim: usize = std::env::var("MP_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let vocab_cap: usize = std::env::var("MP_VOCAB_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);

    let state_dir = data_dir.join("service_state_rs");
    let factory = || -> Arc<dyn Extractor> { Arc::new(RegexExtractor::new()) };
    let svc = if state_dir.join("service.json").exists() {
        MemoryService::load_state(&state_dir, factory)?
    } else {
        MemoryService::new(factory, dim, vocab_cap)
    };
    Ok((svc, data_dir, default_user))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extractor::RegexExtractor;
    use crate::fact::Fact;

    fn build() -> MemoryService {
        MemoryService::new(
            || Arc::new(RegexExtractor::new()),
            512,
            256,
        )
    }

    #[test]
    fn user_lazy_create() {
        let mut svc = build();
        assert_eq!(svc.n_users(), 0);
        svc.user("alice").unwrap();
        assert_eq!(svc.n_users(), 1);
    }

    #[test]
    fn per_user_isolation() {
        let mut svc = build();
        svc.user("alice").unwrap().store_fact(&Fact::new("user", "p", "v_a", "")).unwrap();
        svc.user("bob").unwrap().store_fact(&Fact::new("user", "p", "v_b", "")).unwrap();
        assert_eq!(svc.user("alice").unwrap().recall("p", None).unwrap(), Some("v_a".into()));
        assert_eq!(svc.user("bob").unwrap().recall("p", None).unwrap(), Some("v_b".into()));
    }

    #[test]
    fn remove_user_clears() {
        let mut svc = build();
        svc.user("alice").unwrap().store_fact(&Fact::new("user", "p", "v", "")).unwrap();
        assert!(svc.remove_user("alice"));
        assert_eq!(svc.n_users(), 0);
    }
}
