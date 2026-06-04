//! Zero-blob persistence — saves a Memory Plant state to a directory.
//!
//! Key insight: every tensor in Memory Plant is **deterministically
//! reproducible** from text + seed:
//!
//! - `vocab` tensor — `random unit-norm rows` seeded by a fixed value.
//!   Save the seed, recompute on load. Zero MB on disk vs `cap*dim*4` B.
//! - `roles` — `SHA-256(key) → ChaCha8 → bipolar`. Save nothing,
//!   recompute on demand.
//! - `memory M tensors` — `Σ bind(role, vocab[idx])`. Save only the
//!   `(key, vocab_idx)` map; recompute M by replaying every store.
//!
//! Therefore the entire on-disk format is **JSON metadata** —
//! human-readable, diffable, version-controllable. No `.npy` or
//! `.safetensors` files. Persistence size scales linearly with the
//! number of facts (~50-100 bytes per fact in JSON), independent of
//! `dim`.
//!
//! Layout:
//! ```text
//! state/
//! ├── adaptive.json   — dim, shard_capacity, vocab_cap, vocab_seed,
//! │                     vocab_keys, shards[].k2v map
//! ├── personal.json   — user_id, schema  (when saving PersonalMemory)
//! └── audit.json      — bounded in-memory deque  (when saving AuditTrail)
//! ```

use crate::adaptive::AdaptiveMemory;
use crate::audit::{AuditEvent, AuditTrail};
use crate::extractor::Extractor;
use crate::hlb::HlbError;
use crate::personal::PersonalMemory;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::Arc;

#[derive(Debug, thiserror::Error)]
pub enum PersistError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("serde: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("hlb: {0}")]
    Hlb(#[from] HlbError),
    #[error("corrupted state: {0}")]
    Corrupt(String),
}

// ============================================================
// AdaptiveMemory persistence
// ============================================================

#[derive(Serialize, Deserialize)]
struct AdaptiveSnapshot {
    schema_version: u32,
    dim: usize,
    shard_capacity: usize,
    vocab_cap: usize,
    vocab_seed: u64,
    /// Ordered list of vocab values (re-registration order).
    vocab_keys: Vec<String>,
    /// Per-shard ordered (key, vocab_idx) pairs — preserves both the
    /// shard partitioning and the within-shard store order, so the
    /// replay deterministically reconstructs M.
    shards: Vec<Vec<(String, usize)>>,
}

const SCHEMA_VERSION: u32 = 1;
const ADAPTIVE_SEED_DEFAULT: u64 = 42;

impl AdaptiveMemory {
    /// Build the on-disk snapshot (shared by plaintext + sealed save).
    fn build_snapshot(&self) -> AdaptiveSnapshot {
        let shards: Vec<Vec<(String, usize)>> = (0..self.n_shards())
            .map(|i| self._shard_pairs_impl(i))
            .collect();
        AdaptiveSnapshot {
            schema_version: SCHEMA_VERSION,
            dim: self.dim(),
            shard_capacity: self.shard_capacity(),
            vocab_cap: self.vocab.cap(),
            vocab_seed: ADAPTIVE_SEED_DEFAULT,
            vocab_keys: (0..self.vocab.len())
                .filter_map(|i| self.vocab.key_at(i).map(String::from))
                .collect(),
            shards,
        }
    }

    /// Reconstruct from a snapshot (shared by plaintext + sealed load).
    /// Recomputes every tensor from scratch — vocab from seed, M from replay.
    fn from_snapshot(snap: AdaptiveSnapshot) -> Result<Self, PersistError> {
        if snap.schema_version != SCHEMA_VERSION {
            return Err(PersistError::Corrupt(format!(
                "schema {} != current {}",
                snap.schema_version, SCHEMA_VERSION
            )));
        }
        let mut am = AdaptiveMemory::new(
            snap.dim,
            snap.vocab_cap,
            Some(snap.shard_capacity),
            snap.vocab_seed,
        )?;
        // Replay every fact in original (per-shard) order; vocab re-registers
        // lazily in the same order, preserving indices.
        for shard in &snap.shards {
            for (key, vocab_idx) in shard {
                let value = snap.vocab_keys.get(*vocab_idx).ok_or_else(|| {
                    PersistError::Corrupt(format!("vocab_idx {} out of range", vocab_idx))
                })?;
                am.store(key, value)?;
            }
        }
        Ok(am)
    }

    /// Serialize to a directory (plaintext `adaptive.json`, no binary blobs).
    pub fn save_state(&self, path: impl AsRef<Path>) -> Result<(), PersistError> {
        let path = path.as_ref();
        fs::create_dir_all(path)?;
        let json = serde_json::to_string_pretty(&self.build_snapshot())?;
        fs::write(path.join("adaptive.json"), json)?;
        Ok(())
    }

    /// Restore from a plaintext directory written by save_state.
    pub fn load_state(path: impl AsRef<Path>) -> Result<Self, PersistError> {
        let path = path.as_ref();
        let raw = fs::read_to_string(path.join("adaptive.json"))?;
        let snap: AdaptiveSnapshot = serde_json::from_str(&raw)?;
        Self::from_snapshot(snap)
    }

    /// Encrypted-at-rest save: ChaCha20-Poly1305 over the snapshot bytes,
    /// written to `adaptive.json.enc`. (P3: real AEAD, not the ChaCha8 RNG.)
    pub fn save_state_sealed(
        &self,
        path: impl AsRef<Path>,
        key: &[u8; 32],
    ) -> Result<(), PersistError> {
        let path = path.as_ref();
        fs::create_dir_all(path)?;
        let json = serde_json::to_vec(&self.build_snapshot())?;
        let sealed = crate::crypto::seal(&json, key);
        fs::write(path.join("adaptive.json.enc"), sealed)?;
        Ok(())
    }

    /// Restore from a directory written by save_state_sealed.
    pub fn load_state_sealed(
        path: impl AsRef<Path>,
        key: &[u8; 32],
    ) -> Result<Self, PersistError> {
        let path = path.as_ref();
        let sealed = fs::read(path.join("adaptive.json.enc"))?;
        let json = crate::crypto::open(&sealed, key)
            .map_err(|e| PersistError::Corrupt(e.to_string()))?;
        let snap: AdaptiveSnapshot = serde_json::from_slice(&json)?;
        Self::from_snapshot(snap)
    }
}

// ============================================================
// PersonalMemory persistence
// ============================================================

#[derive(Serialize, Deserialize)]
struct PersonalSnapshot {
    schema_version: u32,
    user_id: String,
    schema: HashMap<String, Vec<String>>,
}

impl PersonalMemory {
    /// Save user_id + schema + adaptive state. Caller-provided
    /// extractor is NOT serialized (it's a runtime concern); on load
    /// you supply a fresh one.
    pub fn save_state(&self, path: impl AsRef<Path>) -> Result<(), PersistError> {
        let path = path.as_ref();
        fs::create_dir_all(path)?;
        self.mp.save_state(path)?;
        let snap = PersonalSnapshot {
            schema_version: SCHEMA_VERSION,
            user_id: self.user_id.clone(),
            schema: self.schema.clone(),
        };
        fs::write(
            path.join("personal.json"),
            serde_json::to_string_pretty(&snap)?,
        )?;
        Ok(())
    }

    pub fn load_state(
        path: impl AsRef<Path>,
        extractor: Arc<dyn Extractor>,
    ) -> Result<Self, PersistError> {
        let path = path.as_ref();
        let raw = fs::read_to_string(path.join("personal.json"))?;
        let snap: PersonalSnapshot = serde_json::from_str(&raw)?;
        if snap.schema_version != SCHEMA_VERSION {
            return Err(PersistError::Corrupt(format!(
                "personal schema {} != {}",
                snap.schema_version, SCHEMA_VERSION
            )));
        }
        let mp = AdaptiveMemory::load_state(path)?;
        Ok(PersonalMemory {
            user_id: snap.user_id,
            mp,
            extractor,
            schema: snap.schema,
        })
    }
}

// ============================================================
// AuditTrail persistence
// ============================================================

#[derive(Serialize, Deserialize)]
struct AuditSnapshot {
    schema_version: u32,
    capacity: usize,
    counter: u64,
    preview_chars: usize,
    events: Vec<AuditEvent>,
}

impl AuditTrail {
    pub fn save_state(&self, path: impl AsRef<Path>) -> Result<(), PersistError> {
        let path = path.as_ref();
        fs::create_dir_all(path)?;
        let events: Vec<AuditEvent> = self.recent(usize::MAX, None, None)
            .iter()
            .map(|e| (*e).clone())
            .rev() // back to oldest-first for consistent reload
            .collect();
        let snap = AuditSnapshot {
            schema_version: SCHEMA_VERSION,
            capacity: self.capacity(),
            counter: self.counter(),
            preview_chars: self.preview_chars(),
            events,
        };
        fs::write(path.join("audit.json"), serde_json::to_string_pretty(&snap)?)?;
        Ok(())
    }

    pub fn load_state(path: impl AsRef<Path>) -> Result<Self, PersistError> {
        let path = path.as_ref();
        let raw = fs::read_to_string(path.join("audit.json"))?;
        let snap: AuditSnapshot = serde_json::from_str(&raw)?;
        if snap.schema_version != SCHEMA_VERSION {
            return Err(PersistError::Corrupt(format!(
                "audit schema {} != {}",
                snap.schema_version, SCHEMA_VERSION
            )));
        }
        let mut t = AuditTrail::new(snap.capacity, snap.preview_chars);
        t.restore(snap.events, snap.counter);
        Ok(t)
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive::AdaptiveMemory;
    use crate::audit::EventKind;
    use crate::extractor::RegexExtractor;
    use crate::fact::Fact;
    use tempfile::TempDir;

    fn store_fixture() -> AdaptiveMemory {
        let mut am = AdaptiveMemory::new(1024, 64, None, ADAPTIVE_SEED_DEFAULT).unwrap();
        am.store("user|works_as", "engineer").unwrap();
        am.store("user|lives_in", "almaty").unwrap();
        am.store("tesla|founded_by", "elon musk").unwrap();
        am
    }

    #[test]
    fn adaptive_save_load_roundtrip() {
        let am1 = store_fixture();
        let dir = TempDir::new().unwrap();
        am1.save_state(dir.path()).unwrap();

        let am2 = AdaptiveMemory::load_state(dir.path()).unwrap();
        assert_eq!(am2.total_facts(), 3);
        assert_eq!(
            am2.retrieve("user|works_as").unwrap(),
            Some("engineer".into())
        );
        assert_eq!(
            am2.retrieve("user|lives_in").unwrap(),
            Some("almaty".into())
        );
        assert_eq!(
            am2.retrieve("tesla|founded_by").unwrap(),
            Some("elon musk".into())
        );
    }

    #[test]
    fn personal_save_load_roundtrip() {
        let am = AdaptiveMemory::new(1024, 64, None, ADAPTIVE_SEED_DEFAULT).unwrap();
        let mut pm = PersonalMemory::new("alice", am, Arc::new(RegexExtractor::new()));
        pm.store_fact(&Fact::new("user", "works_as", "engineer", ""))
            .unwrap();
        pm.store_fact(&Fact::new("user", "likes", "sushi", ""))
            .unwrap();

        let dir = TempDir::new().unwrap();
        pm.save_state(dir.path()).unwrap();

        let pm2 = PersonalMemory::load_state(dir.path(), Arc::new(RegexExtractor::new()))
            .unwrap();
        assert_eq!(pm2.user_id, "alice");
        assert_eq!(pm2.recall("works_as", None).unwrap(), Some("engineer".into()));
        assert_eq!(pm2.recall("likes", None).unwrap(), Some("sushi".into()));
        assert_eq!(pm2.schema["likes"], vec!["sushi".to_string()]);
    }

    #[test]
    fn audit_save_load_roundtrip() {
        let mut t = AuditTrail::default();
        t.record(EventKind::Store, "stored x", "alice", HashMap::new(), None)
            .unwrap();
        t.record(EventKind::Forget, "forgot y", "alice", HashMap::new(), None)
            .unwrap();

        let dir = TempDir::new().unwrap();
        t.save_state(dir.path()).unwrap();

        let t2 = AuditTrail::load_state(dir.path()).unwrap();
        assert_eq!(t2.len(), 2);
        let recent = t2.recent(10, None, None);
        assert_eq!(recent[0].kind, EventKind::Forget);
        assert_eq!(recent[1].kind, EventKind::Store);
    }

    #[test]
    fn corrupt_schema_version_errors() {
        let dir = TempDir::new().unwrap();
        fs::write(
            dir.path().join("adaptive.json"),
            r#"{"schema_version":99,"dim":64,"shard_capacity":20,"vocab_cap":4,"vocab_seed":42,"vocab_keys":[],"shards":[]}"#,
        )
        .unwrap();
        let r = AdaptiveMemory::load_state(dir.path());
        assert!(matches!(r, Err(PersistError::Corrupt(_))));
    }

    #[test]
    fn zero_blob_persistence() {
        // The whole on-disk state should be one JSON file.
        let am = store_fixture();
        let dir = TempDir::new().unwrap();
        am.save_state(dir.path()).unwrap();
        let files: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().into_string().unwrap())
            .collect();
        assert_eq!(files, vec!["adaptive.json"]);
    }

    #[test]
    fn sealed_save_load_roundtrip_and_not_plaintext() {
        let am = store_fixture();
        let key = [42u8; 32];
        let dir = TempDir::new().unwrap();
        am.save_state_sealed(dir.path(), &key).unwrap();

        // on disk it's encrypted — no plaintext schema marker leaks
        let enc = fs::read(dir.path().join("adaptive.json.enc")).unwrap();
        assert!(
            !enc.windows(14).any(|w| w == b"schema_version"),
            "plaintext leaked into the sealed blob"
        );

        // faithful round-trip: am2 produces the SAME plaintext snapshot as am
        let am2 = AdaptiveMemory::load_state_sealed(dir.path(), &key).unwrap();
        let da = TempDir::new().unwrap();
        let db = TempDir::new().unwrap();
        am.save_state(da.path()).unwrap();
        am2.save_state(db.path()).unwrap();
        assert_eq!(
            fs::read_to_string(da.path().join("adaptive.json")).unwrap(),
            fs::read_to_string(db.path().join("adaptive.json")).unwrap(),
        );

        // wrong key fails (tamper/auth)
        assert!(AdaptiveMemory::load_state_sealed(dir.path(), &[0u8; 32]).is_err());
    }
}
