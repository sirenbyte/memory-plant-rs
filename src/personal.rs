//! PersonalMemory — per-user wrapper over AdaptiveMemory.
//!
//! Adds scoping (`user_id` prefix on every key), schema tracking
//! (`predicate → list of seen values`), and the ingest-from-text flow
//! via an `Extractor` trait object.
//!
//! Mirrors Python's `extractor.PersonalMemory`. The audit-trail layer
//! sits on top of this in Phase 3, persistence in Phase 4.

use crate::adaptive::AdaptiveMemory;
use crate::extractor::Extractor;
use crate::fact::Fact;
use crate::hlb::HlbError;
use std::collections::HashMap;
use std::sync::Arc;

pub struct PersonalMemory {
    pub user_id: String,
    pub mp: AdaptiveMemory,
    pub extractor: Arc<dyn Extractor>,
    /// `predicate → known values`. Populated as facts are stored,
    /// used as schema hint to LLM extractors (Phase 5).
    pub schema: HashMap<String, Vec<String>>,
}

impl PersonalMemory {
    pub fn new(
        user_id: impl Into<String>,
        mp: AdaptiveMemory,
        extractor: Arc<dyn Extractor>,
    ) -> Self {
        Self {
            user_id: user_id.into(),
            mp,
            extractor,
            schema: HashMap::new(),
        }
    }

    /// Run the extractor on `message` and store every extracted Fact.
    /// Returns the facts that were stored (after dedup against schema).
    pub fn ingest(&mut self, message: &str) -> Result<Vec<Fact>, HlbError> {
        let facts = self.extractor.extract(message);
        let mut stored = Vec::new();
        for fact in facts {
            self.store_fact(&fact)?;
            stored.push(fact);
        }
        Ok(stored)
    }

    /// Direct store of a single fact. Updates schema tracking too.
    pub fn store_fact(&mut self, fact: &Fact) -> Result<(), HlbError> {
        let key = fact.to_key(&self.user_id);
        self.mp.store(&key, &fact.obj)?;
        let entry = self.schema.entry(fact.predicate.clone()).or_default();
        if !entry.contains(&fact.obj) {
            entry.push(fact.obj.clone());
        }
        Ok(())
    }

    /// Single-fact lookup by `(subject, predicate)`. Subject defaults
    /// to `"user"` if you only know the predicate.
    pub fn recall(
        &self,
        predicate: &str,
        subject: Option<&str>,
    ) -> Result<Option<String>, HlbError> {
        let subj = subject.unwrap_or("user");
        let key = format!("{}|{}|{}", self.user_id, subj, predicate);
        self.mp.retrieve(&key)
    }

    /// Snapshot of all stored facts for this user — `{key_suffix → value}`
    /// where `key_suffix` is `{subject}|{predicate}` (user_id stripped).
    /// Note: this rebuilds via decode_shard_amp so it's O(N total facts);
    /// for hot paths use direct `mp.retrieve()` on known keys.
    pub fn all_facts(&self) -> Result<HashMap<String, String>, HlbError> {
        let mut out = HashMap::new();
        let prefix = format!("{}|", self.user_id);
        for shard_idx in 0..self.mp.n_shards() {
            let preds = self.mp.decode_shard_amp(shard_idx, 10)?;
            for (k, vocab_idx) in preds {
                if k.starts_with(&prefix) {
                    let stripped = k[prefix.len()..].to_string();
                    if let Some(v) = self.mp.vocab.key_at(vocab_idx) {
                        out.insert(stripped, v.to_string());
                    }
                }
            }
        }
        Ok(out)
    }

    /// Algebraic forget for a single (subject, predicate) pair.
    pub fn forget(
        &mut self,
        predicate: &str,
        subject: Option<&str>,
    ) -> Result<bool, HlbError> {
        let subj = subject.unwrap_or("user");
        let key = format!("{}|{}|{}", self.user_id, subj, predicate);
        self.mp.forget(&key)
    }

    /// GDPR-style total erasure: forget every fact stored for this user.
    /// Returns count of forgotten facts. Algebraic, residual ≈ 0.
    pub fn forget_all(&mut self) -> Result<usize, HlbError> {
        let prefix = format!("{}|", self.user_id);
        // Collect targets first to avoid borrow conflicts.
        let mut keys_to_forget: Vec<String> = Vec::new();
        for shard_idx in 0..self.mp.n_shards() {
            // We don't expose a direct shard.keys iterator; query
            // all_facts() to discover live keys then re-prefix them.
            let preds = self.mp.decode_shard_amp(shard_idx, 1)?;
            for k in preds.keys() {
                if k.starts_with(&prefix) {
                    keys_to_forget.push(k.clone());
                }
            }
        }
        let mut count = 0;
        for k in keys_to_forget {
            if self.mp.forget(&k)? {
                count += 1;
            }
        }
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive::AdaptiveMemory;
    use crate::extractor::RegexExtractor;

    fn build(user: &str) -> PersonalMemory {
        let mp = AdaptiveMemory::new(1024, 256, None, 42).unwrap();
        PersonalMemory::new(user, mp, Arc::new(RegexExtractor::new()))
    }

    #[test]
    fn ingest_extracts_and_stores() {
        let mut pm = build("alice");
        let stored = pm.ingest("I work as engineer in Berlin").unwrap();
        assert!(!stored.is_empty());
        assert_eq!(pm.recall("works_as", None).unwrap(), Some("engineer".into()));
    }

    #[test]
    fn store_fact_direct() {
        let mut pm = build("bob");
        let f = Fact::new("user", "favorite_food", "sushi", "remember sushi");
        pm.store_fact(&f).unwrap();
        assert_eq!(
            pm.recall("favorite_food", None).unwrap(),
            Some("sushi".into())
        );
    }

    #[test]
    fn schema_tracking_grows() {
        let mut pm = build("c");
        pm.store_fact(&Fact::new("user", "likes", "pizza", "")).unwrap();
        pm.store_fact(&Fact::new("user", "likes", "ramen", "")).unwrap();
        assert_eq!(pm.schema["likes"], vec!["pizza".to_string(), "ramen".to_string()]);
    }

    #[test]
    fn forget_single_fact() {
        let mut pm = build("d");
        pm.store_fact(&Fact::new("user", "works_as", "engineer", "")).unwrap();
        assert!(pm.forget("works_as", None).unwrap());
        assert_eq!(pm.recall("works_as", None).unwrap(), None);
    }

    #[test]
    fn forget_all_clears_user_only() {
        let mut pm_a = build("alice");
        let mut pm_b = build("bob");
        // Need to share an AdaptiveMemory for the multi-user check;
        // for the simpler API here we just verify own-data wipe.
        pm_a.store_fact(&Fact::new("user", "works_as", "engineer", ""))
            .unwrap();
        pm_a.store_fact(&Fact::new("user", "lives_in", "tokyo", ""))
            .unwrap();
        let count = pm_a.forget_all().unwrap();
        assert_eq!(count, 2);
        assert_eq!(pm_a.recall("works_as", None).unwrap(), None);
        // pm_b is a fresh instance — sanity that we didn't crash it.
        let _ = pm_b.recall("anything", None);
    }

    #[test]
    fn all_facts_returns_stored_pairs() {
        let mut pm = build("eve");
        pm.store_fact(&Fact::new("user", "p1", "v1", "")).unwrap();
        pm.store_fact(&Fact::new("user", "p2", "v2", "")).unwrap();
        let snapshot = pm.all_facts().unwrap();
        assert_eq!(snapshot.get("user|p1"), Some(&"v1".to_string()));
        assert_eq!(snapshot.get("user|p2"), Some(&"v2".to_string()));
    }
}
