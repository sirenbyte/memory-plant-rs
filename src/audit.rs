//! AuditTrail — split-pattern audit log.
//!
//! Mirrors Python's `audit_memory.AuditTrail`. Records every
//! significant memory event with a categorical `kind` tag (stored in
//! the HLB layer for fast filtering) and a long-form natural-language
//! `context` (stored in-memory here; will become FAISS-indexed in
//! Phase 5 when an encoder lands).
//!
//! Architectural invariants:
//! - Event-id is unique across all calls (millisecond timestamp +
//!   monotone counter).
//! - The same event-id is used as the HLB subject AND as the future
//!   doc-id, so a forget-by-event-id will sweep both layers atomically.
//! - `_skip_audit` flag prevents the recursion that AuditTrail would
//!   otherwise trigger (audit record → HLB store → audit record ...).

use crate::adaptive::AdaptiveMemory;
use crate::hlb::HlbError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

/// Event-kind enum — same values as Python's EVENT_KIND_* constants.
/// String repr is what lands in the HLB vocab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventKind {
    Store,
    Forget,
    IngestLink,
    RejectPii,
    RejectVocab,
    RejectSchema,
    RejectStrict,
    RejectOther,
    AssistantMsg,
}

impl EventKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Store => "store",
            Self::Forget => "forget",
            Self::IngestLink => "ingest_link",
            Self::RejectPii => "reject_pii",
            Self::RejectVocab => "reject_vocab",
            Self::RejectSchema => "reject_schema",
            Self::RejectStrict => "reject_strict",
            Self::RejectOther => "reject_other",
            Self::AssistantMsg => "assistant_msg",
        }
    }
}

/// Classify a free-form validator rejection reason into a structured
/// EventKind. Same heuristic as Python's `classify_rejection`.
pub fn classify_rejection(reason: &str) -> EventKind {
    let r = reason.to_lowercase();
    if r.contains("pii") && r.contains("detected") {
        EventKind::RejectPii
    } else if r.contains("vocab") && (r.contains("cap") || r.contains("full")) {
        EventKind::RejectVocab
    } else if r.contains("not in allowed set") || r.contains("whitelist") || r.contains("predicate")
    {
        EventKind::RejectSchema
    } else if r.contains("strict_source_match") {
        EventKind::RejectStrict
    } else {
        EventKind::RejectOther
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub event_id: String,
    pub kind: EventKind,
    pub ts: f64, // unix epoch seconds
    pub user_id: String,
    pub context_preview: String,
    pub metadata: HashMap<String, String>,
}

pub struct AuditTrail {
    in_memory: VecDeque<AuditEvent>,
    capacity: usize,
    counter: u64,
    preview_chars: usize,
}

impl Default for AuditTrail {
    fn default() -> Self { Self::new(10_000, 200) }
}

impl AuditTrail {
    pub fn new(in_memory_capacity: usize, context_preview_chars: usize) -> Self {
        Self {
            in_memory: VecDeque::with_capacity(in_memory_capacity),
            capacity: in_memory_capacity,
            counter: 0,
            preview_chars: context_preview_chars,
        }
    }

    /// Append a new event. Optionally writes a HLB-layer record into
    /// the provided AdaptiveMemory under `event_id` → `kind`. The
    /// caller passes `mp: Some(&mut)` only when this event is NOT
    /// itself an audit-internal store (which would recurse).
    pub fn record(
        &mut self,
        kind: EventKind,
        context_text: &str,
        user_id: impl Into<String>,
        metadata: HashMap<String, String>,
        mp: Option<&mut AdaptiveMemory>,
    ) -> Result<AuditEvent, HlbError> {
        let user_id = user_id.into();
        self.counter += 1;
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        let event_id = format!(
            "{}_audit_{}_{:06}",
            user_id,
            (ts * 1000.0) as u64,
            self.counter
        );

        let preview = if context_text.len() > self.preview_chars {
            context_text[..self.preview_chars].to_string()
        } else {
            context_text.to_string()
        };

        // HLB layer: (event_id, "audit_kind") → kind.as_str()
        // This is the categorical compression — only ~9 distinct
        // string values ever enter the vocab regardless of how many
        // events accumulate.
        if let Some(mp_ref) = mp {
            let key = format!("{}|audit_kind", event_id);
            let _ = mp_ref.store(&key, kind.as_str())?;
        }

        let event = AuditEvent {
            event_id,
            kind,
            ts,
            user_id,
            context_preview: preview,
            metadata,
        };
        if self.in_memory.len() >= self.capacity {
            self.in_memory.pop_front();
        }
        self.in_memory.push_back(event.clone());
        Ok(event)
    }

    /// Most recent N events, optionally filtered by kind / user_id.
    pub fn recent(
        &self,
        n: usize,
        kind: Option<EventKind>,
        user_id: Option<&str>,
    ) -> Vec<&AuditEvent> {
        self.in_memory
            .iter()
            .rev()
            .filter(|e| kind.map_or(true, |k| e.kind == k))
            .filter(|e| user_id.map_or(true, |u| e.user_id == u))
            .take(n)
            .collect()
    }

    /// Histogram of events by kind. O(N events).
    pub fn count_by(&self, user_id: Option<&str>) -> HashMap<EventKind, usize> {
        let mut out = HashMap::new();
        for e in &self.in_memory {
            if user_id.map_or(true, |u| e.user_id == u) {
                *out.entry(e.kind).or_insert(0) += 1;
            }
        }
        out
    }

    pub fn len(&self) -> usize { self.in_memory.len() }
    pub fn is_empty(&self) -> bool { self.in_memory.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mp() -> AdaptiveMemory {
        AdaptiveMemory::new(512, 64, None, 42).unwrap()
    }

    #[test]
    fn record_grows_log() {
        let mut t = AuditTrail::default();
        let mut mp = make_mp();
        t.record(EventKind::Store, "stored fact x", "alice", HashMap::new(), Some(&mut mp))
            .unwrap();
        t.record(EventKind::Store, "stored fact y", "alice", HashMap::new(), Some(&mut mp))
            .unwrap();
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn recent_filters_by_kind() {
        let mut t = AuditTrail::default();
        t.record(EventKind::Store, "s", "u", HashMap::new(), None).unwrap();
        t.record(EventKind::Forget, "f", "u", HashMap::new(), None).unwrap();
        t.record(EventKind::Store, "s2", "u", HashMap::new(), None).unwrap();
        let stores = t.recent(10, Some(EventKind::Store), None);
        assert_eq!(stores.len(), 2);
        assert!(stores.iter().all(|e| e.kind == EventKind::Store));
    }

    #[test]
    fn count_by_groups() {
        let mut t = AuditTrail::default();
        t.record(EventKind::Store, "", "u", HashMap::new(), None).unwrap();
        t.record(EventKind::Store, "", "u", HashMap::new(), None).unwrap();
        t.record(EventKind::Forget, "", "u", HashMap::new(), None).unwrap();
        let counts = t.count_by(None);
        assert_eq!(counts[&EventKind::Store], 2);
        assert_eq!(counts[&EventKind::Forget], 1);
    }

    #[test]
    fn capacity_evicts_oldest() {
        let mut t = AuditTrail::new(3, 50);
        for i in 0..5 {
            t.record(
                EventKind::Store,
                &format!("evt{i}"),
                "u",
                HashMap::new(),
                None,
            ).unwrap();
        }
        assert_eq!(t.len(), 3); // oldest 2 dropped
        let r = t.recent(10, None, None);
        // Newest first — evt4, evt3, evt2.
        assert_eq!(r[0].context_preview, "evt4");
        assert_eq!(r[2].context_preview, "evt2");
    }

    #[test]
    fn classify_rejection_pii() {
        assert_eq!(classify_rejection("PII detected: email"), EventKind::RejectPii);
        assert_eq!(
            classify_rejection("vocab cap reached"),
            EventKind::RejectVocab
        );
        assert_eq!(
            classify_rejection("strict_source_match: obj not found"),
            EventKind::RejectStrict
        );
        assert_eq!(classify_rejection("something else"), EventKind::RejectOther);
    }

    #[test]
    fn hlb_layer_records_categorical_only() {
        let mut t = AuditTrail::default();
        let mut mp = make_mp();
        for i in 0..50 {
            t.record(
                EventKind::Store,
                &format!("context for event {i}"),
                "u",
                HashMap::new(),
                Some(&mut mp),
            )
            .unwrap();
        }
        // Only ONE distinct vocab value entered: "store".
        // event_ids are subjects (not in vocab), kind is the value.
        // So vocab should have exactly 1 entry.
        assert_eq!(mp.vocab.len(), 1);
        assert_eq!(mp.total_facts(), 50);
    }
}
