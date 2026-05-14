//! Fact — the structured (subject, predicate, object) tuple that
//! Memory Plant stores. Mirrors Python's `extractor.Fact` dataclass.
//!
//! Convention:
//! - `subject` is `"user"` for self-facts, or an entity name for
//!   external facts (knowledge graph triplets).
//! - `predicate` is `snake_case` relation name (`works_as`,
//!   `lives_in`, `likes`).
//! - `obj` is a short canonical lowercase value that becomes a vocab
//!   entry in `AdaptiveMemory`.
//! - `source` is the raw text fragment the fact was extracted from,
//!   capped at 200 chars (audit trail).
//!
//! Storage key in HLB: `"{user_id}|{subject}|{predicate}"` →
//! `obj` (vocab idx).

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Fact {
    pub subject: String,
    pub predicate: String,
    pub obj: String,
    pub source: String,
}

impl Fact {
    pub fn new(
        subject: impl Into<String>,
        predicate: impl Into<String>,
        obj: impl Into<String>,
        source: impl Into<String>,
    ) -> Self {
        let source: String = source.into();
        let source = if source.len() > 200 {
            source[..200].to_string()
        } else {
            source
        };
        Self {
            subject: subject.into().to_lowercase(),
            predicate: predicate.into().to_lowercase(),
            obj: obj.into().to_lowercase(),
            source,
        }
    }

    /// Deterministic storage key — same shape as Python ref.
    pub fn to_key(&self, user_id: &str) -> String {
        format!("{}|{}|{}", user_id, self.subject, self.predicate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowercases_fields() {
        let f = Fact::new("USER", "Works_As", "Engineer", "I work as engineer");
        assert_eq!(f.subject, "user");
        assert_eq!(f.predicate, "works_as");
        assert_eq!(f.obj, "engineer");
    }

    #[test]
    fn source_capped() {
        let long = "x".repeat(300);
        let f = Fact::new("user", "p", "v", long);
        assert_eq!(f.source.len(), 200);
    }

    #[test]
    fn to_key_format() {
        let f = Fact::new("user", "works_as", "engineer", "");
        assert_eq!(f.to_key("alice"), "alice|user|works_as");
    }
}
