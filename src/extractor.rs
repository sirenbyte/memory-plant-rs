//! Extractor — maps free-form text to a list of `Fact` tuples.
//!
//! Three planned backends (only Regex implemented in Phase 2):
//!
//! - `RegexExtractor` — offline pattern matching for English + Russian
//!   common-case predicates. Free, zero deps beyond `regex`. Same
//!   shape as Python's `extractor.RegexExtractor`.
//! - `AnthropicExtractor` (Phase 5+) — Claude API via HTTP, prompt
//!   caching, retry / backoff.
//! - `SamplingExtractor` (Phase 5+) — MCP sampling — fact extraction
//!   under the host Claude Code session's auth.

use crate::fact::Fact;
use regex::Regex;
use std::sync::OnceLock;

/// Anything that can turn free-form text into structured facts.
pub trait Extractor: Send + Sync {
    fn extract(&self, text: &str) -> Vec<Fact>;
}

/// Parse a `{"facts": [...]}` JSON response into Fact tuples. Tolerant
/// of markdown fences and surrounding prose — LLM outputs often have
/// either. Shared by AnthropicExtractor and (future) sampling-based
/// extractor. Mirrors Python's `parse_facts_json`.
pub fn parse_facts_json(raw: &str, source: &str) -> Vec<Fact> {
    let stripped = raw.trim();
    // Strip optional ```json ... ``` fences.
    let inner = if let Some(rest) = stripped.strip_prefix("```json") {
        rest.trim_end_matches("```").trim()
    } else if let Some(rest) = stripped.strip_prefix("```") {
        rest.trim_end_matches("```").trim()
    } else {
        stripped
    };

    // Try direct parse first; on failure, salvage the outermost {...}.
    let payload: serde_json::Value = match serde_json::from_str(inner) {
        Ok(v) => v,
        Err(_) => {
            // Find first '{' and last '}' as a fallback.
            let start = inner.find('{');
            let end = inner.rfind('}');
            match (start, end) {
                (Some(s), Some(e)) if e > s => {
                    match serde_json::from_str(&inner[s..=e]) {
                        Ok(v) => v,
                        Err(_) => return Vec::new(),
                    }
                }
                _ => return Vec::new(),
            }
        }
    };

    let items = match payload.get("facts").and_then(|v| v.as_array()) {
        Some(arr) => arr,
        None => return Vec::new(),
    };
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let obj = item
            .get("object")
            .and_then(|v| v.as_str())
            .map(|s| s.to_lowercase().trim().to_string());
        let predicate = item
            .get("predicate")
            .and_then(|v| v.as_str())
            .map(|s| s.to_lowercase().trim().to_string());
        let subject = item
            .get("subject")
            .and_then(|v| v.as_str())
            .map(|s| s.to_lowercase().trim().to_string())
            .unwrap_or_else(|| "user".to_string());
        if let (Some(p), Some(o)) = (predicate, obj) {
            if p.is_empty() || o.is_empty() {
                continue;
            }
            out.push(Fact::new(subject, p, o, source));
        }
    }
    out
}

// ============================================================
// RegexExtractor — offline default
// ============================================================

/// Patterns mirror the Python reference. Each pattern has a single
/// capture group that becomes the `obj` of the resulting Fact.
fn patterns() -> &'static [(&'static str, &'static str, Regex)] {
    static PATS: OnceLock<Vec<(&'static str, &'static str, Regex)>> = OnceLock::new();
    PATS.get_or_init(|| {
        vec![
            (
                "user",
                "works_as",
                Regex::new(
                    r"(?i)\b(?:I\s+(?:work\s+as|am)\s+(?:an?\s+)?|я\s+(?:работаю|работаем)\s+(?:как\s+)?)([A-Za-zА-Яа-яёЁ\-]+)",
                )
                .unwrap(),
            ),
            (
                "user",
                "lives_in",
                Regex::new(
                    r"(?i)\b(?:I\s+(?:live|reside)\s+in\s+|я\s+(?:живу|нахожусь)\s+в\s+)([A-Za-zА-Яа-яёЁ\-]+)",
                )
                .unwrap(),
            ),
            (
                "user",
                "likes",
                Regex::new(
                    r"(?i)\b(?:I\s+(?:like|love|enjoy|adore)\s+|я\s+(?:люблю|обожаю|нравится)\s+)([A-Za-zА-Яа-яёЁ\-]+)",
                )
                .unwrap(),
            ),
            (
                "user",
                "owns",
                Regex::new(
                    r"(?i)\b(?:I\s+(?:own|have|drive|bought)\s+(?:an?\s+)?|я\s+(?:купил|владею|имею|езжу\s+на)\s+)([A-Za-zА-Яа-яёЁ0-9\-]+)",
                )
                .unwrap(),
            ),
            (
                "user",
                "speaks",
                Regex::new(
                    r"(?i)\b(?:I\s+(?:speak|know)\s+|я\s+(?:говорю\s+на|знаю)\s+)([A-Za-zА-Яа-яёЁ\-]+)",
                )
                .unwrap(),
            ),
        ]
        .into_iter()
        .map(|(s, p, r)| (s, p, r))
        .collect()
    })
}

#[derive(Default)]
pub struct RegexExtractor;

impl RegexExtractor {
    pub fn new() -> Self { Self }
}

impl Extractor for RegexExtractor {
    fn extract(&self, text: &str) -> Vec<Fact> {
        let mut out: Vec<Fact> = Vec::new();
        let mut seen: std::collections::HashSet<(String, String, String)> =
            std::collections::HashSet::new();
        for (subj, pred, re) in patterns().iter() {
            for cap in re.captures_iter(text) {
                let whole = cap.get(0).unwrap().as_str();
                let obj = cap.get(1).unwrap().as_str().to_lowercase();
                let tag = (subj.to_string(), pred.to_string(), obj.clone());
                if seen.contains(&tag) {
                    continue;
                }
                seen.insert(tag);
                out.push(Fact::new(*subj, *pred, obj, whole));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_works_as_en() {
        let f = RegexExtractor::new().extract("I work as engineer at Acme");
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].predicate, "works_as");
        assert_eq!(f[0].obj, "engineer");
    }

    #[test]
    fn extract_works_as_ru() {
        let f = RegexExtractor::new().extract("я работаю инженером в Алматы");
        // Pattern captures "инженером" then "Алматы" separately — only one of these
        // should be works_as. Then "Алматы" hits lives_in pattern.
        let preds: Vec<&str> = f.iter().map(|x| x.predicate.as_str()).collect();
        assert!(preds.contains(&"works_as"));
    }

    #[test]
    fn extract_lives_in() {
        let f = RegexExtractor::new().extract("I live in tokyo");
        assert!(f.iter().any(|x| x.predicate == "lives_in" && x.obj == "tokyo"));
    }

    #[test]
    fn extract_multiple_facts() {
        let f = RegexExtractor::new()
            .extract("I work as architect. I live in tokyo. I love ramen.");
        let predicates: std::collections::HashSet<_> =
            f.iter().map(|x| x.predicate.clone()).collect();
        assert!(predicates.contains("works_as"));
        assert!(predicates.contains("lives_in"));
        assert!(predicates.contains("likes"));
    }

    #[test]
    fn extract_dedups_repeated_matches() {
        let f = RegexExtractor::new()
            .extract("I work as engineer. I work as engineer.");
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn extract_empty_for_unrelated() {
        let f = RegexExtractor::new().extract("Today is a sunny day in May.");
        assert!(f.is_empty());
    }

    #[test]
    fn parse_facts_json_plain() {
        let raw = r#"{"facts":[{"subject":"user","predicate":"works_as","object":"engineer"}]}"#;
        let facts = parse_facts_json(raw, "I work as engineer");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].predicate, "works_as");
        assert_eq!(facts[0].obj, "engineer");
    }

    #[test]
    fn parse_facts_json_with_markdown_fence() {
        let raw = "```json\n{\"facts\":[{\"predicate\":\"likes\",\"object\":\"sushi\"}]}\n```";
        let facts = parse_facts_json(raw, "src");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].obj, "sushi");
    }

    #[test]
    fn parse_facts_json_salvage_from_prose() {
        let raw = "Sure thing! {\"facts\":[{\"predicate\":\"lives_in\",\"object\":\"tokyo\"}]} done.";
        let facts = parse_facts_json(raw, "");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].obj, "tokyo");
    }

    #[test]
    fn parse_facts_json_empty_facts_list() {
        let facts = parse_facts_json(r#"{"facts":[]}"#, "");
        assert!(facts.is_empty());
    }

    #[test]
    fn parse_facts_json_garbage_yields_empty() {
        let facts = parse_facts_json("lol no", "");
        assert!(facts.is_empty());
    }

    #[test]
    fn parse_facts_json_skips_missing_fields() {
        let raw = r#"{"facts":[{"subject":"user"},{"predicate":"likes","object":"pizza"}]}"#;
        let facts = parse_facts_json(raw, "");
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].obj, "pizza");
    }
}
