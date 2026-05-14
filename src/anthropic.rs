//! AnthropicExtractor — LLM-driven fact extraction via the Anthropic
//! Messages API. Mirrors Python's `extractor.AnthropicExtractor`.
//!
//! Synchronous (blocking) HTTP via `ureq` for minimal dep weight on
//! edge / on-device targets. ~200 KB binary impact vs ~5 MB for
//! reqwest+tokio. Trade-off: callers wanting concurrent extraction
//! manage their own thread pool.
//!
//! Defaults to Claude Haiku 4.5 — cheapest path that still
//! handles most production extraction quality. Override with
//! `model: "claude-sonnet-4-6"` etc.

use crate::extractor::{parse_facts_json, Extractor};
use crate::fact::Fact;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const SYSTEM_PROMPT: &str = r#"You extract structured personal facts from user messages.

Return JSON with this exact shape:
  {"facts": [
      {"subject": "...", "predicate": "...", "object": "..."},
      ...
  ]}

Field rules:
  - subject:   "user" by default, or the named entity the fact is about
  - predicate: snake_case relationship name
               (e.g. "works_as", "lives_in", "likes", "owns", "speaks")
  - object:    short canonical lowercase noun phrase

Extraction rules:
  - Extract ONLY facts EXPLICITLY stated in the text. Do not infer.
  - If no extractable facts, return {"facts": []}.
  - Output JSON only — no prose, no markdown fences, no commentary."#;

pub struct AnthropicExtractor {
    api_key: String,
    model: String,
    api_url: String,
    max_tokens: u32,
    timeout: Duration,
    /// Activity counters surfaced via stats(). Match the Python
    /// reference's `total_calls / total_tokens_in / total_tokens_out`.
    total_calls: AtomicU64,
    total_tokens_in: AtomicU64,
    total_tokens_out: AtomicU64,
}

impl AnthropicExtractor {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "claude-haiku-4-5".to_string(),
            api_url: "https://api.anthropic.com/v1/messages".to_string(),
            max_tokens: 1024,
            timeout: Duration::from_secs(30),
            total_calls: AtomicU64::new(0),
            total_tokens_in: AtomicU64::new(0),
            total_tokens_out: AtomicU64::new(0),
        }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Override the API endpoint — useful for mocking in tests and
    /// for routing through proxies (e.g. Bedrock-compatible gateways).
    pub fn with_api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = url.into();
        self
    }

    /// Snapshot of activity counters. Mirrors the Python `stats()` dict.
    pub fn stats(&self) -> ExtractorStats {
        ExtractorStats {
            total_calls: self.total_calls.load(Ordering::Relaxed),
            total_tokens_in: self.total_tokens_in.load(Ordering::Relaxed),
            total_tokens_out: self.total_tokens_out.load(Ordering::Relaxed),
        }
    }

    /// One call to the Anthropic API; parses + returns Facts.
    /// Internal — `Extractor::extract` wraps with full error swallow
    /// so a transient API failure can't crash the caller's ingest loop.
    fn call(&self, text: &str) -> Result<Vec<Fact>, String> {
        let payload = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "system": SYSTEM_PROMPT,
            "messages": [
                {
                    "role": "user",
                    "content": format!(
                        "{}\n\nReturn {{\"facts\": [...]}}.",
                        text
                    )
                }
            ]
        });
        let resp = ureq::post(&self.api_url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .config()
            .timeout_global(Some(self.timeout))
            .build()
            .send_json(&payload)
            .map_err(|e| format!("anthropic http error: {e}"))?;
        let body: serde_json::Value = resp
            .into_body()
            .read_json()
            .map_err(|e| format!("anthropic json parse: {e}"))?;

        // Tally usage.
        if let Some(usage) = body.get("usage") {
            if let Some(t) = usage.get("input_tokens").and_then(|v| v.as_u64()) {
                self.total_tokens_in.fetch_add(t, Ordering::Relaxed);
            }
            if let Some(t) = usage.get("output_tokens").and_then(|v| v.as_u64()) {
                self.total_tokens_out.fetch_add(t, Ordering::Relaxed);
            }
        }
        self.total_calls.fetch_add(1, Ordering::Relaxed);

        // Concat every text block (usually one).
        let raw = body
            .get("content")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|b| {
                        if b.get("type").and_then(|t| t.as_str()) == Some("text") {
                            b.get("text").and_then(|t| t.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("")
            })
            .unwrap_or_default();
        Ok(parse_facts_json(&raw, text))
    }
}

impl Extractor for AnthropicExtractor {
    fn extract(&self, text: &str) -> Vec<Fact> {
        match self.call(text) {
            Ok(facts) => facts,
            Err(e) => {
                eprintln!("AnthropicExtractor: {e}");
                Vec::new()
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExtractorStats {
    pub total_calls: u64,
    pub total_tokens_in: u64,
    pub total_tokens_out: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_pattern_sets_fields() {
        let ex = AnthropicExtractor::new("test-key")
            .with_model("claude-sonnet-4-6")
            .with_max_tokens(2048)
            .with_timeout(Duration::from_secs(10));
        assert_eq!(ex.model, "claude-sonnet-4-6");
        assert_eq!(ex.max_tokens, 2048);
        assert_eq!(ex.timeout, Duration::from_secs(10));
    }

    #[test]
    fn default_model_is_haiku() {
        let ex = AnthropicExtractor::new("k");
        assert_eq!(ex.model, "claude-haiku-4-5");
    }

    #[test]
    fn stats_start_zero() {
        let ex = AnthropicExtractor::new("k");
        let s = ex.stats();
        assert_eq!(s.total_calls, 0);
        assert_eq!(s.total_tokens_in, 0);
        assert_eq!(s.total_tokens_out, 0);
    }

    /// Live network test — gated behind ANTHROPIC_API_KEY env var so
    /// CI without credentials skips it. Run with:
    ///   ANTHROPIC_API_KEY=sk-... cargo test --release \
    ///     -- --ignored anthropic_live_call
    #[test]
    #[ignore]
    fn anthropic_live_call() {
        let key = std::env::var("ANTHROPIC_API_KEY")
            .expect("set ANTHROPIC_API_KEY to run");
        let ex = AnthropicExtractor::new(key);
        let facts = ex.extract("I work as a software engineer in Tokyo.");
        assert!(!facts.is_empty(), "expected at least one fact");
        let preds: Vec<_> = facts.iter().map(|f| f.predicate.as_str()).collect();
        assert!(
            preds.iter().any(|p| p.contains("work")),
            "expected works_as-like predicate, got {preds:?}"
        );
    }
}
