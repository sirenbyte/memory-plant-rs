//! OpenAIExtractor — LLM-driven fact extraction via the OpenAI
//! Chat Completions API. Mirrors Python's `extractor.OpenAIExtractor`.
//!
//! Same shape as `anthropic::AnthropicExtractor`: blocking HTTP via
//! `ureq`, synchronous Extractor trait, atomic usage counters,
//! `error → empty Vec` fail-safe. The only differences are the URL,
//! the auth header, and the response shape — extracted text comes
//! from `choices[0].message.content` instead of `content[0].text`.

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

pub struct OpenAIExtractor {
    api_key: String,
    model: String,
    api_url: String,
    max_tokens: u32,
    timeout: Duration,
    total_calls: AtomicU64,
    total_tokens_in: AtomicU64,
    total_tokens_out: AtomicU64,
}

impl OpenAIExtractor {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: "gpt-4o-mini".to_string(),
            api_url: "https://api.openai.com/v1/chat/completions".to_string(),
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
    pub fn with_max_tokens(mut self, m: u32) -> Self { self.max_tokens = m; self }
    pub fn with_timeout(mut self, t: Duration) -> Self { self.timeout = t; self }
    pub fn with_api_url(mut self, url: impl Into<String>) -> Self {
        self.api_url = url.into();
        self
    }

    pub fn stats(&self) -> crate::anthropic::ExtractorStats {
        crate::anthropic::ExtractorStats {
            total_calls: self.total_calls.load(Ordering::Relaxed),
            total_tokens_in: self.total_tokens_in.load(Ordering::Relaxed),
            total_tokens_out: self.total_tokens_out.load(Ordering::Relaxed),
        }
    }

    fn call(&self, text: &str) -> Result<Vec<Fact>, String> {
        let payload = json!({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "response_format": { "type": "json_object" },
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": format!("{}\n\nReturn {{\"facts\": [...]}}.", text)},
            ]
        });
        let resp = ureq::post(&self.api_url)
            .header("authorization", &format!("Bearer {}", self.api_key))
            .header("content-type", "application/json")
            .config()
            .timeout_global(Some(self.timeout))
            .build()
            .send_json(&payload)
            .map_err(|e| format!("openai http error: {e}"))?;
        let body: serde_json::Value = resp
            .into_body()
            .read_json()
            .map_err(|e| format!("openai json parse: {e}"))?;

        if let Some(usage) = body.get("usage") {
            if let Some(t) = usage.get("prompt_tokens").and_then(|v| v.as_u64()) {
                self.total_tokens_in.fetch_add(t, Ordering::Relaxed);
            }
            if let Some(t) = usage.get("completion_tokens").and_then(|v| v.as_u64()) {
                self.total_tokens_out.fetch_add(t, Ordering::Relaxed);
            }
        }
        self.total_calls.fetch_add(1, Ordering::Relaxed);

        let raw = body
            .get("choices")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        Ok(parse_facts_json(&raw, text))
    }
}

impl Extractor for OpenAIExtractor {
    fn extract(&self, text: &str) -> Vec<Fact> {
        match self.call(text) {
            Ok(facts) => facts,
            Err(e) => {
                eprintln!("OpenAIExtractor: {e}");
                Vec::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_to_gpt4o_mini() {
        let ex = OpenAIExtractor::new("test");
        assert_eq!(ex.model, "gpt-4o-mini");
    }

    #[test]
    fn builder_pattern() {
        let ex = OpenAIExtractor::new("test")
            .with_model("gpt-4o")
            .with_max_tokens(2048);
        assert_eq!(ex.model, "gpt-4o");
        assert_eq!(ex.max_tokens, 2048);
    }

    #[test]
    fn stats_start_zero() {
        let ex = OpenAIExtractor::new("test");
        let s = ex.stats();
        assert_eq!(s.total_calls, 0);
    }

    /// Live API test — set OPENAI_API_KEY to run.
    #[test]
    #[ignore]
    fn openai_live_call() {
        let key = std::env::var("OPENAI_API_KEY")
            .expect("set OPENAI_API_KEY to run");
        let ex = OpenAIExtractor::new(key);
        let facts = ex.extract("I work as a software engineer in Tokyo.");
        assert!(!facts.is_empty());
    }
}
