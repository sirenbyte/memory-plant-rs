//! MCP server — stdio JSON-RPC port of mp_mcp.py.
//!
//! Implements a synchronous, line-delimited JSON-RPC 2.0 server that
//! speaks the MCP protocol over stdin/stdout. No tokio — MCP requests
//! are processed one at a time, so a single-threaded blocking loop
//! is the simplest correct design.
//!
//! Tools exposed (Phase 5d minimal — 7 of the 21 in the Python ref):
//!
//! - mp_stats          — service-wide counters
//! - mp_recall_fact    — single-fact lookup
//! - mp_store_fact     — direct (predicate, value) write
//! - mp_ingest_message — extract facts from natural-language text
//! - mp_forget_fact    — algebraic forget single fact
//! - mp_forget_user    — GDPR Article 17 erasure
//! - mp_export_user    — GDPR Article 15 right-of-access
//!
//! The remaining 14 tools (triplets, documents, audit, sessions) can
//! be added later — same dispatch pattern.

use crate::fact::Fact;
use crate::service::{build_default_service, MemoryService};
use serde_json::{json, Value};
use std::io::{self, BufRead, Write};
use std::sync::Mutex;

const PROTOCOL_VERSION: &str = "2025-03-26";

pub struct ServerCtx {
    pub service: Mutex<MemoryService>,
    pub default_user: String,
    pub state_dir: std::path::PathBuf,
}

/// Run the MCP server loop, reading newline-delimited JSON-RPC from
/// stdin and writing responses to stdout.
pub fn run() -> io::Result<()> {
    let (svc, data_dir, default_user) =
        build_default_service().map_err(|e| io::Error::other(format!("init: {e}")))?;
    let ctx = ServerCtx {
        service: Mutex::new(svc),
        default_user,
        state_dir: data_dir.join("service_state_rs"),
    };

    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut input = stdin.lock();
    let mut line = String::new();
    loop {
        line.clear();
        match input.read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("stdin error: {e}");
                break;
            }
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("malformed JSON-RPC: {e}");
                continue;
            }
        };
        if let Some(resp) = handle_request(&req, &ctx) {
            writeln!(out, "{}", serde_json::to_string(&resp)?)?;
            out.flush()?;
        }
    }
    // Best-effort save on shutdown.
    if let Ok(svc) = ctx.service.lock() {
        let _ = svc.save_state(&ctx.state_dir);
    }
    Ok(())
}

fn handle_request(req: &Value, ctx: &ServerCtx) -> Option<Value> {
    let method = req.get("method").and_then(|v| v.as_str())?;
    let id = req.get("id").cloned();
    let params = req.get("params");

    // Notifications have no id and need no response.
    let is_notification = id.is_none();

    let result_or_err: Result<Value, McpError> = match method {
        "initialize" => Ok(initialize_result()),
        "tools/list" => Ok(tools_list_result()),
        "tools/call" => tool_call_result(params, ctx),
        "notifications/initialized" => return None,
        "ping" => Ok(json!({})),
        _ => Err(McpError::method_not_found(method)),
    };

    if is_notification {
        return None;
    }
    Some(match result_or_err {
        Ok(result) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": result,
        }),
        Err(e) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": {
                "code": e.code,
                "message": e.message,
            },
        }),
    })
}

// ============================================================
// Handshake
// ============================================================

fn initialize_result() -> Value {
    json!({
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {},
        },
        "serverInfo": {
            "name": "memory-plant-rs",
            "version": env!("CARGO_PKG_VERSION"),
        },
    })
}

// ============================================================
// Tool catalog
// ============================================================

fn tools_list_result() -> Value {
    json!({
        "tools": [
            tool_def("mp_stats",
                "Service-wide statistics — number of users, total facts.",
                json!({"type": "object", "properties": {}})),
            tool_def("mp_recall_fact",
                "Look up a stored fact by predicate. Returns null if missing.",
                json!({
                    "type": "object",
                    "properties": {
                        "predicate": {"type": "string"},
                        "subject":   {"type": "string", "description": "default 'user'"},
                        "user_id":   {"type": "string"},
                    },
                    "required": ["predicate"],
                })),
            tool_def("mp_store_fact",
                "Direct (predicate, value) write, no LLM extraction.",
                json!({
                    "type": "object",
                    "properties": {
                        "predicate": {"type": "string"},
                        "value":     {"type": "string"},
                        "subject":   {"type": "string"},
                        "user_id":   {"type": "string"},
                    },
                    "required": ["predicate", "value"],
                })),
            tool_def("mp_ingest_message",
                "Extract facts from natural language via configured extractor.",
                json!({
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "user_id": {"type": "string"},
                    },
                    "required": ["message"],
                })),
            tool_def("mp_forget_fact",
                "Algebraic forget for a single (subject, predicate) — provable, residual ≈ 0.",
                json!({
                    "type": "object",
                    "properties": {
                        "predicate": {"type": "string"},
                        "subject":   {"type": "string"},
                        "user_id":   {"type": "string"},
                    },
                    "required": ["predicate"],
                })),
            tool_def("mp_forget_user",
                "GDPR Article 17 — erase every fact for a user.",
                json!({
                    "type": "object",
                    "properties": {"user_id": {"type": "string"}},
                })),
            tool_def("mp_export_user",
                "GDPR Article 15 — return everything we know about a user.",
                json!({
                    "type": "object",
                    "properties": {"user_id": {"type": "string"}},
                })),
        ],
    })
}

fn tool_def(name: &str, description: &str, schema: Value) -> Value {
    json!({
        "name": name,
        "description": description,
        "inputSchema": schema,
    })
}

// ============================================================
// Tool dispatch
// ============================================================

fn tool_call_result(params: Option<&Value>, ctx: &ServerCtx) -> Result<Value, McpError> {
    let p = params.ok_or_else(|| McpError::invalid_params("missing params"))?;
    let name = p
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| McpError::invalid_params("missing tool name"))?;
    let args = p.get("arguments").cloned().unwrap_or(json!({}));
    let uid = args
        .get("user_id")
        .and_then(|v| v.as_str())
        .map(String::from)
        .unwrap_or_else(|| ctx.default_user.clone());

    let payload: Value = match name {
        "mp_stats" => {
            let svc = ctx.service.lock().unwrap();
            json!({
                "n_users": svc.n_users(),
                "total_facts": svc.total_facts(),
                "dim": svc.dim(),
                "vocab_cap": svc.vocab_cap(),
                "user_ids": svc.user_ids(),
            })
        }
        "mp_recall_fact" => {
            let pred = args.get("predicate").and_then(|v| v.as_str())
                .ok_or_else(|| McpError::invalid_params("predicate required"))?;
            let subj = args.get("subject").and_then(|v| v.as_str());
            let mut svc = ctx.service.lock().unwrap();
            let pm = svc.user(&uid).map_err(McpError::internal)?;
            let val = pm.recall(pred, subj).map_err(McpError::internal)?;
            json!({"predicate": pred, "value": val})
        }
        "mp_store_fact" => {
            let pred = args.get("predicate").and_then(|v| v.as_str())
                .ok_or_else(|| McpError::invalid_params("predicate required"))?;
            let value = args.get("value").and_then(|v| v.as_str())
                .ok_or_else(|| McpError::invalid_params("value required"))?;
            let subject = args.get("subject").and_then(|v| v.as_str()).unwrap_or("user");
            let mut svc = ctx.service.lock().unwrap();
            let pm = svc.user(&uid).map_err(McpError::internal)?;
            pm.store_fact(&Fact::new(subject, pred, value, "direct"))
                .map_err(McpError::internal)?;
            json!({"status": "stored"})
        }
        "mp_ingest_message" => {
            let msg = args.get("message").and_then(|v| v.as_str())
                .ok_or_else(|| McpError::invalid_params("message required"))?;
            let mut svc = ctx.service.lock().unwrap();
            let pm = svc.user(&uid).map_err(McpError::internal)?;
            let facts = pm.ingest(msg).map_err(McpError::internal)?;
            json!({
                "extracted": facts.len(),
                "facts": facts.iter().map(|f| json!({
                    "subject": f.subject, "predicate": f.predicate, "obj": f.obj,
                })).collect::<Vec<_>>(),
            })
        }
        "mp_forget_fact" => {
            let pred = args.get("predicate").and_then(|v| v.as_str())
                .ok_or_else(|| McpError::invalid_params("predicate required"))?;
            let subj = args.get("subject").and_then(|v| v.as_str());
            let mut svc = ctx.service.lock().unwrap();
            let pm = svc.user(&uid).map_err(McpError::internal)?;
            let removed = pm.forget(pred, subj).map_err(McpError::internal)?;
            json!({"removed": removed, "residual_note": "~0.001 (HLB algebraic)"})
        }
        "mp_forget_user" => {
            let mut svc = ctx.service.lock().unwrap();
            let dropped = svc.remove_user(&uid);
            json!({"user_id": uid, "dropped": dropped})
        }
        "mp_export_user" => {
            let mut svc = ctx.service.lock().unwrap();
            let pm = svc.user(&uid).map_err(McpError::internal)?;
            let facts = pm.all_facts().map_err(McpError::internal)?;
            json!({
                "user_id": uid,
                "facts": facts,
                "schema": pm.schema,
            })
        }
        other => return Err(McpError::method_not_found(other)),
    };

    // MCP wraps tool results in a content array.
    Ok(json!({
        "content": [
            {
                "type": "text",
                "text": serde_json::to_string_pretty(&payload).unwrap_or_default(),
            }
        ],
    }))
}

// ============================================================
// Errors
// ============================================================

pub struct McpError {
    pub code: i32,
    pub message: String,
}

impl McpError {
    pub fn method_not_found(method: &str) -> Self {
        Self { code: -32601, message: format!("method not found: {method}") }
    }
    pub fn invalid_params(msg: &str) -> Self {
        Self { code: -32602, message: msg.into() }
    }
    pub fn internal<E: std::fmt::Display>(e: E) -> Self {
        Self { code: -32603, message: e.to_string() }
    }
}

// ============================================================
// Tests — request/response shapes, no real stdio
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extractor::RegexExtractor;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn make_ctx() -> ServerCtx {
        let tmp = TempDir::new().unwrap();
        let svc = MemoryService::new(
            || Arc::new(RegexExtractor::new()) as Arc<dyn crate::extractor::Extractor>,
            512,
            256,
        );
        ServerCtx {
            service: Mutex::new(svc),
            default_user: "test".into(),
            state_dir: tmp.path().to_path_buf(),
        }
    }

    #[test]
    fn initialize_returns_caps() {
        let req = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {},
        });
        let ctx = make_ctx();
        let resp = handle_request(&req, &ctx).unwrap();
        assert_eq!(resp["id"], 1);
        assert_eq!(resp["result"]["protocolVersion"], PROTOCOL_VERSION);
        assert!(resp["result"]["capabilities"]["tools"].is_object());
    }

    #[test]
    fn tools_list_returns_seven() {
        let req = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list"});
        let ctx = make_ctx();
        let resp = handle_request(&req, &ctx).unwrap();
        let tools = resp["result"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 7);
    }

    #[test]
    fn store_then_recall_via_tool_call() {
        let ctx = make_ctx();
        let store = json!({
            "jsonrpc": "2.0", "id": 10, "method": "tools/call",
            "params": {"name": "mp_store_fact", "arguments": {
                "predicate": "works_as", "value": "engineer"
            }}
        });
        let r1 = handle_request(&store, &ctx).unwrap();
        assert!(r1["result"]["content"][0]["text"].as_str().unwrap().contains("stored"));

        let recall = json!({
            "jsonrpc": "2.0", "id": 11, "method": "tools/call",
            "params": {"name": "mp_recall_fact", "arguments": {
                "predicate": "works_as"
            }}
        });
        let r2 = handle_request(&recall, &ctx).unwrap();
        let txt = r2["result"]["content"][0]["text"].as_str().unwrap();
        assert!(txt.contains("engineer"));
    }

    #[test]
    fn unknown_method_yields_error() {
        let req = json!({"jsonrpc": "2.0", "id": 99, "method": "unknown/thing"});
        let ctx = make_ctx();
        let resp = handle_request(&req, &ctx).unwrap();
        assert_eq!(resp["error"]["code"], -32601);
    }

    #[test]
    fn notifications_have_no_response() {
        let req = json!({"jsonrpc": "2.0", "method": "notifications/initialized"});
        let ctx = make_ctx();
        let resp = handle_request(&req, &ctx);
        assert!(resp.is_none());
    }

    #[test]
    fn ingest_extracts_via_regex() {
        let ctx = make_ctx();
        let req = json!({
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "mp_ingest_message", "arguments": {
                "message": "I work as engineer"
            }}
        });
        let resp = handle_request(&req, &ctx).unwrap();
        let txt = resp["result"]["content"][0]["text"].as_str().unwrap();
        assert!(txt.contains("engineer"));
    }

    #[test]
    fn forget_after_store_returns_true() {
        let ctx = make_ctx();
        let s = json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{
            "name":"mp_store_fact","arguments":{"predicate":"p","value":"v"}}});
        handle_request(&s, &ctx);
        let f = json!({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{
            "name":"mp_forget_fact","arguments":{"predicate":"p"}}});
        let resp = handle_request(&f, &ctx).unwrap();
        let txt = resp["result"]["content"][0]["text"].as_str().unwrap();
        assert!(txt.contains("\"removed\": true"));
    }
}
