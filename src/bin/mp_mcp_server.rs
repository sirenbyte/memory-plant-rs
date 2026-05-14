//! Entry point for the `mp-mcp-server-rs` binary.
//!
//! Reads MCP JSON-RPC requests on stdin, writes responses on stdout.
//! Configured via env vars:
//!   MP_DATA_DIR        — where to persist state (default ~/.memory-plant)
//!   MP_DEFAULT_USER    — fallback user_id when not specified
//!   MP_DIM             — HLB dimensionality (default 512)
//!   MP_VOCAB_CAP       — max distinct values (default 4096)
//!
//! Wire it into Claude Code by adding this to ~/.claude.json under
//! `mcpServers`:
//!   "memory-plant-rs": {
//!     "type": "stdio",
//!     "command": "/path/to/mp-mcp-server-rs"
//!   }

fn main() {
    if let Err(e) = memory_plant::mcp_server::run() {
        eprintln!("mp-mcp-server fatal: {e}");
        std::process::exit(1);
    }
}
