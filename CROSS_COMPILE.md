# Memory Plant Rust — Cross-platform builds

The Rust port compiles to every major target out of the box thanks
to its minimal dependency tree (no GUI, no C++ deps in default
features, no async runtime). Verified targets and instructions:

## Native (host)

```bash
cargo build --release
# → target/release/libmemory_plant.rlib  (2.0 MB)
# → target/release/mp-mcp-server-rs       (1.8 MB)
```

## iOS

```bash
rustup target add aarch64-apple-ios aarch64-apple-ios-sim
cargo build --release --target aarch64-apple-ios --lib       # device
cargo build --release --target aarch64-apple-ios-sim --lib   # simulator (arm64)
# → 2.0 MB .rlib per target

# Universal framework (combine device + simulator):
lipo -create \
  target/aarch64-apple-ios/release/libmemory_plant.a \
  target/aarch64-apple-ios-sim/release/libmemory_plant.a \
  -output target/universal/libmemory_plant.a
```

To produce a `.framework` for Swift Package Manager:
```bash
cbindgen --crate memory-plant --output target/memory_plant.h
# wrap in xcframework — see UniFFI for automated path (Phase 7).
```

## Android

```bash
rustup target add aarch64-linux-android armv7-linux-androideabi
# Requires Android NDK installed (e.g. via Android Studio).
# Configure cargo to use the NDK toolchain:
#   ~/.cargo/config.toml:
#     [target.aarch64-linux-android]
#     linker = "/path/to/ndk/.../aarch64-linux-android21-clang"
cargo build --release --target aarch64-linux-android --lib
# → libmemory_plant.so for Android arm64 (used in JNI from Kotlin/Java)
```

## WebAssembly (browser / Node / WASI)

WASM needs a few extra knobs because:
1. `getrandom` lacks a default WASM backend (must use `wasm_js`)
2. `std::time::SystemTime` panics under wasm32-unknown-unknown
3. `std::fs` doesn't exist there
4. HTTP / blocking I/O is unavailable

Currently **not supported in default builds** — pending a Phase 7
wasm-bindgen integration which would gate `mcp_server`, `anthropic`,
`persistence`, `service` modules behind `cfg(not(target_arch = "wasm32"))`.

Hot core (`hlb`, `vocab`, `adaptive`, `audit` + math) is target-clean
and would compile under WASM once the above gates and a `getrandom`
backend flag are added.

## Linux server

```bash
rustup target add x86_64-unknown-linux-musl  # static binary
cargo build --release --target x86_64-unknown-linux-musl
# → fully static mp-mcp-server-rs (~3 MB, runs on Alpine etc.)
```

## Windows

```bash
rustup target add x86_64-pc-windows-msvc
# Requires Windows + MSVC toolchain or cross-compile from WSL.
cargo build --release --target x86_64-pc-windows-msvc
```

## Build sizes summary

| Target | `mp-mcp-server-rs` | `libmemory_plant.rlib` |
|---|---|---|
| `aarch64-apple-darwin` (Mac M-series) | 1.8 MB | 2.0 MB |
| `aarch64-apple-ios` | n/a (library) | 2.0 MB |
| `aarch64-apple-ios-sim` | n/a (library) | 2.0 MB |
| `aarch64-linux-android` | n/a (library) | ~2 MB |
| `x86_64-unknown-linux-musl` | ~3 MB | n/a |

## Feature flags

| Feature | Effect | Size impact |
|---|---|---|
| `fastembed` | Real sentence-transformer encoder via ONNX | +~10-15 MB to binary, +30 MB cached model on first run |
| (default) | MockEncoder only, RegexExtractor + AnthropicExtractor + HLB core | baseline |

```bash
cargo build --release --features fastembed
```

## CI matrix (suggested)

A minimal GitHub Actions matrix:

```yaml
strategy:
  matrix:
    target:
      - x86_64-unknown-linux-musl
      - aarch64-apple-darwin
      - aarch64-apple-ios
      - aarch64-apple-ios-sim
steps:
  - uses: actions/checkout@v4
  - uses: dtolnay/rust-toolchain@stable
    with:
      targets: ${{ matrix.target }}
  - run: cargo build --release --target ${{ matrix.target }} --lib
  - run: cargo test --release --lib    # only on host targets
```
