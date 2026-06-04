# Memory Plant — mobile bindings (UniFFI)

The Rust core (`MemoryService`) is exposed to **Swift (iOS)** and **Kotlin
(Android)** via [UniFFI](https://mozilla.github.io/uniffi-rs/) 0.31, proc-macro
mode (no `.udl`). The FFI surface lives in [`../src/ffi.rs`](../src/ffi.rs) and
wraps the engine in a thread-safe `MemoryPlant` object.

## What's exposed

```text
MemoryPlant(dim, vocab_cap, user)              // ctor: EMPTY, in-memory only (never touches disk)
MemoryPlant.loadOrCreate(path, dim, …, user)   // ctor: DURABLE — load from `path` or start fresh
  storeFact(predicate, value)                  // store one fact
  recallFact(predicate) -> String?             // exact recall (nil if absent; values lower-cased)
  ingestMessage(message) -> [FactDto]           // extract + store facts from text
  forgetFact(predicate) -> Bool                 // algebraic forget (GDPR)
  exportUser() -> [String: String]              // {"subject|predicate": value}
  forgetUser() -> Bool                          // drop the whole user (Art. 17)
  totalFacts() -> UInt64
  save(path)                                    // persist facts + documents (plaintext)

  // --- Document RAG (semantic search over chunked files) ---
  addDocument(docId, chunks, embeddings, metadata) -> n    // caller-embedded chunks
  search(queryEmbedding, k, metadata, containsText, minScore, docIds) -> [DocHit]
  forgetDocument(docId) -> Bool
  nDocuments() -> UInt32

chunkText(text, chunkSize, chunkOverlap) -> [String]   // free fn: split a big file

struct FactDto { subject, predicate, obj, source }
struct DocHit  { chunkId, docId, score, text, metadata }
enum  MpError  { Memory(msg) }                  // thrown as Swift `throws` / Kotlin exception
```

**Persistence (cross-session memory).** `MemoryPlant(...)` is in-memory only — its
state is lost on process exit. For durable memory that survives an app restart,
construct with `loadOrCreate(path:…)` and call `save(path)` on suspend/exit:
`loadOrCreate(path) → store/ingest/forget → save(path)` round-trips through disk.
(The factory is named `loadOrCreate`, not `open`, because `open` is a reserved
keyword in both Swift and Kotlin.)

**Encrypted-at-rest (privacy-first — recommended).** `save` writes a *plaintext*
JSON tree. For a privacy-first product use the **sealed** pair instead:
`loadOrCreateSealed(path, key, …)` + `saveSealed(path, key)`. The entire on-disk
footprint — values, keys, schema **and** service metadata — is encrypted with
**ChaCha20-Poly1305 AEAD**; no plaintext touches disk, and the audit trail is not
persisted in sealed mode (no side-channel). The `key` is exactly **32 bytes** and
must match across save/load — a wrong key fails AEAD authentication (no silent
fallback). Derive/store it in the **iOS Keychain** / **Android Keystore**; never
hardcode it. (Don't mix `save` and `saveSealed` at the same `path` — pick one.)

**Document RAG (store big files + semantic search).** The FFI does document RAG
**without bundling an embedding model** — the app supplies the vectors, keeping
the binding light and ORT-free. Flow for a large file:

1. `chunkText(text, size, overlap)` — pure-Rust word-window split (no model). Free.
2. Embed each chunk with **your own** embedder (a small on-device model, the
   device LLM, or e5 via ORT when available).
3. `addDocument(docId, chunks, embeddings, metadata)` — store them.
4. `search(queryEmbedding, k, metadata, containsText, minScore, docIds)` —
   top-k cosine with **rich out-of-box filters** (all ANDed): metadata exact
   match, case-insensitive substring, min cosine score, doc-id restriction.

Documents persist (and encrypt) alongside facts via `save`/`saveSealed`.

The heavy embedding step is deliberately the caller's job. An *internal* e5
encoder (`FastembedEncoder::multilingual`) exists in the engine behind
`--features fastembed` for host/when-ORT-is-available, but is not wired into the
default FFI. Other heavy/optional surfaces (ANN index, LLM fact-extractors) are
also out of the FFI; the default fact extractor is the offline `RegexExtractor`.

## Status (2026-06-04)

| Target | Status |
|---|---|
| Host build + `ffi` unit test | ✅ green (`cargo test ffi`) |
| Swift bindings generated | ✅ `swift/memory_plant.swift` (+ `…FFI.h`, `…FFI.modulemap`) |
| Kotlin bindings generated | ✅ `kotlin/uniffi/memory_plant/memory_plant.kt` |
| iOS device static lib (`aarch64-apple-ios`) | ✅ cross-builds (`.a`, arm64) |
| iOS simulator static lib (`aarch64-apple-ios-sim`) | ✅ cross-builds (`.a`, arm64) |
| `MemoryPlant.xcframework` (device + sim) | ✅ assembled via `xcodebuild -create-xcframework` |
| Android `.so` (NDK) | ⏳ **blocked: Android NDK not installed** (see below) |
| On-device runtime test (real phone/sim) | ⏳ not run (needs an Xcode/AS project harness) |

### Key gotcha — iOS SDK

The active developer dir is **CommandLineTools**, which ships **no iPhoneOS
SDK** → `cargo build --target aarch64-apple-ios` link step fails with
`using sysroot for 'MacOSX' but targeting 'iPhone'`. Two fixes:

1. **Per-build env (no sudo, used here):**
   `export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`
2. Or globally: `sudo xcode-select -s /Applications/Xcode.app` (needs password).

Also: build the **`staticlib`** crate-type for iOS (the `.a`), *not* `cdylib` —
a static lib is just an `ar` archive (no system-lib link step), and Xcode
supplies the iOS SDK when it links your app. The assembled `.xcframework`
carries both device + simulator slices.

## Build recipe

```sh
cd rust
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer   # iOS only

# 1. Build the host cdylib so uniffi-bindgen can read embedded metadata.
cargo build

# 2. Generate the source bindings (library mode).
cargo run --bin uniffi-bindgen -- generate \
    --library target/debug/libmemory_plant.dylib --language swift  --out-dir bindings/swift
cargo run --bin uniffi-bindgen -- generate \
    --library target/debug/libmemory_plant.dylib --language kotlin --out-dir bindings/kotlin

# 3. iOS static libs (device + simulator).
cargo rustc --target aarch64-apple-ios     --lib --crate-type staticlib
cargo rustc --target aarch64-apple-ios-sim --lib --crate-type staticlib

# 4. Headers dir for the xcframework (rename the modulemap).
mkdir -p bindings/ios/headers
cp bindings/swift/memory_plantFFI.h        bindings/ios/headers/
cp bindings/swift/memory_plantFFI.modulemap bindings/ios/headers/module.modulemap

# 5. Assemble the xcframework.
xcodebuild -create-xcframework \
  -library target/aarch64-apple-ios/debug/libmemory_plant.a     -headers bindings/ios/headers \
  -library target/aarch64-apple-ios-sim/debug/libmemory_plant.a -headers bindings/ios/headers \
  -output bindings/ios/MemoryPlant.xcframework
```

> Ship-quality: add `--release` (profile already sets `lto=fat`, `opt-level=3`,
> `codegen-units=1`) — the 200 MB debug `.a` drops to ~10 MB stripped.

## iOS usage (Swift)

Drag `MemoryPlant.xcframework` into your target, add `bindings/swift/memory_plant.swift`:

```swift
import Foundation

let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
             .appendingPathComponent("memory").path
let key: Data = loadOrCreateKeychainKey()                   // 32 bytes from Keychain

// Encrypted-at-rest (recommended). Survives restart; nothing plaintext on disk.
let mp = try MemoryPlant.loadOrCreateSealed(path: dir, key: key,
                                            dim: 512, vocabCap: 4096, user: "default")
try mp.storeFact(predicate: "works_as", value: "engineer")
let job = try mp.recallFact(predicate: "works_as")          // "engineer"

let facts = try mp.ingestMessage(message: "I live in Almaty and prefer Rust")
for f in facts { print(f.predicate, f.obj) }

try mp.forgetFact(predicate: "works_as")                    // GDPR erase
try mp.saveSealed(path: dir, key: key)                      // persist on suspend/exit
print(mp.totalFacts())

// Plaintext alternative (no key): MemoryPlant.loadOrCreate(path:…) + mp.save(path:)
```

### Document RAG over a big file (Swift)

```swift
// 1. split  2. embed (your model)  3. store  4. search
let chunks = chunkText(text: bigFileText, chunkSize: 200, chunkOverlap: 20)
let embeddings: [[Float]] = chunks.map { myEmbedder.embed($0) }   // your on-device model
_ = try mp.addDocument(docId: "report.pdf", chunks: chunks,
                       embeddings: embeddings, metadata: ["kind": "pdf"])
try mp.saveSealed(path: dir, key: key)

let q = myEmbedder.embed("what did the report say about Q3 revenue?")
let hits = mp.search(queryEmbedding: q, k: 5,
                     metadata: ["kind": "pdf"],     // filter
                     containsText: nil, minScore: 0.3, docIds: nil)
for h in hits { print(h.docId, h.score, h.text) }
```

## Android usage (Kotlin)

Add `bindings/kotlin/uniffi/memory_plant/memory_plant.kt` + the JNA dependency
(`net.java.dev.jna:jna:5.x@aar`) and bundle `libmemory_plant.so` per ABI under
`src/main/jniLibs/<abi>/`.

```kotlin
import uniffi.memory_plant.MemoryPlant

val dir = "${context.filesDir}/memory"
val key: ByteArray = loadOrCreateKeystoreKey()               // 32 bytes from Keystore

// Encrypted-at-rest (recommended). Survives restart; nothing plaintext on disk.
val mp = MemoryPlant.loadOrCreateSealed(dir, key, 512u, 4096u, "default")
mp.storeFact("works_as", "engineer")
val job = mp.recallFact("works_as")                          // "engineer"
val facts = mp.ingestMessage("I live in Almaty and prefer Rust")
mp.forgetFact("works_as")
mp.saveSealed(dir, key)                                      // persist on onStop()
mp.close()                                                   // Disposable

// Plaintext alternative (no key): MemoryPlant.loadOrCreate(dir, …) + mp.save(dir)
```

### Android `.so` — remaining work

The Android shared libs are **not built here** — the NDK is not installed on
this host. To finish:

```sh
rustup target add aarch64-linux-android armv7-linux-androideabi \
                  x86_64-linux-android i686-linux-android
# Install Android NDK (via Android Studio SDK Manager or `sdkmanager "ndk;26.x"`),
# then use cargo-ndk to set the right linker/sysroot per ABI:
cargo install cargo-ndk
cargo ndk -t arm64-v8a -t armeabi-v7a -o src/main/jniLibs build --release --lib
```

This produces `libmemory_plant.so` per ABI (cdylib). The Kotlin bindings above
load it via JNA. No code changes needed — same `ffi.rs` surface.
