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
  save(path)                                    // persist (plaintext JSON tree)

struct FactDto { subject, predicate, obj, source }
enum  MpError  { Memory(msg) }                  // thrown as Swift `throws` / Kotlin exception
```

**Persistence (cross-session memory).** `MemoryPlant(...)` is in-memory only — its
state is lost on process exit. For durable memory that survives an app restart,
construct with `loadOrCreate(path:…)` and call `save(path)` on suspend/exit:
`loadOrCreate(path) → store/ingest/forget → save(path)` round-trips through disk.
(The factory is named `loadOrCreate`, not `open`, because `open` is a reserved
keyword in both Swift and Kotlin.) Today `save` writes a plaintext JSON tree;
encrypted-at-rest (`crypto.rs` AEAD / redb) is wired in the engine and is the
next step to expose through the FFI.

Heavy/optional surfaces (fastembed/ORT embeddings, ANN index, the LLM
extractors, document semantic-search) are **intentionally not** in the FFI — the
mobile binding stays light and cross-compiles without ONNX Runtime. The default
extractor is the offline `RegexExtractor`.

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

// Durable: survives app restart. Point at the app's Application Support dir.
let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
             .appendingPathComponent("memory").path
let mp = try MemoryPlant.loadOrCreate(path: dir, dim: 512, vocabCap: 4096, user: "default")

try mp.storeFact(predicate: "works_as", value: "engineer")
let job = try mp.recallFact(predicate: "works_as")          // "engineer"

let facts = try mp.ingestMessage(message: "I live in Almaty and prefer Rust")
for f in facts { print(f.predicate, f.obj) }

try mp.forgetFact(predicate: "works_as")                    // GDPR erase
try mp.save(path: dir)                                      // persist on suspend/exit
print(mp.totalFacts())
```

## Android usage (Kotlin)

Add `bindings/kotlin/uniffi/memory_plant/memory_plant.kt` + the JNA dependency
(`net.java.dev.jna:jna:5.x@aar`) and bundle `libmemory_plant.so` per ABI under
`src/main/jniLibs/<abi>/`.

```kotlin
import uniffi.memory_plant.MemoryPlant

// Durable: survives app restart. Use the app's filesDir.
val dir = "${context.filesDir}/memory"
val mp = MemoryPlant.loadOrCreate(dir, 512u, 4096u, "default")
mp.storeFact("works_as", "engineer")
val job = mp.recallFact("works_as")                          // "engineer"
val facts = mp.ingestMessage("I live in Almaty and prefer Rust")
mp.forgetFact("works_as")
mp.save(dir)                                                 // persist on onStop()
mp.close()                                                   // Disposable
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
