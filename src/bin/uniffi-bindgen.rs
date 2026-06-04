//! Library-mode UniFFI bindings generator. Build the crate, then:
//!   cargo run --bin uniffi-bindgen -- generate \
//!       --library target/debug/libmemory_plant.dylib --language swift  --out-dir bindings/swift
//!   cargo run --bin uniffi-bindgen -- generate \
//!       --library target/debug/libmemory_plant.dylib --language kotlin --out-dir bindings/kotlin
fn main() {
    uniffi::uniffi_bindgen_main()
}
