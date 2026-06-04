//! At-rest encryption for persisted Memory Plant state (P3).
//!
//! ChaCha20-Poly1305 AEAD — real authenticated encryption + tamper detection.
//! NOTE: distinct from the ChaCha8 *RNG* (`rand_chacha`) used to deterministically
//! derive roles/vocab; that is key derivation, this is on-disk secrecy. Pure-Rust
//! (RustCrypto), compiles for iOS/Android/WASM. The 32-byte key is expected to live
//! in the device secure enclave / Android Keystore (see P4 sync) — this module only
//! seals/opens byte buffers and never persists the key.

use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
use rand::RngCore;

#[derive(Debug, thiserror::Error)]
pub enum CryptoError {
    #[error("decrypt failed (wrong key or tampered data)")]
    Decrypt,
    #[error("sealed blob too short")]
    TooShort,
}

const NONCE_LEN: usize = 12;

/// Encrypt `plaintext` under `key`. Layout: nonce(12) || ciphertext+tag(16).
pub fn seal(plaintext: &[u8], key: &[u8; 32]) -> Vec<u8> {
    let cipher = ChaCha20Poly1305::new(Key::from_slice(key));
    let mut nonce_bytes = [0u8; NONCE_LEN];
    rand::rng().fill_bytes(&mut nonce_bytes);
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ct = cipher
        .encrypt(nonce, plaintext)
        .expect("ChaCha20-Poly1305 encryption infallible for valid key/nonce");
    let mut out = Vec::with_capacity(NONCE_LEN + ct.len());
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ct);
    out
}

/// Decrypt a blob produced by [`seal`]. Errors on wrong key or any tampering.
pub fn open(sealed: &[u8], key: &[u8; 32]) -> Result<Vec<u8>, CryptoError> {
    if sealed.len() < NONCE_LEN {
        return Err(CryptoError::TooShort);
    }
    let (nonce_bytes, ct) = sealed.split_at(NONCE_LEN);
    let cipher = ChaCha20Poly1305::new(Key::from_slice(key));
    cipher
        .decrypt(Nonce::from_slice(nonce_bytes), ct)
        .map_err(|_| CryptoError::Decrypt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contains(hay: &[u8], needle: &[u8]) -> bool {
        hay.windows(needle.len()).any(|w| w == needle)
    }

    #[test]
    fn seal_open_roundtrip_and_not_plaintext() {
        let key = [7u8; 32];
        let msg = b"secret memory: user lives in Almaty";
        let sealed = seal(msg, &key);
        assert_ne!(sealed.as_slice(), &msg[..]);
        assert!(!contains(&sealed, b"Almaty"), "plaintext leaked into ciphertext");
        assert_eq!(open(&sealed, &key).unwrap(), msg);
    }

    #[test]
    fn wrong_key_fails() {
        let sealed = seal(b"hi there", &[1u8; 32]);
        assert!(open(&sealed, &[2u8; 32]).is_err());
    }

    #[test]
    fn tamper_fails() {
        let key = [3u8; 32];
        let mut sealed = seal(b"hello world", &key);
        let n = sealed.len();
        sealed[n - 1] ^= 0xFF; // flip a tag byte
        assert!(open(&sealed, &key).is_err());
    }

    #[test]
    fn too_short_fails() {
        assert!(matches!(open(&[0u8; 4], &[0u8; 32]), Err(CryptoError::TooShort)));
    }

    #[test]
    fn nonce_randomized_per_seal() {
        let key = [9u8; 32];
        assert_ne!(seal(b"same", &key), seal(b"same", &key)); // different nonce → different blob
    }
}
