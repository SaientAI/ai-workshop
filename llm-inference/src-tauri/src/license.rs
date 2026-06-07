//! license.rs — offline 30-day trial → paid unlock via a signed license key.
//!
//! Paid keys are Ed25519-signed: the app embeds only the PUBLIC key, so a valid key
//! can be minted only by the holder of the private seed (off-repo, scripts/keygen.py).
//! Even with our binary in hand, nobody can forge a key.
//!
//! Trial state is NOT a plain JSON file you can edit or delete to reset:
//!   • It's stored as an opaque, HMAC-signed blob (secret compiled into the binary),
//!     so hand-editing any field (first_run / tier / "licensed") fails verification.
//!   • It's mirrored to two locations and merged on read (earliest first_run, latest
//!     last_seen win), so deleting one copy doesn't reset the clock.
//!   • Elapsed time is measured against a ratcheting `last_seen`, so winding the
//!     system clock back buys no extra trial days.
//! A determined reverse-engineer can still extract the secret and wipe every copy —
//! that's the ceiling for any fully-offline trial — but it is no longer a 10-second
//! "delete a json" job.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use base64::Engine;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use hmac::{Hmac, Mac};
use sha2::Sha256;

type HmacSha256 = Hmac<Sha256>;

const TRIAL_DAYS: i64 = 30;
const DAY_SECS: i64 = 86_400;

/// Ed25519 public key (generated alongside the private seed in ~/.saient-keys/).
const PUBKEY: [u8; 32] = [
    0x66, 0xfd, 0xf5, 0x1a, 0xf3, 0x1c, 0x48, 0xa1, 0xbb, 0x13, 0xbc, 0x5b, 0x8a, 0x6c, 0x1b, 0xec,
    0x77, 0x8e, 0xa1, 0x68, 0x76, 0x53, 0x16, 0x22, 0x83, 0xed, 0xd9, 0x8f, 0x11, 0x78, 0x20, 0xc6,
];

/// Secret that authenticates the local trial record. Compiled in; rotating it
/// resets everyone's trial, so leave it alone once shipped.
const STATE_SECRET: &[u8] = &[
    0xac, 0x64, 0xf4, 0xa6, 0x4e, 0x16, 0xfc, 0xdd, 0xc3, 0xcf, 0x1d, 0xc7, 0x51, 0xf5, 0xa4, 0x5b,
    0x90, 0x3c, 0x9b, 0xd1, 0xcf, 0xe2, 0xfa, 0x66, 0x59, 0x76, 0x55, 0x06, 0xc9, 0xb5, 0x0a, 0xb9,
];

#[derive(Serialize, Deserialize, Default, Clone)]
struct Record {
    first_run: Option<i64>,   // unix seconds — earliest launch
    last_seen: Option<i64>,   // unix seconds — ratchets up; anti clock-rollback
    key:       Option<String>,
}

/// Decoded license payload (what keygen.py signs).
#[derive(Deserialize)]
struct Payload {
    #[serde(default)] id:   String,
    #[serde(default)] tier: String,
}

/// State reported to the frontend.
#[derive(Serialize, Clone)]
pub struct LicenseState {
    pub status:     String,        // "trial" | "expired" | "licensed"
    pub days_left:  i64,           // trial days remaining (0 when expired)
    pub trial_days: i64,           // total trial length (30)
    pub tier:       Option<String>,
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64).unwrap_or(0)
}

fn b64() -> base64::engine::general_purpose::GeneralPurpose {
    base64::engine::general_purpose::URL_SAFE_NO_PAD
}

// ── Opaque, HMAC-signed storage (two mirrored locations) ──────────────────────────

fn state_paths() -> Vec<PathBuf> {
    let mut v = vec![crate::setup::config_dir().join("license.dat")];
    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(home) = std::env::var("HOME") {
            v.push(PathBuf::from(home).join(".saient").join("seat.dat"));
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Ok(la) = std::env::var("LOCALAPPDATA") {
            v.push(PathBuf::from(la).join("saient").join("seat.dat"));
        }
    }
    v
}

fn mac(data: &[u8]) -> Vec<u8> {
    let mut m = HmacSha256::new_from_slice(STATE_SECRET).expect("hmac key");
    m.update(data);
    m.finalize().into_bytes().to_vec()
}

/// blob = base64url(json) "." base64url(hmac(json))
fn encode(r: &Record) -> Option<String> {
    let json = serde_json::to_vec(r).ok()?;
    Some(format!("{}.{}", b64().encode(&json), b64().encode(mac(&json))))
}

fn decode(blob: &str) -> Option<Record> {
    let (d_b64, t_b64) = blob.trim().split_once('.')?;
    let data = b64().decode(d_b64).ok()?;
    let tag = b64().decode(t_b64).ok()?;
    // Constant-time-ish compare via the MAC type.
    let mut m = HmacSha256::new_from_slice(STATE_SECRET).ok()?;
    m.update(&data);
    m.verify_slice(&tag).ok()?;          // reject any hand-edited record
    serde_json::from_slice(&data).ok()
}

fn merge(into: &mut Record, other: &Record) {
    into.first_run = match (into.first_run, other.first_run) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, b) => a.or(b),
    };
    into.last_seen = match (into.last_seen, other.last_seen) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (a, b) => a.or(b),
    };
    if into.key.is_none() {
        into.key = other.key.clone();
    }
}

/// Load + merge every valid copy of the record, plus the legacy plaintext file
/// (one-time migration). Tampered/unsigned copies are ignored.
fn load_merged() -> Record {
    let mut rec = Record::default();
    for p in state_paths() {
        if let Some(r) = std::fs::read_to_string(&p).ok().and_then(|s| decode(&s)) {
            merge(&mut rec, &r);
        }
    }
    // Legacy ~/.config/saient/license.json (plaintext) → import once.
    let legacy = crate::setup::config_dir().join("license.json");
    if let Some(r) = std::fs::read_to_string(&legacy).ok().and_then(|s| serde_json::from_str::<Record>(&s).ok()) {
        merge(&mut rec, &r);
    }
    rec
}

fn save_all(r: &Record) {
    if let Some(blob) = encode(r) {
        for p in state_paths() {
            if let Some(parent) = p.parent() { let _ = std::fs::create_dir_all(parent); }
            let _ = std::fs::write(&p, &blob);
        }
    }
    // Retire the legacy plaintext file so it can't be edited to override us.
    let legacy = crate::setup::config_dir().join("license.json");
    let _ = std::fs::remove_file(legacy);
}

/// Verify a license key. Format: base64url(payload_json) "." base64url(signature).
/// The signature covers the raw decoded payload bytes. Returns the payload on success.
fn verify_key(key: &str) -> Option<Payload> {
    let (p_b64, s_b64) = key.trim().split_once('.')?;
    let payload = b64().decode(p_b64).ok()?;
    let sig_bytes = b64().decode(s_b64).ok()?;
    let vk = VerifyingKey::from_bytes(&PUBKEY).ok()?;
    let sig = Signature::from_slice(&sig_bytes).ok()?;
    vk.verify(&payload, &sig).ok()?;
    serde_json::from_slice::<Payload>(&payload).ok()
}

/// Current license state. Anchors first_run on first launch and ratchets last_seen.
pub fn status() -> LicenseState {
    // Dev (debug) builds are always unlocked — we never gate our own tinkering,
    // and the trial/paywall only applies to shipped (release) builds.
    if cfg!(debug_assertions) {
        return LicenseState {
            status: "licensed".into(), days_left: 0, trial_days: TRIAL_DAYS,
            tier: Some("dev".into()),
        };
    }
    let mut rec = load_merged();

    // Paid key wins and never expires.
    if let Some(k) = rec.key.clone() {
        if let Some(p) = verify_key(&k) {
            // Make sure the verified key is persisted in the signed store.
            if rec.first_run.is_none() { rec.first_run = Some(now()); }
            save_all(&rec);
            return LicenseState {
                status: "licensed".into(), days_left: 0, trial_days: TRIAL_DAYS,
                tier: Some(if p.tier.is_empty() { "pro".into() } else { p.tier }),
            };
        }
    }

    let n = now();
    let first = rec.first_run.unwrap_or(n);
    // Effective time can only move forward: rolling the clock back gains nothing.
    let last = rec.last_seen.unwrap_or(n).max(n);

    let dirty = rec.first_run.is_none() || rec.last_seen != Some(last);
    rec.first_run = Some(first);
    rec.last_seen = Some(last);
    if dirty { save_all(&rec); }

    let elapsed_days = (last.saturating_sub(first)).max(0) / DAY_SECS;
    let days_left = (TRIAL_DAYS - elapsed_days).max(0);
    LicenseState {
        status: if days_left > 0 { "trial".into() } else { "expired".into() },
        days_left, trial_days: TRIAL_DAYS, tier: None,
    }
}

/// Validate and store a purchased key. Returns the new (licensed) state, or an error.
pub fn activate(key: &str) -> Result<LicenseState, String> {
    if verify_key(key).is_none() {
        return Err("That license key isn't valid. Check for typos or copy it again from your email.".into());
    }
    let mut rec = load_merged();
    rec.key = Some(key.trim().to_string());
    if rec.first_run.is_none() { rec.first_run = Some(now()); }
    save_all(&rec);
    Ok(status())
}

// ── Tauri commands ──────────────────────────────────────────────────────────────

#[tauri::command]
pub fn license_status() -> LicenseState { status() }

#[tauri::command]
pub fn license_activate(key: String) -> Result<LicenseState, String> { activate(&key) }

#[cfg(test)]
mod tests {
    use super::*;
    // Minted by ~/.saient-keys/keygen.py for {"id":"test","tier":"pro"} against the embedded pubkey.
    const VALID: &str = "eyJpZCI6InRlc3QiLCJ0aWVyIjoicHJvIn0.HLrlFLuUF4ImIWC_o-LXgir3OK2c_-umYV-Cwm2JJPh623-xH20-b6hmZ2-aQLUnPoNcCTqo2pwJDHZHc9CWBQ";

    #[test]
    fn accepts_valid_and_reads_tier() {
        let p = verify_key(VALID).expect("valid key should verify");
        assert_eq!(p.tier, "pro");
    }
    #[test]
    fn rejects_tampered_key() {
        let bad = VALID.replace("HLrl", "HLrm");
        assert!(verify_key(&bad).is_none());
        assert!(verify_key("garbage").is_none());
        assert!(verify_key("no-dot-here").is_none());
    }
    #[test]
    fn record_blob_roundtrips_and_rejects_edits() {
        let r = Record { first_run: Some(1000), last_seen: Some(2000), key: None };
        let blob = encode(&r).unwrap();
        let back = decode(&blob).expect("our own blob verifies");
        assert_eq!(back.first_run, Some(1000));
        assert_eq!(back.last_seen, Some(2000));
        // Hand-edit the payload (e.g. push first_run far into the future) → MAC fails.
        let (d, t) = blob.split_once('.').unwrap();
        let mut data = b64().decode(d).unwrap();
        data[0] ^= 0xff;
        let forged = format!("{}.{}", b64().encode(&data), t);
        assert!(decode(&forged).is_none(), "edited record must be rejected");
    }
    #[test]
    fn merge_takes_earliest_first_and_latest_seen() {
        let mut a = Record { first_run: Some(500), last_seen: Some(900), key: None };
        let b = Record { first_run: Some(300), last_seen: Some(1500), key: Some("k".into()) };
        merge(&mut a, &b);
        assert_eq!(a.first_run, Some(300));   // earliest wins (can't reset by deleting newer copy)
        assert_eq!(a.last_seen, Some(1500));  // latest wins (anti-rollback)
        assert_eq!(a.key.as_deref(), Some("k"));
    }
}
