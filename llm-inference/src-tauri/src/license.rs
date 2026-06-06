//! license.rs — offline 30-day trial → paid unlock via a signed license key.
//!
//! The app embeds an Ed25519 PUBLIC key and verifies keys offline; only the holder of the
//! matching private key (kept off-repo, used by scripts/keygen.py) can mint valid keys.
//! Trial state lives in ~/.config/saient/license.json. The trial is best-effort
//! (a determined user can reset it); the paid key is unforgeable.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use base64::Engine;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};

const TRIAL_DAYS: i64 = 30;
const DAY_SECS: i64 = 86_400;

/// Ed25519 public key (generated alongside the private seed in ~/.saient-keys/).
const PUBKEY: [u8; 32] = [
    0x66, 0xfd, 0xf5, 0x1a, 0xf3, 0x1c, 0x48, 0xa1, 0xbb, 0x13, 0xbc, 0x5b, 0x8a, 0x6c, 0x1b, 0xec,
    0x77, 0x8e, 0xa1, 0x68, 0x76, 0x53, 0x16, 0x22, 0x83, 0xed, 0xd9, 0x8f, 0x11, 0x78, 0x20, 0xc6,
];

#[derive(Serialize, Deserialize, Default)]
struct Record {
    first_run: Option<i64>,   // unix seconds
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

fn record_path() -> PathBuf { crate::setup::config_dir().join("license.json") }

fn read_record() -> Record {
    std::fs::read_to_string(record_path()).ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn write_record(r: &Record) {
    let p = record_path();
    if let Some(parent) = p.parent() { let _ = std::fs::create_dir_all(parent); }
    if let Ok(s) = serde_json::to_string(r) { let _ = std::fs::write(p, s); }
}

/// Verify a license key. Format: base64url(payload_json) "." base64url(signature).
/// The signature covers the raw decoded payload bytes. Returns the payload on success.
fn verify_key(key: &str) -> Option<Payload> {
    let (p_b64, s_b64) = key.trim().split_once('.')?;
    let b64 = base64::engine::general_purpose::URL_SAFE_NO_PAD;
    let payload = b64.decode(p_b64).ok()?;
    let sig_bytes = b64.decode(s_b64).ok()?;
    let vk = VerifyingKey::from_bytes(&PUBKEY).ok()?;
    let sig = Signature::from_slice(&sig_bytes).ok()?;
    vk.verify(&payload, &sig).ok()?;
    serde_json::from_slice::<Payload>(&payload).ok()
}

/// Current license state. Creates the first-run marker on first call.
pub fn status() -> LicenseState {
    let mut rec = read_record();

    // Paid key wins.
    if let Some(k) = rec.key.clone() {
        if let Some(p) = verify_key(&k) {
            return LicenseState {
                status: "licensed".into(), days_left: 0, trial_days: TRIAL_DAYS,
                tier: Some(if p.tier.is_empty() { "pro".into() } else { p.tier }),
            };
        }
    }

    // Trial: anchor first_run on first launch.
    let first = match rec.first_run {
        Some(f) => f,
        None => { let n = now(); rec.first_run = Some(n); write_record(&rec); n }
    };
    let elapsed_days = (now().saturating_sub(first)).max(0) / DAY_SECS;
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
    let mut rec = read_record();
    rec.key = Some(key.trim().to_string());
    write_record(&rec);
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
    fn rejects_tampered() {
        // flip one char in the signature
        let bad = VALID.replace("HLrl", "HLrm");
        assert!(verify_key(&bad).is_none());
        assert!(verify_key("garbage").is_none());
        assert!(verify_key("no-dot-here").is_none());
    }
}
