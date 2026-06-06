//! auth.rs — optional local launch password (argon2).
//!
//! Scope (chosen for v1): deter casual access to someone's chats/agent. The password hash
//! lives in ~/.config/saient/auth.json; data on disk is NOT encrypted (that's a separate,
//! larger feature). The UI gates the app behind the lock screen until verify() succeeds.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use argon2::password_hash::{rand_core::OsRng, SaltString};

#[derive(Serialize, Deserialize, Default)]
struct Auth { hash: Option<String> }

fn auth_path() -> PathBuf { crate::setup::config_dir().join("auth.json") }

fn read() -> Auth {
    std::fs::read_to_string(auth_path()).ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn write(a: &Auth) {
    let p = auth_path();
    if let Some(parent) = p.parent() { let _ = std::fs::create_dir_all(parent); }
    if let Ok(s) = serde_json::to_string(a) { let _ = std::fs::write(p, s); }
}

fn hash_password(password: &str) -> Result<String, String> {
    let salt = SaltString::generate(&mut OsRng);
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| e.to_string())
}

fn verify_password(password: &str, phc: &str) -> bool {
    PasswordHash::new(phc)
        .map(|parsed| Argon2::default().verify_password(password.as_bytes(), &parsed).is_ok())
        .unwrap_or(false)
}

// ── Tauri commands ──────────────────────────────────────────────────────────────

/// Is a launch password configured?
#[tauri::command]
pub fn password_is_set() -> bool { read().hash.is_some() }

/// Set or change the password. Changing requires the current one.
#[tauri::command]
pub fn password_set(new: String, current: Option<String>) -> Result<(), String> {
    if new.trim().len() < 4 {
        return Err("Password must be at least 4 characters.".into());
    }
    let auth = read();
    if let Some(existing) = auth.hash.as_deref() {
        let cur = current.unwrap_or_default();
        if !verify_password(&cur, existing) {
            return Err("Current password is incorrect.".into());
        }
    }
    write(&Auth { hash: Some(hash_password(&new)?) });
    Ok(())
}

/// Verify a password at the lock screen. Returns true if no password is set (open).
#[tauri::command]
pub fn password_verify(password: String) -> bool {
    match read().hash {
        Some(h) => verify_password(&password, &h),
        None => true,
    }
}

/// Remove the password (requires the current one).
#[tauri::command]
pub fn password_clear(current: String) -> Result<(), String> {
    match read().hash {
        Some(h) if verify_password(&current, &h) => { write(&Auth::default()); Ok(()) }
        Some(_) => Err("Current password is incorrect.".into()),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hash_roundtrip() {
        let h = hash_password("hunter2").unwrap();
        assert!(verify_password("hunter2", &h));
        assert!(!verify_password("wrong", &h));
        assert!(!verify_password("hunter2", "not-a-phc-string"));
    }
}
