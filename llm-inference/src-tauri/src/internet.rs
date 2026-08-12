use serde::Deserialize;
use std::sync::atomic::{AtomicBool, Ordering};

// The user's durable runtime preference. Setup authorization is deliberately a
// separate, in-memory capability: granting Full Setup network access must not
// turn Internet on for the agent, update checks, or the next app launch.
static INTERNET_ENABLED: AtomicBool = AtomicBool::new(false);
static SETUP_INTERNET_AUTHORIZED: AtomicBool = AtomicBool::new(false);
static UPDATE_INTERNET_AUTHORIZED: AtomicBool = AtomicBool::new(false);

#[derive(Deserialize)]
struct InternetPref {
    enabled: bool,
}

pub fn init_from_disk() {
    INTERNET_ENABLED.store(load_pref(), Ordering::Relaxed);
    // A crashed or restarted setup never inherits temporary authority.
    SETUP_INTERNET_AUTHORIZED.store(false, Ordering::Relaxed);
    // Manual update access is equally session-scoped and fail-closed.
    UPDATE_INTERNET_AUTHORIZED.store(false, Ordering::Relaxed);
}

pub fn enabled() -> bool {
    INTERNET_ENABLED.load(Ordering::Relaxed)
}

pub fn set_enabled(enabled: bool) -> Result<(), String> {
    let file = pref_file();
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&file, serde_json::json!({ "enabled": enabled }).to_string())
        .map_err(|e| e.to_string())?;
    INTERNET_ENABLED.store(enabled, Ordering::Relaxed);
    Ok(())
}

/// Grant or revoke network access for first-run setup only.
///
/// This is intentionally not written to disk and intentionally does not affect
/// `enabled()`, which is the gate used by normal runtime features.
pub fn set_setup_authorized(authorized: bool) {
    SETUP_INTERNET_AUTHORIZED.store(authorized, Ordering::Relaxed);
}

pub fn setup_authorized() -> bool {
    SETUP_INTERNET_AUTHORIZED.load(Ordering::Relaxed)
}

/// Grant or revoke network access for the signed updater only. The Update
/// dialog controls this capability around an explicit Check/Install click; it
/// never changes the durable Internet preference used by the agent.
pub fn set_update_authorized(authorized: bool) {
    UPDATE_INTERNET_AUTHORIZED.store(authorized, Ordering::Relaxed);
}

pub fn update_authorized() -> bool {
    UPDATE_INTERNET_AUTHORIZED.load(Ordering::Relaxed)
}

pub fn require_setup_enabled(feature: &str) -> Result<(), String> {
    if enabled() || setup_authorized() {
        Ok(())
    } else {
        Err(format!(
            "{feature} needs temporary Internet access. Authorize it in Full Setup to continue."
        ))
    }
}

pub fn require_enabled(feature: &str) -> Result<(), String> {
    if enabled() {
        Ok(())
    } else {
        Err(format!(
            "{feature} needs Internet access. Turn it on in Settings > Internet first."
        ))
    }
}

pub fn require_update_enabled(feature: &str) -> Result<(), String> {
    if enabled() || update_authorized() {
        Ok(())
    } else {
        Err(format!(
            "{feature} needs temporary Internet access. Click Check now in Updates to authorize this request."
        ))
    }
}

fn load_pref() -> bool {
    let Ok(raw) = std::fs::read_to_string(pref_file()) else {
        return false;
    };
    serde_json::from_str::<InternetPref>(&raw)
        .map(|pref| pref.enabled)
        .unwrap_or(false)
}

fn pref_file() -> std::path::PathBuf {
    crate::paths::config_dir().join("internet.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scoped_authority_is_ephemeral_and_does_not_change_runtime_preference() {
        let prior_runtime = INTERNET_ENABLED.swap(false, Ordering::Relaxed);
        let prior_setup = SETUP_INTERNET_AUTHORIZED.swap(false, Ordering::Relaxed);
        let prior_update = UPDATE_INTERNET_AUTHORIZED.swap(false, Ordering::Relaxed);

        assert!(!enabled());
        assert!(require_setup_enabled("Full setup").is_err());

        set_setup_authorized(true);
        assert!(setup_authorized());
        assert!(require_setup_enabled("Full setup").is_ok());
        assert!(
            !enabled(),
            "setup authority leaked into the runtime preference"
        );

        set_setup_authorized(false);
        assert!(require_setup_enabled("Full setup").is_err());

        assert!(require_update_enabled("Update checks").is_err());
        set_update_authorized(true);
        assert!(update_authorized());
        assert!(require_update_enabled("Update checks").is_ok());
        assert!(
            !enabled(),
            "update authority leaked into the runtime preference"
        );
        set_update_authorized(false);
        assert!(require_update_enabled("Update checks").is_err());

        INTERNET_ENABLED.store(prior_runtime, Ordering::Relaxed);
        SETUP_INTERNET_AUTHORIZED.store(prior_setup, Ordering::Relaxed);
        UPDATE_INTERNET_AUTHORIZED.store(prior_update, Ordering::Relaxed);
    }
}
