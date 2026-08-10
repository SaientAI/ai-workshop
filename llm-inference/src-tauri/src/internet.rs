use serde::Deserialize;
use std::sync::atomic::{AtomicBool, Ordering};

static INTERNET_ENABLED: AtomicBool = AtomicBool::new(false);

#[derive(Deserialize)]
struct InternetPref {
    enabled: bool,
}

pub fn init_from_disk() {
    INTERNET_ENABLED.store(load_pref(), Ordering::Relaxed);
}

pub fn enabled() -> bool {
    INTERNET_ENABLED.load(Ordering::Relaxed)
}

pub fn set_enabled(enabled: bool) -> Result<(), String> {
    let file = pref_file();
    if let Some(parent) = file.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(
        &file,
        serde_json::json!({ "enabled": enabled }).to_string(),
    )
    .map_err(|e| e.to_string())?;
    INTERNET_ENABLED.store(enabled, Ordering::Relaxed);
    Ok(())
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
