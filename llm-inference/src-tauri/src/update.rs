//! update.rs — lightweight "is there a newer version?" check.
//!
//! Best-effort and privacy-respecting: fetches a tiny version.json from the site
//! and compares it to the running app version. We never auto-download or run
//! anything — we just tell the user and point them at the site.

use serde::Serialize;
use tauri::AppHandle;

const VERSION_URL: &str = "https://saient.co.uk/version.json";
const DOWNLOAD_URL: &str = "https://saient.co.uk/#download";

#[derive(Serialize)]
pub struct UpdateInfo {
    pub current: String,
    pub latest: String,
    pub update_available: bool,
    pub url: String,
    pub notes: String,
}

/// Parse a dotted version ("v1.2.3" / "1.2") into numeric components.
fn parse_ver(s: &str) -> Vec<u64> {
    s.trim()
        .trim_start_matches('v')
        .split('.')
        .map(|p| {
            p.chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse()
                .unwrap_or(0)
        })
        .collect()
}

fn is_newer(latest: &str, current: &str) -> bool {
    let (l, c) = (parse_ver(latest), parse_ver(current));
    for i in 0..l.len().max(c.len()) {
        let a = l.get(i).copied().unwrap_or(0);
        let b = c.get(i).copied().unwrap_or(0);
        if a != b {
            return a > b;
        }
    }
    false
}

#[tauri::command]
pub async fn check_update(app: AppHandle) -> Result<UpdateInfo, String> {
    let current = app.package_info().version.to_string();

    #[derive(serde::Deserialize)]
    struct Remote {
        version: String,
        #[serde(default)]
        url: Option<String>,
        #[serde(default)]
        notes: Option<String>,
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .map_err(|e| e.to_string())?;
    let resp = client
        .get(VERSION_URL)
        .header("User-Agent", "Saient")
        .send()
        .await
        .map_err(|e| format!("update check failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("update check: HTTP {}", resp.status()));
    }
    let r: Remote = resp
        .json()
        .await
        .map_err(|e| format!("bad version.json: {e}"))?;

    let latest = r.version.trim().to_string();
    Ok(UpdateInfo {
        update_available: is_newer(&latest, &current),
        current,
        url: r.url.unwrap_or_else(|| DOWNLOAD_URL.to_string()),
        notes: r.notes.unwrap_or_default(),
        latest,
    })
}

#[cfg(test)]
mod tests {
    use super::is_newer;
    #[test]
    fn version_compare() {
        assert!(is_newer("1.0.1", "1.0.0"));
        assert!(is_newer("1.1.0", "1.0.9"));
        assert!(is_newer("2.0", "1.9.9"));
        assert!(!is_newer("1.0.0", "1.0.0"));
        assert!(!is_newer("1.0.0", "1.0.1"));
        assert!(is_newer("v1.2.0", "1.1.0"));
    }
}
