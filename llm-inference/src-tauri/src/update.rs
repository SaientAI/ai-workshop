//! Signed, Pi-hosted desktop updates.
//!
//! The public site provides lightweight version metadata for compatibility with
//! older clients and signed release metadata for installation. Debian/Ubuntu
//! uses Saient's Ed25519 manifest, SHA-256 verification and apt. Windows uses
//! Tauri's signed updater and a current-user NSIS installer.

use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

const VERSION_URL: &str = "https://saient.co.uk/version.json";
const MANIFEST_URL: &str = "https://saient.co.uk/release-manifest.json";
const DOWNLOAD_URL: &str = "https://saient.co.uk/#download";
const UPDATE_PUBLIC_KEY_B64: &str = "ttgUn+oifVXNf001dm4HDfId1pCCkwCSSI7ju5aa8lo=";
const MAX_UPDATE_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Serialize)]
pub struct UpdateInfo {
    pub current: String,
    pub latest: String,
    pub update_available: bool,
    pub install_supported: bool,
    pub url: String,
    pub notes: String,
}

#[derive(Clone, Serialize)]
pub struct UpdateProgress {
    pub phase: String,
    pub downloaded: u64,
    pub total: u64,
    pub message: String,
}

#[derive(Debug, Deserialize)]
struct ReleaseManifest {
    release: String,
    artifact: ReleaseArtifact,
}

#[derive(Debug, Deserialize)]
struct ReleaseArtifact {
    platform: String,
    url: String,
    bytes: u64,
    sha256: String,
    signature: String,
}

#[derive(Serialize)]
pub struct InstallUpdateResult {
    pub version: String,
    pub restart_required: bool,
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

fn valid_version(version: &str) -> bool {
    let version = version.trim().trim_start_matches('v');
    !version.is_empty()
        && version.len() <= 32
        && version.split('.').all(|part| {
            !part.is_empty() && part.len() <= 10 && part.chars().all(|c| c.is_ascii_digit())
        })
}

fn update_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(30 * 60))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|e| e.to_string())
}

fn signed_message(manifest: &ReleaseManifest) -> String {
    format!(
        "saient-update-v1\nrelease={}\nplatform={}\nurl={}\nbytes={}\nsha256={}\n",
        manifest.release.trim(),
        manifest.artifact.platform.trim(),
        manifest.artifact.url.trim(),
        manifest.artifact.bytes,
        manifest.artifact.sha256.trim().to_ascii_lowercase(),
    )
}

fn verify_manifest_signature_with_key(
    manifest: &ReleaseManifest,
    public_key_b64: &str,
) -> Result<(), String> {
    let public_key: [u8; 32] = BASE64
        .decode(public_key_b64)
        .map_err(|_| "update public key is invalid".to_string())?
        .try_into()
        .map_err(|_| "update public key has the wrong length".to_string())?;
    let signature: [u8; 64] = BASE64
        .decode(manifest.artifact.signature.trim())
        .map_err(|_| "update signature is invalid".to_string())?
        .try_into()
        .map_err(|_| "update signature has the wrong length".to_string())?;

    let verifying_key = VerifyingKey::from_bytes(&public_key)
        .map_err(|_| "update public key could not be loaded".to_string())?;
    verifying_key
        .verify(
            signed_message(manifest).as_bytes(),
            &Signature::from_bytes(&signature),
        )
        .map_err(|_| "update signature verification failed".to_string())
}

fn validate_manifest(manifest: &ReleaseManifest, current: &str) -> Result<(), String> {
    if !valid_version(&manifest.release) {
        return Err("update manifest contains an invalid version".into());
    }
    if !is_newer(&manifest.release, current) {
        return Err("the published release is not newer than this Saient version".into());
    }
    if manifest.artifact.platform != "Debian/Ubuntu amd64" {
        return Err("the published update is not a Debian/Ubuntu amd64 package".into());
    }
    if manifest.artifact.bytes == 0 || manifest.artifact.bytes > MAX_UPDATE_BYTES {
        return Err("the published update size is invalid".into());
    }
    let sha = manifest.artifact.sha256.trim();
    if sha.len() != 64 || !sha.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err("the published update SHA-256 is invalid".into());
    }

    let url = reqwest::Url::parse(manifest.artifact.url.trim())
        .map_err(|_| "the published update URL is invalid".to_string())?;
    if url.scheme() != "https"
        || url.host_str() != Some("saient.co.uk")
        || url.port().is_some()
        || !url.path().starts_with("/downloads/")
        || !url.path().ends_with(".deb")
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err("the update is not hosted on the approved saient.co.uk download path".into());
    }

    verify_manifest_signature_with_key(manifest, UPDATE_PUBLIC_KEY_B64)
}

async fn fetch_manifest(client: &reqwest::Client) -> Result<ReleaseManifest, String> {
    let response = client
        .get(MANIFEST_URL)
        .header("User-Agent", "Saient")
        .send()
        .await
        .map_err(|e| format!("update manifest request failed: {e}"))?;
    if !response.status().is_success() {
        return Err(format!("update manifest: HTTP {}", response.status()));
    }
    response
        .json::<ReleaseManifest>()
        .await
        .map_err(|e| format!("invalid update manifest: {e}"))
}

fn emit_progress(app: &AppHandle, phase: &str, downloaded: u64, total: u64, message: &str) {
    let _ = app.emit(
        "update-progress",
        UpdateProgress {
            phase: phase.to_string(),
            downloaded,
            total,
            message: message.to_string(),
        },
    );
}

async fn remove_quietly(path: &Path) {
    let _ = tokio::fs::remove_file(path).await;
}

async fn download_and_verify(
    app: &AppHandle,
    client: &reqwest::Client,
    manifest: &ReleaseManifest,
) -> Result<PathBuf, String> {
    let cache_dir = app
        .path()
        .app_cache_dir()
        .map_err(|e| format!("could not locate the update cache: {e}"))?
        .join("updates");
    tokio::fs::create_dir_all(&cache_dir)
        .await
        .map_err(|e| format!("could not create the update cache: {e}"))?;

    let filename = format!("Saient_{}_amd64.deb", manifest.release.trim());
    let final_path = cache_dir.join(filename);
    let partial_path = final_path.with_extension("deb.part");
    remove_quietly(&partial_path).await;

    emit_progress(
        app,
        "downloading",
        0,
        manifest.artifact.bytes,
        "Downloading the signed update from saient.co.uk…",
    );
    let response = client
        .get(manifest.artifact.url.trim())
        .header("User-Agent", "Saient")
        .send()
        .await
        .map_err(|e| format!("update download failed: {e}"))?;
    if !response.status().is_success() {
        return Err(format!("update download: HTTP {}", response.status()));
    }
    if let Some(length) = response.content_length() {
        if length != manifest.artifact.bytes {
            return Err(format!(
                "update size mismatch before download: expected {}, server reported {length}",
                manifest.artifact.bytes
            ));
        }
    }

    let mut file = tokio::fs::File::create(&partial_path)
        .await
        .map_err(|e| format!("could not create the update file: {e}"))?;
    let mut stream = response.bytes_stream();
    let mut hasher = Sha256::new();
    let mut downloaded = 0u64;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("update download interrupted: {e}"))?;
        downloaded = downloaded
            .checked_add(chunk.len() as u64)
            .ok_or_else(|| "update size overflow".to_string())?;
        if downloaded > manifest.artifact.bytes || downloaded > MAX_UPDATE_BYTES {
            remove_quietly(&partial_path).await;
            return Err("update exceeded its signed size".into());
        }
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("could not write the update: {e}"))?;
        hasher.update(&chunk);
        emit_progress(
            app,
            "downloading",
            downloaded,
            manifest.artifact.bytes,
            "Downloading the signed update from saient.co.uk…",
        );
    }

    file.flush()
        .await
        .map_err(|e| format!("could not flush the update: {e}"))?;
    file.sync_all()
        .await
        .map_err(|e| format!("could not sync the update: {e}"))?;
    drop(file);

    if downloaded != manifest.artifact.bytes {
        remove_quietly(&partial_path).await;
        return Err(format!(
            "update size mismatch: expected {}, downloaded {downloaded}",
            manifest.artifact.bytes
        ));
    }
    let actual_sha = format!("{:x}", hasher.finalize());
    if !actual_sha.eq_ignore_ascii_case(manifest.artifact.sha256.trim()) {
        remove_quietly(&partial_path).await;
        return Err("update SHA-256 verification failed".into());
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        tokio::fs::set_permissions(&partial_path, std::fs::Permissions::from_mode(0o600))
            .await
            .map_err(|e| format!("could not secure the update file: {e}"))?;
    }

    remove_quietly(&final_path).await;
    tokio::fs::rename(&partial_path, &final_path)
        .await
        .map_err(|e| format!("could not finalize the update: {e}"))?;
    emit_progress(
        app,
        "verifying",
        downloaded,
        manifest.artifact.bytes,
        "Signature and SHA-256 verified. Checking the Debian package…",
    );
    Ok(final_path)
}

async fn deb_field(package: &Path, field: &str) -> Result<String, String> {
    let output = Command::new("/usr/bin/dpkg-deb")
        .arg("--field")
        .arg(package)
        .arg(field)
        .output()
        .await
        .map_err(|e| format!("could not inspect the Debian package: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "Debian package inspection failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

async fn validate_deb(package: &Path, expected_version: &str) -> Result<(), String> {
    let name = deb_field(package, "Package").await?;
    let version = deb_field(package, "Version").await?;
    let architecture = deb_field(package, "Architecture").await?;
    if name != "saient" {
        return Err(format!("refusing update package named {name:?}"));
    }
    if version != expected_version {
        return Err(format!(
            "update package version mismatch: expected {expected_version}, found {version}"
        ));
    }
    if architecture != "amd64" {
        return Err(format!("refusing update architecture {architecture:?}"));
    }
    Ok(())
}

#[tauri::command]
pub async fn check_update(app: AppHandle) -> Result<UpdateInfo, String> {
    crate::internet::require_update_enabled("Update checks")?;
    let current = app.package_info().version.to_string();

    #[derive(Deserialize)]
    struct Remote {
        version: String,
        #[serde(default)]
        url: Option<String>,
        #[serde(default)]
        notes: Option<String>,
    }

    let client = update_client()?;
    let response = client
        .get(VERSION_URL)
        .header("User-Agent", "Saient")
        .send()
        .await
        .map_err(|e| format!("update check failed: {e}"))?;
    if !response.status().is_success() {
        return Err(format!("update check: HTTP {}", response.status()));
    }
    let remote: Remote = response
        .json()
        .await
        .map_err(|e| format!("bad version.json: {e}"))?;

    let latest = remote.version.trim().to_string();
    Ok(UpdateInfo {
        update_available: is_newer(&latest, &current),
        install_supported: cfg!(all(
            any(target_os = "linux", target_os = "windows"),
            target_arch = "x86_64"
        )),
        current,
        url: remote.url.unwrap_or_else(|| DOWNLOAD_URL.to_string()),
        notes: remote.notes.unwrap_or_default(),
        latest,
    })
}

#[tauri::command]
pub async fn install_update(
    app: AppHandle,
    expected_version: String,
) -> Result<InstallUpdateResult, String> {
    crate::internet::require_update_enabled("Installing updates")?;

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        use tauri_plugin_updater::UpdaterExt;

        if !valid_version(&expected_version) {
            return Err("the requested Windows update version is invalid".into());
        }
        emit_progress(
            &app,
            "checking",
            0,
            0,
            "Checking the signed Windows release…",
        );
        let update = app
            .updater()
            .map_err(|e| format!("could not initialize the Windows updater: {e}"))?
            .check()
            .await
            .map_err(|e| format!("Windows update check failed: {e}"))?
            .ok_or_else(|| "the Windows updater did not find a newer release".to_string())?;

        if update.version.trim() != expected_version.trim() {
            return Err(format!(
                "the latest Windows release changed from {} to {}; check again before installing",
                expected_version.trim(),
                update.version.trim()
            ));
        }

        let progress_app = app.clone();
        let finished_app = app.clone();
        let mut downloaded = 0u64;
        update
            .download_and_install(
                move |chunk_length, content_length| {
                    downloaded = downloaded.saturating_add(chunk_length as u64);
                    emit_progress(
                        &progress_app,
                        "downloading",
                        downloaded,
                        content_length.unwrap_or(0),
                        "Downloading the signed Windows update from saient.co.uk…",
                    );
                },
                move || {
                    emit_progress(
                        &finished_app,
                        "installing",
                        0,
                        0,
                        "Download verified. Windows is installing the update…",
                    );
                },
            )
            .await
            .map_err(|e| format!("Windows update installation failed: {e}"))?;

        return Ok(InstallUpdateResult {
            version: expected_version.trim().to_string(),
            restart_required: true,
        });
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        let current = app.package_info().version.to_string();
        let client = update_client()?;
        emit_progress(
            &app,
            "checking",
            0,
            0,
            "Checking the signed release manifest…",
        );
        let manifest = fetch_manifest(&client).await?;
        if manifest.release.trim() != expected_version.trim() {
            return Err(format!(
                "the latest release changed from {} to {}; check again before installing",
                expected_version.trim(),
                manifest.release.trim()
            ));
        }
        validate_manifest(&manifest, &current)?;
        let package = download_and_verify(&app, &client, &manifest).await?;
        validate_deb(&package, manifest.release.trim()).await?;

        if !Path::new("/usr/bin/apt-get").is_file() {
            return Err(format!(
                "automatic installation needs apt-get; the verified package is at {}",
                package.display()
            ));
        }

        let passwordless_sudo = Path::new("/usr/bin/sudo").is_file()
            && Command::new("/usr/bin/sudo")
                .arg("-n")
                .arg("/usr/bin/true")
                .status()
                .await
                .map(|status| status.success())
                .unwrap_or(false);
        if !passwordless_sudo && !Path::new("/usr/bin/pkexec").is_file() {
            return Err(format!(
                "automatic installation needs sudo or pkexec; the verified package is at {}",
                package.display()
            ));
        }

        emit_progress(
            &app,
            "installing",
            manifest.artifact.bytes,
            manifest.artifact.bytes,
            if passwordless_sudo {
                "Installing with this machine's existing administrator permission…"
            } else {
                "Waiting for Linux administrator approval…"
            },
        );
        let mut installer = if passwordless_sudo {
            let mut command = Command::new("/usr/bin/sudo");
            command.arg("-n").arg("/usr/bin/apt-get");
            command
        } else {
            let mut command = Command::new("/usr/bin/pkexec");
            command.arg("/usr/bin/apt-get");
            command
        };
        let status = installer
            .arg("install")
            .arg("-y")
            .arg("--no-install-recommends")
            .arg(&package)
            .status()
            .await
            .map_err(|e| format!("could not start the system package installer: {e}"))?;
        if !status.success() {
            return Err(match status.code() {
                Some(126) | Some(127) if !passwordless_sudo => {
                    "administrator approval was cancelled".to_string()
                }
                Some(code) => format!("system package installation failed with exit code {code}"),
                None => "system package installation was interrupted".to_string(),
            });
        }

        remove_quietly(&package).await;
        emit_progress(
            &app,
            "installed",
            manifest.artifact.bytes,
            manifest.artifact.bytes,
            "Update installed. Restarting Saient…",
        );
        Ok(InstallUpdateResult {
            version: manifest.release.trim().to_string(),
            restart_required: true,
        })
    }

    #[cfg(not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "x86_64")
    )))]
    {
        let _ = app;
        let _ = expected_version;
        Err("automatic installation is available on Debian/Ubuntu and Windows x86_64".into())
    }
}

#[tauri::command]
pub fn relaunch_after_update(app: AppHandle) {
    app.restart();
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn test_manifest() -> ReleaseManifest {
        ReleaseManifest {
            release: "1.0.5".into(),
            artifact: ReleaseArtifact {
                platform: "Debian/Ubuntu amd64".into(),
                url: "https://saient.co.uk/downloads/Saient_1.0.5_amd64.deb".into(),
                bytes: 1234,
                sha256: "01".repeat(32),
                signature: String::new(),
            },
        }
    }

    #[test]
    fn version_compare() {
        assert!(is_newer("1.0.1", "1.0.0"));
        assert!(is_newer("1.1.0", "1.0.9"));
        assert!(is_newer("2.0", "1.9.9"));
        assert!(!is_newer("1.0.0", "1.0.0"));
        assert!(!is_newer("1.0.0", "1.0.1"));
        assert!(is_newer("v1.2.0", "1.1.0"));
        assert!(valid_version("1.2.3"));
        assert!(!valid_version("1.2.latest"));
    }

    #[test]
    fn signed_manifest_verifies_and_tampering_fails() {
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let public_key_b64 = BASE64.encode(signing_key.verifying_key().as_bytes());
        let mut manifest = test_manifest();
        manifest.artifact.signature = BASE64.encode(
            signing_key
                .sign(signed_message(&manifest).as_bytes())
                .to_bytes(),
        );
        assert!(verify_manifest_signature_with_key(&manifest, &public_key_b64).is_ok());

        manifest.artifact.bytes += 1;
        assert!(verify_manifest_signature_with_key(&manifest, &public_key_b64).is_err());
    }

    #[test]
    fn manifest_rejects_external_or_unsigned_downloads() {
        let mut manifest = test_manifest();
        assert!(validate_manifest(&manifest, "1.0.4").is_err());
        manifest.artifact.url = "https://github.com/SaientAI/saient-releases/update.deb".into();
        assert!(validate_manifest(&manifest, "1.0.4").is_err());
    }
}
