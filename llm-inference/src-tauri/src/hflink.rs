//! `saient://` deep links — the Hugging Face "Use this model → Saient" route.
//!
//! A deep link is UNTRUSTED, remotely-triggerable input: any web page can hand
//! the app one, and the user may not have meant to click it. It therefore has
//! exactly two powers here — parse a repo id, and ask the UI to offer an import.
//!
//! It grants no network authority of its own. There is deliberately no
//! `set_deeplink_internet_authorized` sibling to the setup/update grants in
//! `internet.rs`: the import runs through the ordinary Hugging Face commands
//! and hits their existing Internet gate, so with Internet off a link does
//! nothing but show the usual "turn it on in Settings" message.

use std::sync::Mutex;
use tauri::{Emitter, Manager};

/// The one link shape we answer to.
const MODEL_PREFIX: &str = "models/huggingface/";
const SCHEME: &str = "saient://";

/// A link that arrived before the webview was listening. In memory only, so it
/// can never survive into a later launch as a stale prompt.
static PENDING: Mutex<Option<String>> = Mutex::new(None);

/// Is this a repo id we are willing to put into a Hugging Face URL?
///
/// Exactly `owner/name`, both segments non-empty and made only of the
/// characters HF ids actually use. This mirrors `HfBrowser.svelte`'s `repoRe`
/// for typed input, but it is enforced here because a deep link is attacker
/// reachable in a way a text box is not: no traversal, no query or fragment
/// smuggling, no percent-escapes.
fn valid_repo(repo: &str) -> bool {
    let mut parts = repo.split('/');
    let (Some(owner), Some(name), None) = (parts.next(), parts.next(), parts.next()) else {
        return false;
    };
    [owner, name].iter().all(|seg| {
        !seg.is_empty()
            && seg.len() <= 96
            && *seg != "."
            && *seg != ".."
            && seg
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_'))
    })
}

/// Extract the repo id from `saient://models/huggingface/<owner>/<name>`.
/// Returns `None` for anything else — including other `saient://` paths, which
/// are not an error, just not ours.
pub fn parse_model_link(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if !raw.to_ascii_lowercase().starts_with(SCHEME) {
        return None;
    }
    let rest = &raw[SCHEME.len()..]; // the scheme is ASCII, so this index is safe
    let rest = rest.split(['?', '#']).next()?;
    let repo = rest.strip_prefix(MODEL_PREFIX)?.trim_end_matches('/');
    valid_repo(repo).then(|| repo.to_string())
}

/// Handle a batch of candidate URLs (an `on_open_url` event, or a process's
/// argv). Non-links are ignored silently.
pub fn deliver(app: &tauri::AppHandle, urls: impl IntoIterator<Item = String>) {
    let Some(repo) = urls.into_iter().find_map(|u| parse_model_link(&u)) else {
        return;
    };
    // Stored first, then announced: on a cold start the window exists before any
    // JS listener does, so the event can be dropped. The frontend consumes the
    // pending value on mount, which closes that gap.
    if let Ok(mut slot) = PENDING.lock() {
        *slot = Some(repo.clone());
    }
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.set_focus();
        let _ = window.emit("hf-deeplink", serde_json::json!({ "repo": repo }));
    }
}

/// Take the repo id from a deep link that has not been shown yet, if any.
/// Taking clears it, so one link opens one prompt.
#[tauri::command]
pub fn hf_pending_deeplink() -> Option<String> {
    PENDING.lock().ok().and_then(|mut slot| slot.take())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_the_hugging_face_model_link() {
        assert_eq!(
            parse_model_link("saient://models/huggingface/Qwen/Qwen3-8B-GGUF").as_deref(),
            Some("Qwen/Qwen3-8B-GGUF")
        );
        // Scheme casing is not under our control; a trailing slash is harmless.
        assert_eq!(
            parse_model_link("SAIENT://models/huggingface/bartowski/Llama-3.2-3B-Instruct-GGUF/")
                .as_deref(),
            Some("bartowski/Llama-3.2-3B-Instruct-GGUF")
        );
    }

    #[test]
    fn rejects_anything_that_is_not_a_plain_repo_id() {
        for bad in [
            "saient://models/huggingface/../../etc/passwd",
            "saient://models/huggingface/owner",
            "saient://models/huggingface/owner/name/extra",
            "saient://models/huggingface/owner/",
            "saient://models/huggingface/",
            "saient://models/huggingface/ow ner/name",
            "saient://models/huggingface/owner/na%2fme",
            "saient://run/rm-rf",
            "https://huggingface.co/owner/name",
            "",
        ] {
            assert_eq!(parse_model_link(bad), None, "should have rejected {bad:?}");
        }
    }

    #[test]
    fn a_query_or_fragment_does_not_reach_the_repo_id() {
        assert_eq!(
            parse_model_link("saient://models/huggingface/owner/name?token=abc#frag").as_deref(),
            Some("owner/name")
        );
    }

    #[test]
    fn pending_is_taken_once() {
        *PENDING.lock().unwrap() = Some("owner/name".into());
        assert_eq!(hf_pending_deeplink().as_deref(), Some("owner/name"));
        assert_eq!(hf_pending_deeplink(), None);
    }
}
