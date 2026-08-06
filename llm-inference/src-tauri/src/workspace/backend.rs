//! The sandbox backend interface.
//!
//! Agent logic never talks to a sandbox technology directly — it builds a
//! [`SandboxSpec`] and hands it to whichever [`SandboxBackend`] is active. Adding
//! a backend later (a VM, a remote runner, a Windows equivalent) means
//! implementing this trait and registering it in [`detect_backend`]; nothing in
//! the agent changes.
//!
//! [`SandboxBackend::build_command`] is deliberately pure — it returns the argv
//! rather than running it. That is what lets the tests below assert the exact
//! isolation flags, which is the only way to keep a security boundary honest as
//! the code moves.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Where the workspace is mounted inside every sandbox, whatever the backend.
pub const WORKSPACE_GUEST_PATH: &str = "/workspace";

/// Whether a workspace may reach the network.
///
/// An enum rather than a `bool` on purpose: this is the difference between "a
/// hostile dependency can phone home" and "it cannot", and a bare `true` at a
/// call site is far too easy to misread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NetworkPolicy {
    /// No network namespace access at all. The default for every new workspace.
    Denied,
    /// Egress permitted. Only ever set by an explicit per-workspace grant.
    Allowed,
}

impl Default for NetworkPolicy {
    fn default() -> Self {
        Self::Denied
    }
}

/// Facts about the host filesystem that change how a sandbox must be assembled.
///
/// Passed in rather than probed inside `build_command` so the builder stays pure
/// and both layouts can be tested on any machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostLayout {
    /// True when /bin, /lib, /sbin are symlinks into /usr (Debian, Arch, Fedora).
    /// Such a host must recreate them as symlinks; bind-mounting them over the
    /// top of a /usr bind fails.
    pub merged_usr: bool,
}

impl HostLayout {
    /// Probe the running host.
    pub fn detect() -> Self {
        Self {
            merged_usr: Path::new("/bin").is_symlink(),
        }
    }
}

/// One command to run inside one workspace.
#[derive(Debug, Clone)]
pub struct SandboxSpec {
    /// Host directory that becomes [`WORKSPACE_GUEST_PATH`]. This is the *only*
    /// host path a sandbox may write to, and the only one carrying user content.
    pub workspace_dir: PathBuf,
    pub command: String,
    pub args: Vec<String>,
    pub network: NetworkPolicy,
    /// Extra environment. HOME and PATH are set by the backend and cannot be
    /// overridden here — see `SandboxBackend::build_command`.
    /// A BTreeMap so the generated argv is deterministic and therefore testable.
    pub env: BTreeMap<String, String>,
    pub layout: HostLayout,
}

impl SandboxSpec {
    pub fn new(workspace_dir: impl Into<PathBuf>, command: impl Into<String>) -> Self {
        Self {
            workspace_dir: workspace_dir.into(),
            command: command.into(),
            args: Vec::new(),
            network: NetworkPolicy::Denied,
            env: BTreeMap::new(),
            layout: HostLayout::detect(),
        }
    }

    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.args = args.into_iter().map(Into::into).collect();
        self
    }

    pub fn network(mut self, policy: NetworkPolicy) -> Self {
        self.network = policy;
        self
    }

    pub fn env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env.insert(key.into(), value.into());
        self
    }

    pub fn layout(mut self, layout: HostLayout) -> Self {
        self.layout = layout;
        self
    }
}

/// Written out by hand rather than pulling in `thiserror` for one enum — the
/// crate does not already depend on it.
#[derive(Debug)]
pub enum SandboxError {
    NoBackend(String),
    BadWorkspace(String),
    Spawn {
        backend: &'static str,
        source: std::io::Error,
    },
}

impl std::fmt::Display for SandboxError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoBackend(detail) => write!(
                f,
                "no sandbox backend is available on this system ({detail}). \
                 Install bubblewrap (`apt install bubblewrap`) to run agent tasks in isolation."
            ),
            Self::BadWorkspace(detail) => write!(f, "sandbox workspace path is not usable: {detail}"),
            Self::Spawn { backend, source } => {
                write!(f, "sandbox backend '{backend}' failed to start: {source}")
            }
        }
    }
}

impl std::error::Error for SandboxError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spawn { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// A technology that can run a command in isolation.
pub trait SandboxBackend: Send + Sync {
    /// Stable identifier, surfaced in the UI and in audit records.
    fn name(&self) -> &'static str;

    /// Whether this backend can actually run here, right now.
    fn is_available(&self) -> bool;

    /// Build the argv that runs `spec` in isolation.
    ///
    /// Pure and side-effect free so the isolation flags can be asserted directly.
    /// Implementations must guarantee, for every spec they accept:
    ///   * no host path is writable except `spec.workspace_dir`;
    ///   * HOME points at [`WORKSPACE_GUEST_PATH`], never the real home;
    ///   * the environment starts empty rather than inheriting the host's;
    ///   * [`NetworkPolicy::Denied`] means no network namespace access.
    fn build_command(&self, spec: &SandboxSpec) -> Result<Vec<String>, SandboxError>;
}

/// Pick the backend to use, most preferred first.
///
/// Bubblewrap is the default on Linux: it is a lightweight setuid helper with no
/// daemon, no image pulls and no root requirement, which suits a disposable
/// per-task workspace far better than a container runtime does. Heavier backends
/// can be registered ahead of or behind it later without touching agent code.
pub fn detect_backend() -> Result<Box<dyn SandboxBackend>, SandboxError> {
    let candidates: Vec<Box<dyn SandboxBackend>> = vec![Box::new(super::bwrap::Bubblewrap::new())];

    let mut tried = Vec::new();
    for backend in candidates {
        if backend.is_available() {
            return Ok(backend);
        }
        tried.push(backend.name());
    }
    Err(SandboxError::NoBackend(format!(
        "tried: {}",
        tried.join(", ")
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_defaults_to_denied() {
        // The single most important default in this module.
        assert_eq!(NetworkPolicy::default(), NetworkPolicy::Denied);
        let spec = SandboxSpec::new("/tmp/ws", "ls");
        assert_eq!(spec.network, NetworkPolicy::Denied);
    }

    #[test]
    fn builder_round_trips() {
        let spec = SandboxSpec::new("/tmp/ws", "echo")
            .args(["hello", "world"])
            .network(NetworkPolicy::Allowed)
            .env("FOO", "bar");
        assert_eq!(spec.command, "echo");
        assert_eq!(spec.args, vec!["hello", "world"]);
        assert_eq!(spec.network, NetworkPolicy::Allowed);
        assert_eq!(spec.env.get("FOO").map(String::as_str), Some("bar"));
    }
}
