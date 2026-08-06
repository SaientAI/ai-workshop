//! Disposable, isolated workspaces for agent work.
//!
//! The agent does not work on the host. Every task gets its own workspace
//! directory, all commands run inside a sandbox whose only writable path is that
//! directory, and the host is touched exactly once — when the user explicitly
//! exports the result.
//!
//! This inverts the previous model. Before, the host was the workspace and safety
//! depended on enumerating dangerous commands, which fails the first time
//! something unlisted appears. Here isolation is the default state and there is a
//! single, auditable transition out of it.
//!
//! Module layout:
//!   * [`backend`] — the `SandboxBackend` interface every technology implements
//!   * [`bwrap`]   — Bubblewrap, the default Linux backend

pub mod backend;
pub mod bwrap;

pub use backend::{
    detect_backend, HostLayout, NetworkPolicy, SandboxBackend, SandboxError, SandboxSpec,
    WORKSPACE_GUEST_PATH,
};
