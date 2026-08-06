//! Bubblewrap backend — the default sandbox on Linux.
//!
//! Chosen over a container runtime because it needs no daemon, no root, no image
//! pull and no layer cache, which matches a workspace that is created per task
//! and thrown away. It is unprivileged user-namespace isolation, so it is a
//! strong boundary against an ordinary hostile dependency, but it is not a VM: a
//! kernel-level exploit is out of scope for it. That limit is worth stating
//! plainly rather than implying more than it delivers.

use super::backend::{
    HostLayout, NetworkPolicy, SandboxBackend, SandboxError, SandboxSpec, WORKSPACE_GUEST_PATH,
};

/// PATH inside the sandbox. Fixed rather than inherited — the host's PATH can
/// point at user-writable directories that do not exist in here.
const GUEST_PATH: &str = "/usr/local/bin:/usr/bin:/bin";

/// Read-only host paths a normal toolchain needs. Anything not listed is simply
/// absent inside the sandbox, which is how ~/.ssh, ~/.config and ~/.saient-keys
/// stay unreachable — they are never mounted, rather than blocked after the fact.
const RO_PATHS: &[&str] = &["/usr", "/etc/alternatives"];

/// Additionally mounted only when the workspace has been granted network access:
/// DNS configuration and the CA bundle, without which TLS cannot work.
const RO_PATHS_NETWORK: &[&str] = &["/etc/resolv.conf", "/etc/ssl", "/etc/ca-certificates"];

pub struct Bubblewrap {
    binary: String,
}

impl Bubblewrap {
    pub fn new() -> Self {
        Self {
            binary: "bwrap".into(),
        }
    }
}

impl Default for Bubblewrap {
    fn default() -> Self {
        Self::new()
    }
}

impl SandboxBackend for Bubblewrap {
    fn name(&self) -> &'static str {
        "bubblewrap"
    }

    fn is_available(&self) -> bool {
        std::process::Command::new(&self.binary)
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn build_command(&self, spec: &SandboxSpec) -> Result<Vec<String>, SandboxError> {
        let ws = spec
            .workspace_dir
            .to_str()
            .ok_or_else(|| SandboxError::BadWorkspace("path is not valid UTF-8".into()))?;
        if !spec.workspace_dir.is_absolute() {
            return Err(SandboxError::BadWorkspace(format!(
                "must be absolute, got {ws}"
            )));
        }

        let mut cmd: Vec<String> = vec![self.binary.clone()];
        let mut push = |a: &str| cmd.push(a.to_string());

        // Isolate every namespace, then hand back only what was explicitly granted.
        // Starting from "share nothing" means a new namespace type appearing in a
        // future kernel is denied by default rather than silently shared.
        push("--unshare-all");
        if spec.network == NetworkPolicy::Allowed {
            push("--share-net");
        }

        // Tear the sandbox down with the app, so a crash cannot leave a live
        // process holding the workspace open.
        push("--die-with-parent");
        // Detach the controlling terminal: without this a process inside can push
        // characters onto the parent's tty with TIOCSTI and run commands outside.
        push("--new-session");
        push("--cap-drop");
        push("ALL");

        // Read-only system. Note these are *host* paths appearing at the same guest
        // path — none of them are writable, and the user's home is not among them.
        for path in RO_PATHS {
            if std::path::Path::new(path).exists() {
                push("--ro-bind");
                push(path);
                push(path);
            }
        }
        if spec.network == NetworkPolicy::Allowed {
            for path in RO_PATHS_NETWORK {
                if std::path::Path::new(path).exists() {
                    push("--ro-bind");
                    push(path);
                    push(path);
                }
            }
        }

        // On a merged-/usr host, /bin and friends are symlinks; recreate them as
        // symlinks rather than binding, which would collide with the /usr bind.
        if spec.layout.merged_usr {
            for (target, link) in [
                ("usr/bin", "/bin"),
                ("usr/lib", "/lib"),
                ("usr/lib64", "/lib64"),
                ("usr/sbin", "/sbin"),
            ] {
                push("--symlink");
                push(target);
                push(link);
            }
        } else {
            for path in ["/bin", "/lib", "/lib64", "/sbin"] {
                if std::path::Path::new(path).exists() {
                    push("--ro-bind");
                    push(path);
                    push(path);
                }
            }
        }

        push("--proc");
        push("/proc");
        push("--dev");
        push("/dev");
        // Private /tmp so scratch files cannot be read by, or leak to, the host.
        push("--tmpfs");
        push("/tmp");

        // The one writable host path.
        push("--bind");
        push(ws);
        push(WORKSPACE_GUEST_PATH);
        push("--chdir");
        push(WORKSPACE_GUEST_PATH);

        // Start from an empty environment: the host's is full of tokens, proxy
        // settings and paths that would both leak out and confuse tools in here.
        push("--clearenv");
        push("--setenv");
        push("HOME");
        push(WORKSPACE_GUEST_PATH);
        push("--setenv");
        push("PATH");
        push(GUEST_PATH);
        push("--setenv");
        push("TMPDIR");
        push("/tmp");
        for (key, value) in &spec.env {
            // HOME and PATH are the boundary, not a preference — a caller must not
            // be able to point HOME back at the real one.
            if key == "HOME" || key == "PATH" {
                continue;
            }
            push("--setenv");
            push(key);
            push(value);
        }

        push("--");
        cmd.push(spec.command.clone());
        cmd.extend(spec.args.iter().cloned());
        Ok(cmd)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::backend::SandboxSpec;

    fn spec() -> SandboxSpec {
        SandboxSpec::new("/tmp/ws-test", "sh")
            .args(["-c", "echo hi"])
            .layout(HostLayout { merged_usr: true })
    }

    fn argv(s: &SandboxSpec) -> Vec<String> {
        Bubblewrap::new().build_command(s).expect("builds")
    }

    /// Walk the argv looking for `flag` followed by `value`.
    fn has_pair(v: &[String], flag: &str, value: &str) -> bool {
        v.windows(2).any(|w| w[0] == flag && w[1] == value)
    }

    #[test]
    fn denies_network_by_default() {
        let v = argv(&spec());
        assert!(v.contains(&"--unshare-all".to_string()));
        assert!(
            !v.contains(&"--share-net".to_string()),
            "network must be off unless explicitly granted"
        );
    }

    #[test]
    fn shares_net_only_when_granted() {
        let v = argv(&spec().network(NetworkPolicy::Allowed));
        assert!(v.contains(&"--share-net".to_string()));
        // …and only then are DNS and the CA bundle exposed.
        assert!(has_pair(&v, "--ro-bind", "/etc/resolv.conf") || !std::path::Path::new("/etc/resolv.conf").exists());
    }

    #[test]
    fn resolv_conf_is_absent_without_network() {
        let v = argv(&spec());
        assert!(!has_pair(&v, "--ro-bind", "/etc/resolv.conf"));
        assert!(!has_pair(&v, "--ro-bind", "/etc/ssl"));
    }

    #[test]
    fn home_points_at_the_workspace() {
        let v = argv(&spec());
        let i = v.iter().position(|a| a == "HOME").expect("HOME is set");
        assert_eq!(v[i + 1], WORKSPACE_GUEST_PATH);
    }

    #[test]
    fn caller_cannot_repoint_home_or_path() {
        let v = argv(&spec().env("HOME", "/home/tiny").env("PATH", "/evil"));
        assert!(!v.contains(&"/home/tiny".to_string()), "HOME override leaked");
        assert!(!v.contains(&"/evil".to_string()), "PATH override leaked");
        let i = v.iter().position(|a| a == "HOME").unwrap();
        assert_eq!(v[i + 1], WORKSPACE_GUEST_PATH);
    }

    #[test]
    fn environment_starts_empty() {
        assert!(argv(&spec()).contains(&"--clearenv".to_string()));
    }

    #[test]
    fn only_the_workspace_is_writable() {
        let v = argv(&spec());
        // Exactly one --bind (read-write); everything else is --ro-bind.
        let writable: Vec<&String> = v
            .windows(2)
            .filter(|w| w[0] == "--bind")
            .map(|w| &w[1])
            .collect();
        assert_eq!(writable, vec![&"/tmp/ws-test".to_string()]);
    }

    #[test]
    fn host_secret_directories_are_never_mounted() {
        let v = argv(&spec().network(NetworkPolicy::Allowed)).join(" ");
        for secret in [
            "/home/tiny/.ssh",
            "/home/tiny/.config",
            "/home/tiny/.saient-keys",
            "/home/tiny",
        ] {
            assert!(!v.contains(secret), "{secret} must not appear in the sandbox");
        }
    }

    #[test]
    fn hardening_flags_are_present() {
        let v = argv(&spec());
        for flag in ["--die-with-parent", "--new-session", "--cap-drop"] {
            assert!(v.contains(&flag.to_string()), "missing {flag}");
        }
        assert!(has_pair(&v, "--tmpfs", "/tmp"), "/tmp must be private");
    }

    #[test]
    fn merged_usr_uses_symlinks_not_binds() {
        let v = argv(&spec().layout(HostLayout { merged_usr: true }));
        assert!(has_pair(&v, "--symlink", "usr/bin"));
        assert!(!has_pair(&v, "--ro-bind", "/bin"));
    }

    #[test]
    fn split_usr_binds_the_real_directories() {
        let v = argv(&spec().layout(HostLayout { merged_usr: false }));
        assert!(!has_pair(&v, "--symlink", "usr/bin"));
        // /bin exists on any Linux host, merged or not.
        assert!(has_pair(&v, "--ro-bind", "/bin"));
    }

    #[test]
    fn the_command_lands_after_the_separator() {
        let v = argv(&spec());
        let sep = v.iter().position(|a| a == "--").expect("has separator");
        assert_eq!(v[sep + 1..], ["sh", "-c", "echo hi"]);
    }

    #[test]
    fn relative_workspace_is_refused() {
        let s = SandboxSpec::new("relative/path", "ls");
        assert!(Bubblewrap::new().build_command(&s).is_err());
    }

    // ── Live isolation tests ─────────────────────────────────────────────────
    //
    // The tests above only prove the argv is assembled as intended. These
    // actually spawn bwrap and check the boundary holds, which is the claim that
    // matters. Marked #[ignore] because they need bubblewrap installed:
    //   cargo test --bins live_ -- --ignored --nocapture

    fn run(spec: &SandboxSpec) -> (bool, String, String) {
        let argv = argv(spec);
        let out = std::process::Command::new(&argv[0])
            .args(&argv[1..])
            .output()
            .expect("bwrap spawns");
        (
            out.status.success(),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    }

    fn live_spec(dir: &std::path::Path, cmd: &str) -> SandboxSpec {
        SandboxSpec::new(dir, "sh").args(["-c", cmd])
    }

    #[test]
    #[ignore]
    fn live_home_is_the_workspace_not_the_users() {
        let tmp = tempfile::tempdir().unwrap();
        let (ok, stdout, _) = run(&live_spec(tmp.path(), "echo $HOME; ls -a $HOME"));
        assert!(ok);
        assert!(stdout.contains(WORKSPACE_GUEST_PATH), "got: {stdout}");
        assert!(!stdout.contains(".saient-keys"), "host home leaked: {stdout}");
    }

    #[test]
    #[ignore]
    fn live_host_secrets_are_unreachable() {
        let tmp = tempfile::tempdir().unwrap();
        let real_home = std::env::var("HOME").unwrap_or_else(|_| "/root".into());
        for target in [".ssh", ".saient-keys", ".config"] {
            let probe = format!("cat {real_home}/{target}/* 2>&1 | head -1");
            let (_, stdout, _) = run(&live_spec(tmp.path(), &probe));
            assert!(
                stdout.contains("No such file") || stdout.trim().is_empty(),
                "{target} was readable from inside the sandbox: {stdout}"
            );
        }
    }

    /// Deliberately two-sided.
    ///
    /// The first version of this only checked the denied case, using a probe run
    /// under `sh`. `/dev/tcp` is a bash builtin, so dash failed it identically
    /// whether or not the network was reachable — the test passed while proving
    /// nothing. Asserting the allowed case too means the probe has to actually
    /// work before the denial result counts for anything.
    ///
    /// Raw TCP to a public resolver rather than a hostname, so a failure is the
    /// network namespace and not missing DNS config.
    #[test]
    #[ignore]
    fn live_network_denied_by_default_and_reachable_when_granted() {
        let tmp = tempfile::tempdir().unwrap();
        let probe = r#"timeout 5 bash -c "exec 3<>/dev/tcp/1.1.1.1/443" 2>&1; echo rc=$?"#;

        let spec = SandboxSpec::new(tmp.path(), "bash").args(["-c", probe]);

        let (_, denied, _) = run(&spec);
        assert!(
            !denied.contains("rc=0"),
            "network was reachable while denied: {denied}"
        );

        let (_, allowed, _) = run(&spec.clone().network(NetworkPolicy::Allowed));
        assert!(
            allowed.contains("rc=0"),
            "probe cannot reach the network even when granted, so the denial \
             result above proves nothing: {allowed}"
        );
    }

    #[test]
    #[ignore]
    fn live_workspace_writes_land_on_the_host() {
        let tmp = tempfile::tempdir().unwrap();
        let (ok, _, err) = run(&live_spec(tmp.path(), "echo written > proof.txt"));
        assert!(ok, "stderr: {err}");
        let landed = std::fs::read_to_string(tmp.path().join("proof.txt")).expect("file on host");
        assert_eq!(landed.trim(), "written");
    }

    #[test]
    #[ignore]
    fn live_host_filesystem_is_read_only() {
        let tmp = tempfile::tempdir().unwrap();
        let (_, stdout, _) = run(&live_spec(
            tmp.path(),
            "touch /usr/pwned 2>&1; echo rc=$?",
        ));
        assert!(!stdout.contains("rc=0"), "/usr was writable: {stdout}");
    }
}
