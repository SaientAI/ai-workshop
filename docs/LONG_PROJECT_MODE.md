# Long Project mode

Development feature: **1.0.25-dev.1**. This work does not update the matching public **1.0.24** installers or update feeds. The development implementation is not a claim of verified multi-day operation or native Windows readiness.

Long Project mode keeps the Saient **terminal agent** pursuing one saved goal across many model calls. It is opt-in, uses the existing local model and terminal execution/conscience path, and persists its state between sessions. It does **not** change Planner Auto, Planner Write mode, or ordinary terminal-turn limits.

It is not one endlessly running model call. The controller repeatedly requests an action, checks permissions, records the intention, executes it, and saves the evidence before asking for another action. A model may report a blocker or request acceptance testing; it cannot grant itself permissions or extend its budget.

## Start in the Saient prompt

1. Open **Agent → Terminal**, select the intended workspace, and run `saient`.
2. Load/start the local model and establish its normal Saient binding. Binding problems require explicit `/bind` or `/rebind`; Long Project mode does not bypass them.
3. Expand **Long Project mode** above the terminal to generate a command, or type one of the commands below.
4. Paste `/project …` into Saient's `❯` prompt, **not directly into Bash or PowerShell**.
5. Review the workspace, permissions, acceptance command and allowance. Starting requires a separate `y`/`yes` confirmation.

The helper only copies a command; copying does not start execution. Its permission checkboxes default to off.

For a short, read-only review:

```text
/project start --hours 0.25 --steps 30 -- Review this workspace and identify the next implementation steps
```

For development in an existing project with its test runner already configured:

```text
/project start --hours 24 --steps 10000 --tool-timeout 3600 --allow-writes --verify 'npm test' -- Implement the requested calculator changes and test the result
```

This enables workspace file writes, not arbitrary model-selected shell commands. The exact user-supplied `npm test` acceptance command is separately authorized. If the task needs model-selected build/install/test commands, explicitly add `--allow-shell` only after reviewing the host-access warning below. The named tools, dependencies and tests must exist; the setup wizard does not configure arbitrary project dependencies.

### JSON form, including Windows

JSON avoids the additional quoting rules used by flag parsing. Backslashes inside JSON strings must be doubled. Unicode and spaces are supported. A path in the goal describes the request; **it does not change the selected workspace**.

```text
/project start {"goal":"Review C:\\Users\\me\\Desktop\\Project and identify its failing tests","hours":1,"steps":100,"tool_timeout":600,"allow_write":false,"allow_shell":false}
```

A days-scale example for a Python project, after reviewing its existing tests:

```text
/project start {"goal":"Implement the agreed changes in this workspace and pass its existing tests","hours":72,"steps":20000,"tool_timeout":3600,"allow_write":true,"allow_shell":false,"verify_command":"python -m unittest discover -s tests"}
```

The acceptance command runs through the host's normal shell adapter: PowerShell on Windows, Bash/sh on Linux. Choose a command appropriate to that machine. JSON preserves command text; it does not translate shell syntax.

## Permissions and what completion means

`allow_write` and `allow_shell` both default to `false`. They are independent of Planner Write mode and `/yolo`. Permissions and the acceptance command are saved with the goal; resuming or extending the budget does not broaden them.

- **File tools:** Long Project mode confines them to the saved workspace. It does not adopt an old terminal `@temp` directory as a project root.
- **Host shell:** `--allow-shell` grants unattended command execution with your user account's filesystem, network and service access. This is **not an OS sandbox**. A command can technically reach outside the workspace or write files even when `allow_write` is false. Scope/conscience checks are not an operating-system security boundary.
- **Acceptance command:** `--verify`/`verify_command` is an explicitly authorized host command, even when general shell permission is off. Review it and the scripts it invokes. It can have side effects; “verification” is not a guarantee that a command is read-only.

None of these settings grants permission for unrelated deletion, publication, purchases, credential changes or other work outside your instructions. Keep backups and review generated changes. Untrusted repository contents and generated shell commands still need caution.

When the model requests finishing, the controller runs the configured acceptance command and checks the available task evidence. Only passing that gate can record `COMPLETED`. Without an acceptance command, finishing produces `PAUSED` for your review. Failed or missing evidence does not count as success.

`COMPLETED` means the configured check and recorded requirements passed. It is not proof of arbitrary goals, every possible behavior, security, “bug-free” software, or AGI. A weak test command provides weak assurance; do not weaken tests merely to obtain a pass.

## Budgets and waiting

This mode uses the **local model**, not a purchased-token balance. A step is a reserved model call, not a measured token amount, monetary charge or guarantee of useful work. Reservations are saved before inference; interrupted/failed calls are not silently refunded.

| Setting | Terminal default | Supported range |
|---|---|---|
| `--hours` / `hours` | 24 hours | One second through 366 days |
| `--steps` / `steps` | 10,000 calls | 1–10,000,000 calls |
| `--tool-timeout` / `tool_timeout` | 600 seconds | 1 second through 7 days |
| `--max-no-progress` / `max_no_progress` | 8 consecutive attempts | 1–100,000 attempts |

The UI helper initially selects a **3,600-second** command timeout; its generated JSON includes that explicit value. Short fractional-hour allowances can be supplied directly at the prompt. These limits permit days-scale configuration; they are not evidence of a completed days-long soak test.

The wall-clock deadline starts when the project is created and includes pauses, model outages and time while the app is closed. Each shell command receives at most its configured timeout and the remaining project time. Model-call or wall-time exhaustion records `BUDGET_EXHAUSTED`, not completion.

While the controller remains running, a missing model puts it into `WAITING`. It polls for availability and can continue when the model returns, without resetting its budget or no-progress counter. Repeated failed/rejected/unchanged actions trigger `BLOCKED` for review. This is a bounded repetition check, not a semantic guarantee that every novel action advances the goal.

## Pause, inspect, resume and stop

```text
/project status
/project pause
/project resume
/project stop
```

`/project status` shows saved state and can be read without acquiring the runner's exclusive lock. Only one cooperating runner owns a canonical workspace at a time.

Use `/project pause` or Ctrl-C to interrupt an active run. If a tool may already have changed something, its outcome remains unknown and needs review; pausing does not roll changes back. `/project stop` settles the project as stopped, not completed. A stopped project is not resumed; explicitly archive it before starting a new goal.

Keep the desktop/terminal and model available for unattended continuation. Closing the app or rebooting does **not** automatically resume the project. Reopen the same workspace, run `saient`, inspect `/project status`, and explicitly request `/project resume`. Prior reservations and the original deadline remain in force.

If a detached command supervisor is still running after the terminal closes, its owned command may continue until completion, cancellation or timeout. That is different from automatically restarting the project's model/tool loop.

## Interrupted commands and unknown outcomes

The controller records an action identity **before** execution. A crash between a side effect and its saved result cannot establish whether that action succeeded. Recovery retains the unknown intention and never blindly repeats it.

```text
/project jobs
/project cancel-jobs
/project acknowledge
/project resume
```

Inspect the job's recorded state, output/log path, affected files and any remaining processes first. `cancel-jobs` requests cancellation through each retained supervisor; it does not adopt or signal a saved PID, which might now belong to an unrelated process.

An active supervisor—or uncertain ownership—blocks acknowledgment/resumption until it is settled. A dead supervisor may leave incomplete evidence and possible external effects. After **your actual inspection**, `/project acknowledge` records that you reviewed the unknown outcome; it does not claim success or restart execution. The separate job review is tied to the retained metadata digest. `/project resume --acknowledge-unknown` combines an explicit acknowledgment with resume when recovery permits it.

Do not acknowledge simply to remove a warning. Linux commands that deliberately daemonize or escape their owned process group can outlive ordinary cleanup. Windows uses its process-tree containment/termination path, but native Windows behavior for this development feature still needs verification. Supervisor crashes, detached descendants and uninterruptible operating-system I/O remain important limitations.

## Extend or archive explicitly

To continue the **same goal and history** after reviewing a limit:

```text
/project extend --hours 24 --steps 10000
/project resume
```

Extension is allowed for paused, blocked or budget-exhausted projects with no unresolved action. It adds to the existing allowance, preserves prior call usage and permissions, and does not automatically run. Additional hours extend the **original deadline**, not a fresh timer from now; an overdue project needs enough additional time to cover the elapsed gap. The validated maximums still apply.

To start a **different goal** in the same workspace:

```text
/project stop
/project archive
```

Resolve/review unknown actions first. Archival is allowed only for a settled, non-running/non-waiting project. It saves a SQLite backup before clearing the current project, preserving the lock identity. A subsequent explicit start creates a new goal with its own allowance. Starting again without archival does not overwrite an existing project.

## Saved evidence and storage limits

State lives outside the workspace under:

```text
SAIENT_STATE_DIR/projects/<SHA-256 of canonical workspace>/
  project.sqlite3
  runner.lock
  jobs/<job-id>/
  archives/project-<unique-id>.sqlite3
```

Workspace aliases, including normalized Windows extended-path spellings, use the same project identity. State paths are checked against symlink/junction escapes. This does not protect against a malicious process with your own account's access; the directory is not a security sandbox.

Choose a dedicated project folder. A workspace that contains Saient's own state directory, such as an overly broad home-directory workspace, is rejected rather than placing recovery data inside the area being edited.

The journal retains the **latest 1,000 events**, not an unlimited transcript. Large event payloads are summarized with an explicit truncation marker. Model context uses a bounded recent evidence window, so the agent must re-inspect files when older detail has fallen out of context. Archival preserves the retained database, not events already evicted from the ring.

Each command's captured output log is capped at **8 MiB** and may truncate older output. The controller also checks a **256 MiB retained-command-evidence allowance** before launching another job, reserving 9 MiB for the next command. This is a prelaunch guard, not a global disk quota: project files, archives and other application data are separate.

Logs are not automatically deleted. `/project archive` backs up project state but does **not** prune command logs or reset their storage allowance. If retained evidence prevents another command, preserve and review the logs, settle all jobs, and manage that evidence explicitly before continuing. Do not remove active job metadata or the runner lock to bypass recovery.

## Verification boundary

The development tests exercise deterministic proposals with real local files, SQLite and subprocesses, including persistence, permission failures, recovery and repeated calls. Model/binding fixtures do not demonstrate real-model task quality; consult the matching build's verification record for direct local-model and native Windows CI results. CI checks are not installed Windows GUI/model validation, and short smoke tests are not an actual multi-day uninterrupted run. The public 1.0.24 release remains separate from this development feature.
