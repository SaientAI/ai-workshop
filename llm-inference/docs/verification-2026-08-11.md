# Saient 1.0.12 verification — 2026-08-11

All commands below were run on the Linux development/release workstation. The
Debian package was built locally; no Vercel or cloud deployment was used.

## Source suites

```sh
cd /home/tiny/projects/saient/agi
python3 -m unittest discover -s tests
# Ran 398 tests — OK

cd /home/tiny/projects/saient/desktop/llm-inference
npm test
# 324 passed, 0 failed

node src/modalOwnership.test.js && node src/networkSurface.test.js && node src/setupNetwork.test.js
# modalOwnership.test.js — 5 passed
# networkSurface.test.js — 8 passed
# setupNetwork.test.js — 12 passed

npm run check
# 0 errors, 2 existing warnings (AutonomyConfirm initial-value capture and
# ImageGenScreen non-interactive image click handler)

cd src-tauri
cargo test
# 112 passed, 0 failed, 6 ignored

cargo test canonical_workspace_survives_unusual_spelling_spaces_and_later_actions -- --nocapture
# pty::tests::canonical_workspace_survives_unusual_spelling_spaces_and_later_actions ... ok
```

The last test executes the generated production CLI against a directory named
`Saient begining  odd spelling`. It writes, lists, reads, edits, reads again,
and checks that cwd, `SAIENT_WORKSPACE`, and every safe-path resolution use the
same canonical root.

## Real-model formal binding

The already-running user-selected model was Qwen 2.5 Coder 14B Q4_K_M on a
numeric loopback endpoint.

```sh
cd /home/tiny/projects/saient/desktop/llm-inference
env PYTHONPATH=src-tauri/resources/saient \
  PYTHONDONTWRITEBYTECODE=1 \
  NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
  HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= \
  http_proxy= https_proxy= all_proxy= \
  /usr/bin/time -f 'elapsed=%E exit=%x' \
  python3 src-tauri/resources/saient/binding_bridge.py bind \
  --endpoint http://127.0.0.1:38533 \
  --manifest-dir /home/tiny/projects/saient/desktop/data/config/saient/bindings
```

Result: exit 0 in 7:32.61. The v2 manifest recorded 74 L0 samples with occupancy,
grounding, and expression rates of 1.0 and identity-leak rate 0.0. Both the
`identity_self_model_challenge` and `record_authority_conflict` boundaries
passed. The raw Qwen identity answer and false `stabilize` claim were
intercepted; the recorded `respond`/`explore` actions were preserved.

```sh
python3 tools/test_saient_binding_e2e.py --endpoint http://127.0.0.1:38533
```

Result: PASS. Four consecutive ticks `[1, 2, 3, 4]` all had state context
injected, non-empty responses, and a clean identity boundary. Neutral turns did
not use fallback. The adversarial turns used the deterministic record-grounded
fallback. The persisted final action was `respond`, selected by `rule_policy`.
The dead endpoint returned no output, set `plain_llm_fallback` false, and left
the state at tick 4.

## No-route offline proof

The installed tinyq4 CUDA engine and the same real Qwen model were started
inside a new network namespace after bringing up loopback. Before and after the
test, `ip route show` returned no routes. The E2E command was run under `strace
-ff -e trace=network` inside that namespace.

```sh
offline_root=$(mktemp -d /tmp/saient-offline-trace.XXXXXX)
mkdir -p "$offline_root/state"
sudo env OFFLINE_STATE="$offline_root/state" unshare -n bash -c '
  set -euo pipefail
  ip link set lo up
  ip -brief addr
  ip route show
  env TINYQ4_BIND=127.0.0.1 \
    LD_LIBRARY_PATH=/usr/lib/Saient/resources/engine \
    /usr/lib/Saient/resources/engine/tinyq4-cuda \
    /home/tiny/projects/saient/desktop/data/models/llm/Qwen2.5-Coder-14B-Instruct-Q4_K_M/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf \
    --server 38533 >"$OFFLINE_STATE/tinyq4.log" 2>&1 &
  model_pid=$!
  trap '\''kill "$model_pid" 2>/dev/null || true; wait "$model_pid" 2>/dev/null || true'\'' EXIT
  for attempt in $(seq 1 600); do
    curl --noproxy "*" -fsS http://127.0.0.1:38533/health >/dev/null 2>&1 && break
    kill -0 "$model_pid"
    sleep 0.5
  done
  strace -ff -e trace=network -s 256 -o "$OFFLINE_STATE/network" \
    env PYTHONDONTWRITEBYTECODE=1 \
    HTTP_PROXY= HTTPS_PROXY= ALL_PROXY= http_proxy= https_proxy= all_proxy= \
    NO_PROXY=127.0.0.1,::1 no_proxy=127.0.0.1,::1 \
    python3 /home/tiny/projects/saient/desktop/llm-inference/tools/test_saient_binding_e2e.py \
    --endpoint http://127.0.0.1:38533 \
    --runtime /home/tiny/projects/saient/desktop/llm-inference/src-tauri/resources/saient \
    --state-dir "$OFFLINE_STATE" \
    --manifest-dir /home/tiny/projects/saient/desktop/data/config/saient/bindings
  ip route show
'
```

Result: PASS with ticks `[1, 2, 3, 4]`. The namespace had only `lo` at
`127.0.0.1/::1`, route count remained zero, and traced inference destinations
were only `127.0.0.1:38533` plus the intentional fail-closed probe at
`127.0.0.1:1`. External inference connects: 0.

The installed GUI was also run for 20 seconds under the same no-route namespace
with `strace -ff -e trace=network`. Startup and workspace-runner creation made
0 `AF_INET`/`AF_INET6` socket calls, 0 Internet binds, and 0 Internet connects.
Observed IPC was Unix-domain X11, D-Bus, IBus, and NVIDIA sockets.

## Package and installed process

```sh
cd /home/tiny/projects/saient/desktop/llm-inference
npm run tauri -- build --bundles deb
# Finished release profile; built Saient_1.0.12_amd64.deb

sha256sum src-tauri/target/release/bundle/deb/Saient_1.0.12_amd64.deb
# 41d4ca53f52b5f37d513efed799fd541aca4d4925a82d38c3664e8d43bbe88f5

sudo dpkg -i src-tauri/target/release/bundle/deb/Saient_1.0.12_amd64.deb
# saient 1.0.12 installed over 1.0.11

dpkg-query -W -f='${Package} ${Version} ${Architecture}\n' saient
# saient 1.0.12 amd64
```

The installed binary SHA-256 equals the binary extracted from the package:
`71af1b32bec2de5ca004e6db3902c173204951d920bfce4aa880df09e3535e08`.
The installed binding runtime fingerprint is
`ab47ae6230af5a54e7bee4e1eeef3e0156e2e239993f5477c1d3d8273fa2d102`,
and it accepts the saved v2 manifest.

After restart, `systemctl --user` reported the service active/running. X11
reported one viewable normal Saient window. Opening the mode dialog produced
one sharp dialog in that window; the only additional X surface was a 535x57
`_NET_WM_WINDOW_TYPE_TOOLTIP`, not another normal application window. With the
model unloaded, `lsof -i` for the Saient process tree and `ss` for tinyq4/llama
listeners both returned no entries.

The live runner command retained the exact workspace spelling:

```text
/usr/bin/python3 /usr/lib/Saient/resources/saient/run_saient.py --workspace /home/tiny/projects/saient/desktop/data/projects/Saient begining --interval 30 --enabled-file /home/tiny/projects/saient/desktop/data/saient_enabled
```

`realpath -e` returned that exact path and `stat` reported inode `2065:3934225`.
