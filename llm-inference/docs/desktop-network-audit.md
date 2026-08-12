# Saient Desktop network and IPC audit

This document covers the packaged desktop process in `llm-inference`. It keeps
the user-selected local model API separate from Saient's own IPC.

## Packaged desktop listeners

| Creator | Bind and port | Surface | Authentication and callers | Decision |
| --- | --- | --- | --- | --- |
| Saient Desktop | none | Tauri commands/events use the WebView's native IPC bridge | The application WebView is the caller; there is no TCP port for another local process to open | Keep native IPC |
| `binding_bridge.py` | none | One JSON request/result over inherited stdin/stdout | Child is spawned by the desktop process and dies with it; no independently connectable endpoint | Keep process-private stdio |
| Bundled `tinyq4` child | numeric loopback `127.0.0.1`, OS-selected high port | `GET /health`, `GET /v1/health`, `GET /v1/models`, `POST /v1/chat/completions` | The model API has no bearer authentication, so another process under the same machine can submit inference. It exposes model inference, not Tauri commands, file tools, setup, update, or other privileged desktop actions | HTTP is inherent to the selected OpenAI-compatible model protocol. The desktop now overwrites even a hostile inherited `TINYQ4_BIND=0.0.0.0` with `127.0.0.1`. It never requests a wildcard/LAN bind |
| User-started compatible model selected by Saient | accepted only through a numeric loopback endpoint | The same discovery/inference endpoints | Authentication is determined by that local model server. Saient's binding adapter rejects hostnames, proxies, credentials in URLs, non-loopback addresses, and missing ports | Keep as an explicit local-model boundary |

The packaged production CSP permits frontend connections only to the application
origin. The WebView cannot call even a loopback HTTP/WebSocket service; selected
model HTTP is owned by the Rust/Python backend boundary instead. Production
assets and fonts are bundled, and no CDN origin is allowed.

The Vite server at `127.0.0.1:1421` is a development-only process selected by
`build.devUrl` and an explicit development-only CSP. It is not started or
included by the packaged application.

## Removed desktop HTTP service

The former `remote.rs` service created a desktop-owned HTTP server on port
18788 for phone pairing and image/video operations. It had bearer tokens, but
it duplicated privileged Tauri operations behind a separately connectable
local service. The desktop frontend no longer had a working need for that
surface, and the installed binary could still expose a stale wildcard build.

The server, its Tauri pairing commands, QR dependency, phone settings, and
orphaned remote-video event listeners have been removed. Those operations now
have only their existing Tauri command/event path. No replacement listener was
introduced.

## Outbound connections

| Feature | Destination | Gate and offline behavior |
| --- | --- | --- |
| Full Setup | PyPI/PyTorch package indexes and pinned Hugging Face-hosted model files | Requires a visible, explicit, in-memory setup authorization. The capability is separate from the durable Internet preference and is revoked when setup finishes/closes. Voice/vision files are fetched directly into named `runtime-assets` folders with byte-size checks and `.part` cleanup; setup does not populate a Hugging Face cache |
| Optional in-app model downloads | Hugging Face HTTPS | Requires the durable Internet switch, except the starter model while the first-run setup capability is active |
| Update checks/install | `https://saient.co.uk` | Requires the durable Internet switch before constructing/sending a request; offline startup fails the gate before DNS or a socket syscall |
| Optional RealESRGAN weight | fixed GitHub HTTPS release URL | Requires the durable Internet switch and verifies exact byte length and SHA-256 before installation |
| Normal chat/agent/runtime | the selected numeric loopback model endpoint only | No external connection. The formal binding child clears proxy variables and sets `NO_PROXY` for loopback |
| Vision/TTS/model execution | named local files under Saient's `runtime-assets` folder | Full Setup directly downloads pinned Kokoro, Moondream2, and `moondream/starmie-v1` files, then validates them locally. The setup pins Transformers 4.52.4 because the selected Moondream revision declares that version and does not implement the changed Transformers 5.x tied-weight contract. Every runtime child forces Hub, Transformers, Diffusers, and Datasets offline even when the general Internet switch is on. Small dynamic-module state is run-only under `runtime-tmp` and is removed at app startup |

There is no desktop telemetry, analytics, cloud authentication, licensing call,
remote configuration, crash upload, remote font, or CDN asset path.

## Other Saient services observed on this workstation

These are independently installed household/Pi gateway services, not children,
dependencies, or IPC peers of Saient Desktop:

| Service | Observed listener | Endpoints and controls |
| --- | --- | --- |
| `saient-spotify-gateway.service` | `0.0.0.0:18083` | authenticated `GET /health`, `POST /v1/control`; bearer token, constant-time comparison, source-IP allowlist, action allowlist |
| `saient-terminal-gateway.service` | `0.0.0.0:18084` | authenticated `GET /health`, `POST /v1/tool`, `/v1/approve`, `/v1/reject`; bearer token, source-IP allowlist, sandbox/tool policy |
| `saient-voice-gateway.service` | `0.0.0.0:18085` | authenticated `GET /health` and speech/speaker-lock POST routes; bearer token for POST, source-IP allowlist |

Their systemd units explicitly request wildcard LAN binds for the separate
household/mobile architecture. Removing desktop HTTP does not alter them. They
must not be counted as desktop listeners; conversely, a whole-machine listener
audit must continue to report them rather than pretending the machine has only
the loopback model port.
