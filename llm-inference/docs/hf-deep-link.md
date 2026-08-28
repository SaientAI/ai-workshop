# `saient://` deep links (Hugging Face "Use this model")

Saient answers one link shape:

```
saient://models/huggingface/<owner>/<name>
```

It opens a confirmation prompt naming the repo. Nothing is fetched until the
user picks a specific file, and the import runs through the ordinary Hugging
Face commands, so it hits the existing Internet gate — with Internet off, a link
shows "turn it on in Settings" and does nothing else. A link is remotely
triggerable (any web page can send one), which is why it carries no network
authority of its own and why the repo id is validated in Rust
(`src-tauri/src/hflink.rs`) and not only in the UI.

Pieces:

| Where | What |
|---|---|
| `src-tauri/src/hflink.rs` | Parse + validate the link, hold a pending link, emit `hf-deeplink` |
| `src-tauri/src/main.rs` | `single-instance` (forwards a second launch's argv) then `deep-link` plugin |
| `src-tauri/tauri.conf.json` | `plugins.deep-link.desktop.schemes: ["saient"]` |
| `src-tauri/linux/main.desktop` | Desktop-entry template — see below |
| `src/components/HfImport.svelte` | The prompt; reuses `hf_list_gguf` → `download_starter_model` |
| `src/hfDeeplink.test.js` | Guards all of the above |

## Two things that will bite you

**1. The `.desktop` entry needs `%u`.** Tauri's built-in deb template writes
`Exec=llm-inference` with no argument placeholder. A handler without `%u` is
launched with *no argument* — measured, not assumed — so the scheme registers
and then silently does nothing. `bundle.linux.deb.desktopTemplate` points at
`linux/main.desktop`, which is the built-in template plus `%u`. The AppImage is
built from the same staged deb tree, so it inherits the fix. Windows needs no
equivalent: the NSIS template already registers
`Software\Classes\saient\shell\open\command` with `"%1"`.

**2. A user-level `Saient.desktop` shadows the packaged one.** Desktop entries
are resolved by ID, and `~/.local/share/applications/Saient.desktop` wins over
`/usr/share/applications/Saient.desktop`. If a hand-written launcher with that
name exists (e.g. one that sets `SAIENT_*` env vars) and lacks
`MimeType=x-scheme-handler/saient`, `xdg-open` fails with *"The specified
location is not supported"* even though `xdg-mime query default` looks correct.
Fix on such a machine: add these two things to the user entry —

```
Exec=... %u
MimeType=x-scheme-handler/saient
```

— or delete the user entry and rely on the packaged one.

## Testing it

A debug (`cargo build`) Tauri binary loads `devUrl`, so it shows "Could not
connect to localhost" unless Vite is running on 1421. Either run `npm run dev`
first or test a release build. In a debug build the scheme is registered at
runtime (`register("saient")`); installed builds get it from the packaged
desktop entry instead.

## Verified

| Leg | How |
|---|---|
| Parse + validation | `cargo test hflink` — 4 tests |
| Live link, app already running | `xdg-open` → prompt named the repo (Linux) |
| Cold start, link in argv | Same prompt, via the pending-link path (Linux) |
| Internet gate | With Internet off, "Show files" returns the usual gate message |
| Import | `SmolLM2-135M-Instruct-IQ3_XS.gguf` landed in `models/llm/…` and appeared in the model list |
| Packaged `.deb` / AppImage | `Exec=llm-inference %u` + `MimeType=x-scheme-handler/saient` inside both |
| Installed Linux build | `dpkg -i`, then `xdg-open` reached `/usr/bin/llm-inference` with the URL in argv |
| **Installed Windows build** | **2026-08-28: cross-built NSIS installer on a real Windows laptop — `Win+R` with a `saient://` URL opened the prompt with the correct repo** |

## Next: the Hugging Face side

The app half is done. To make the link reachable from a model page, add an entry
to `LOCAL_APPS` in `huggingface.js/packages/tasks/src/local-apps.ts`:

```typescript
saient: {
	prettyLabel: "Saient",
	docsUrl: "https://saient.co.uk",
	mainTask: "text-generation",
	displayOnModelPage: isLlamaCppGgufModel,
	deeplink: (model) => new URL(`saient://models/huggingface/${model.id}`),
},
```

`isLlamaCppGgufModel` is `!!model.gguf?.context_length`, which is the right
predicate for a tinyq4 host. `mainTask` takes a single `PipelineType`, so one
entry cannot also represent the SDXL/Wan sides of the app.

Add a matching case to `local-apps.spec.ts`, mirroring the `atomic-chat` tests —
one asserting `displayOnModelPage` is true for a GGUF repo and the deeplink
href, and one negative case. There are no published acceptance criteria for
local apps; merging is maintainer discretion. Closed-source and paid apps are
already in the list (LM Studio, Msty).
