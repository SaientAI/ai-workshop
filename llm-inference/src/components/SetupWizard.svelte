<script lang="ts">
  import { onMount } from "svelte";
  import { listen } from "@tauri-apps/api/event";
  import * as T from "../lib/tauri.js";
  import type { SystemInfo } from "../lib/tauri.js";

  let { onDone }: { onDone: () => void } = $props();

  type Step = "choose" | "installing" | "model" | "done";
  let step = $state<Step>("choose");
  let info = $state<SystemInfo | null>(null);
  let profile = $state<"full" | "fast">("full");
  let log = $state<string[]>([]);
  let steps = $state<Record<string, string>>({});  // name → "running"|"done"
  let error = $state("");
  let logEl = $state<HTMLDivElement | null>(null);

  onMount(async () => {
    info = await T.detectSystem().catch(() => null);
  });

  // Auto-scroll the install log.
  $effect(() => {
    void log.length;
    if (logEl) logEl.scrollTop = logEl.scrollHeight;
  });

  const STEP_LABELS: Record<string, string> = {
    engine:   "Inference engine (bundled — GPU/CPU auto-selected)",
    venv:     "Create Python environment",
    torch:    "Install PyTorch (CUDA-matched)",
    creative: "Install diffusers · transformers · kokoro",
    done:     "Finish",
  };
  const stepsFor = $derived(
    profile === "full" ? ["engine", "venv", "torch", "creative"] : ["engine"]
  );

  async function start(p: "full" | "fast") {
    profile = p;
    // The LLM engine is bundled, so Fast needs nothing. Full needs Python for the
    // image/video/voice tools.
    if (p === "full" && !info?.system_python) {
      error = "Full setup needs Python 3.10+ for the creative tools. Install Python 3 (and reopen), or pick Fast.";
      return;
    }
    step = "installing";
    log = []; steps = {}; error = "";
    const unlistenLog = await listen<string>("setup-log", (e) => { log = [...log, e.payload]; });
    const unlistenStep = await listen<{ step: string; status: string }>("setup-step",
      (e) => { steps = { ...steps, [e.payload.step]: e.payload.status }; });
    try {
      await T.runSetup(p);
      step = "model";   // stack installed → offer a starter model
    } catch (e) {
      error = String(e);
    } finally {
      unlistenLog(); unlistenStep();
    }
  }

  async function skip() { await T.skipSetup().catch(() => {}); onDone(); }

  // ── Starter model download ───────────────────────────────────────────────
  import { STARTER_MODELS as MODELS } from "../lib/models.js";
  let downloading = $state("");
  let dlProgress = $state<{ downloaded: number; total: number; done?: boolean }>({ downloaded: 0, total: 0 });
  let dlError = $state("");
  const dlPct = $derived(dlProgress.total > 0 ? Math.round((dlProgress.downloaded / dlProgress.total) * 100) : 0);
  const fmtGB = (b: number) => (b / 1e9).toFixed(2) + " GB";

  async function downloadModel(m: typeof MODELS[number]) {
    downloading = m.file; dlError = ""; dlProgress = { downloaded: 0, total: 0 };
    const dir = await T.getModelsDir().catch(() => "");
    const unlisten = await listen<{ downloaded: number; total: number; done?: boolean }>(
      "model-progress", (e) => { dlProgress = e.payload; });
    try {
      await T.downloadStarterModel(m.repo, m.file, dir);
      step = "done";
    } catch (e) {
      dlError = String(e);
    } finally {
      unlisten(); downloading = "";
    }
  }

  const gpuOk    = $derived(!!info?.gpu_name);
  const pyOk     = $derived(!!info?.system_python);
  const diskOk   = $derived((info?.disk_free_gb ?? 0) >= (profile === "full" ? 12 : 3));
</script>

<div class="wz-backdrop">
  <div class="wz" role="dialog" aria-modal="true" aria-label="Setup">
    <div class="wz-head">
      <span class="wz-logo">Saient</span>
      <span class="wz-sub">first-time setup</span>
    </div>

    {#if step === "choose"}
      <!-- ── System check ─────────────────────────────────────────────── -->
      <div class="wz-sys">
        {#if info}
          <div class="sys-row" class:bad={!gpuOk}>
            <span class="sys-dot" class:ok={gpuOk}></span>
            <span class="sys-k">GPU</span>
            <span class="sys-v">
              {#if gpuOk}{info.gpu_name}{info.vram_gb ? ` · ${info.vram_gb.toFixed(0)} GB` : ""}{:else}none detected — will run on CPU (slow){/if}
            </span>
          </div>
          <div class="sys-row" class:bad={!info.cuda_version}>
            <span class="sys-dot" class:ok={!!info.cuda_version}></span>
            <span class="sys-k">CUDA</span>
            <span class="sys-v">
              {#if info.cuda_version}{info.cuda_version} · torch <code>{info.torch_index}</code>{:else}no NVIDIA driver — CPU torch{/if}
            </span>
          </div>
          <div class="sys-row" class:bad={!pyOk}>
            <span class="sys-dot" class:ok={pyOk}></span>
            <span class="sys-k">Python</span>
            <span class="sys-v">{info.python_version ?? "not found — install Python 3.10+"}</span>
          </div>
          <div class="sys-row" class:bad={!diskOk}>
            <span class="sys-dot" class:ok={diskOk}></span>
            <span class="sys-k">Disk</span>
            <span class="sys-v">{info.disk_free_gb.toFixed(0)} GB free</span>
          </div>
        {:else}
          <div class="sys-row"><span class="sys-dot"></span><span class="sys-v">Detecting system…</span></div>
        {/if}
      </div>

      {#if error}<div class="wz-err">{error}</div>{/if}

      <!-- ── Two paths ────────────────────────────────────────────────── -->
      <div class="wz-paths">
        <button class="wz-card" class:sel={profile === "full"} onclick={() => start("full")}>
          <div class="wz-card-top"><span class="wz-rec">recommended</span></div>
          <div class="wz-card-title">Full setup</div>
          <div class="wz-card-desc">Chat · Agent · Image &amp; Video · Vision · TTS · LoRA</div>
          <div class="wz-card-meta">~6 GB · a few minutes</div>
        </button>
        <button class="wz-card" class:sel={profile === "fast"} onclick={() => start("fast")}>
          <div class="wz-card-top"></div>
          <div class="wz-card-title">Fast setup</div>
          <div class="wz-card-desc">Chat + Agent — engine's bundled, no download</div>
          <div class="wz-card-meta">instant · 0 deps</div>
        </button>
      </div>
      <button class="wz-skip" onclick={skip}>I already have everything — skip setup</button>

    {:else if step === "installing"}
      <!-- ── Install progress ─────────────────────────────────────────── -->
      <div class="wz-steps">
        {#each stepsFor as s}
          <div class="wz-step">
            <span class="wz-step-ic"
              class:run={steps[s] === "running"}
              class:done={steps[s] === "done"}>
              {steps[s] === "done" ? "✓" : steps[s] === "running" ? "⟳" : "○"}
            </span>
            <span class="wz-step-lbl">{STEP_LABELS[s]}</span>
          </div>
        {/each}
      </div>
      <div class="wz-log" bind:this={logEl}>
        {#each log as line}<div class="wz-log-line">{line}</div>{/each}
      </div>
      {#if error}
        <div class="wz-err">{error}</div>
        <div class="wz-actions">
          <button class="wz-btn" onclick={() => (step = "choose")}>← Back</button>
        </div>
      {/if}

    {:else if step === "model"}
      <!-- ── Starter model ────────────────────────────────────────────── -->
      {#if downloading}
        <div class="wz-dl">
          <div class="wz-dl-name">Downloading {downloading}</div>
          <div class="wz-dl-bar"><div class="wz-dl-fill" style="width:{dlPct}%"></div></div>
          <div class="wz-dl-meta">
            {dlPct}% · {fmtGB(dlProgress.downloaded)}{dlProgress.total ? ` / ${fmtGB(dlProgress.total)}` : ""}
          </div>
        </div>
      {:else}
        <div class="wz-model-intro">Pick a model to download, or skip and add your own later.</div>
        <div class="wz-models">
          {#each MODELS as m}
            <button class="wz-card" onclick={() => downloadModel(m)}>
              <div class="wz-card-title">{m.name}</div>
              <div class="wz-card-desc">{m.desc}</div>
              <div class="wz-card-meta">⬇ {m.size}</div>
            </button>
          {/each}
        </div>
        {#if dlError}<div class="wz-err">{dlError}</div>{/if}
        <button class="wz-skip" onclick={() => (step = "done")}>Skip — I'll add a model myself</button>
      {/if}

    {:else}
      <!-- ── Done ─────────────────────────────────────────────────────── -->
      <div class="wz-done">
        <div class="wz-done-ic">✓</div>
        <div class="wz-done-title">You're all set</div>
        <div class="wz-done-desc">
          {profile === "full" ? "Everything's installed." : "Chat + Agent are ready."}
          Pick a model in the sidebar — or hit <b>⬇ Download</b> to grab one from Hugging Face — then Start server.
        </div>
        <ul class="wz-tips">
          <li><b>👁 Vision</b>, <b>🎬 Video</b> and image generation live in the left rail.</li>
          <li><b>🔒</b> in the title bar sets a launch password for shared machines.</li>
          <li><b>⬆</b> checks for updates.</li>
        </ul>
        <button class="wz-btn wz-btn-primary" onclick={onDone}>Get started →</button>
      </div>
    {/if}
  </div>
</div>

<style>
  .wz-backdrop {
    position: fixed; inset: 0; z-index: 2000;
    background: var(--bg); display: flex; align-items: center; justify-content: center;
  }
  .wz {
    width: min(560px, 92vw); max-height: 88vh; overflow: hidden;
    display: flex; flex-direction: column;
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: var(--radius); box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    padding: 24px;
  }
  .wz-head { display: flex; align-items: baseline; gap: 10px; margin-bottom: 18px; }
  .wz-logo { font-size: 17px; font-weight: 700; color: var(--text); letter-spacing: 0.02em; }
  .wz-sub { font-size: 11px; color: var(--text3); text-transform: uppercase; letter-spacing: 0.08em; }

  .wz-sys {
    background: var(--bg3); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 12px 14px; margin-bottom: 18px;
    display: flex; flex-direction: column; gap: 8px;
  }
  .sys-row { display: flex; align-items: center; gap: 10px; font-size: 12px; }
  .sys-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--amber); flex-shrink: 0; }
  .sys-dot.ok { background: var(--green); box-shadow: 0 0 5px var(--green); }
  .sys-row.bad .sys-dot { background: var(--amber); }
  .sys-k { width: 56px; color: var(--text3); font-family: var(--mono); font-size: 11px; }
  .sys-v { color: var(--text2); }
  .sys-v code { font-family: var(--mono); color: var(--accent); background: var(--bg); padding: 1px 5px; border-radius: 3px; }

  .wz-paths { display: flex; gap: 12px; margin-bottom: 12px; }
  .wz-card {
    flex: 1; text-align: left; padding: 14px; border-radius: var(--radius);
    background: var(--bg3); border: 1px solid var(--border); cursor: pointer;
    transition: border-color 0.15s, background 0.15s;
  }
  .wz-card:hover { border-color: var(--accent); background: rgba(108,142,245,0.08); }
  .wz-card-top { min-height: 16px; margin-bottom: 6px; }
  .wz-rec {
    font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--green); background: rgba(0,214,143,0.12);
    padding: 2px 7px; border-radius: 10px;
  }
  .wz-card-title { font-size: 15px; font-weight: 700; color: var(--text); margin-bottom: 5px; }
  .wz-card-desc { font-size: 11px; color: var(--text2); line-height: 1.5; margin-bottom: 8px; }
  .wz-card-meta { font-size: 10px; color: var(--text3); font-family: var(--mono); }

  .wz-skip {
    background: transparent; border: none; color: var(--text3);
    font-size: 11px; cursor: pointer; padding: 4px; text-decoration: underline;
    align-self: center;
  }
  .wz-skip:hover { color: var(--text2); }

  .wz-steps { display: flex; flex-direction: column; gap: 7px; margin-bottom: 14px; }
  .wz-step { display: flex; align-items: center; gap: 10px; font-size: 13px; }
  .wz-step-ic {
    width: 18px; text-align: center; font-family: var(--mono);
    color: var(--text3);
  }
  .wz-step-ic.run { color: var(--amber); animation: wz-spin 1.2s linear infinite; display: inline-block; }
  .wz-step-ic.done { color: var(--green); }
  @keyframes wz-spin { to { transform: rotate(360deg); } }
  .wz-step-lbl { color: var(--text2); }

  .wz-log {
    flex: 1; min-height: 180px; max-height: 320px; overflow-y: auto;
    background: var(--bg0, #0a0a0c); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 10px 12px;
    font-family: var(--mono); font-size: 11px; line-height: 1.5; color: var(--text2);
  }
  .wz-log-line { white-space: pre-wrap; word-break: break-all; }

  .wz-err {
    background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.3);
    color: var(--red); font-size: 12px; padding: 10px 12px;
    border-radius: var(--radius-sm); margin: 12px 0; line-height: 1.5;
  }

  /* Starter model step */
  .wz-model-intro { font-size: 12px; color: var(--text2); margin-bottom: 14px; }
  .wz-models { display: flex; flex-direction: column; gap: 10px; margin-bottom: 12px; }
  .wz-models .wz-card { flex: none; }
  .wz-dl { padding: 24px 0; }
  .wz-dl-name { font-size: 13px; color: var(--text); margin-bottom: 12px; font-family: var(--mono); word-break: break-all; }
  .wz-dl-bar { height: 8px; background: var(--bg3); border-radius: 4px; overflow: hidden; }
  .wz-dl-fill { height: 100%; background: var(--accent); border-radius: 4px; transition: width 0.3s ease; }
  .wz-dl-meta { font-size: 11px; color: var(--text3); font-family: var(--mono); margin-top: 8px; }

  .wz-done { text-align: center; padding: 16px 0; }
  .wz-done-ic {
    width: 48px; height: 48px; margin: 0 auto 14px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px; color: var(--green);
    background: rgba(0,214,143,0.12); border: 1px solid rgba(0,214,143,0.3);
  }
  .wz-done-title { font-size: 17px; font-weight: 700; color: var(--text); margin-bottom: 8px; }
  .wz-done-desc { font-size: 12px; color: var(--text2); line-height: 1.6; margin-bottom: 18px; max-width: 420px; margin-inline: auto; }
  .wz-done-desc b { color: var(--text); }
  .wz-tips {
    list-style: none; text-align: left; max-width: 420px; margin: 0 auto 20px;
    display: flex; flex-direction: column; gap: 7px;
  }
  .wz-tips li {
    font-size: 11.5px; color: var(--text2); line-height: 1.5;
    padding: 8px 11px; background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm);
  }
  .wz-tips b { color: var(--text); }

  .wz-actions { display: flex; justify-content: flex-end; gap: 8px; }
  .wz-btn {
    padding: 8px 16px; border-radius: var(--radius-sm); font-size: 13px;
    background: var(--bg3); border: 1px solid var(--border); color: var(--text2); cursor: pointer;
  }
  .wz-btn:hover { border-color: var(--accent); color: var(--text); }
  .wz-btn-primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 600; }
  .wz-btn-primary:hover { background: #7a9cf7; color: #fff; }
</style>
