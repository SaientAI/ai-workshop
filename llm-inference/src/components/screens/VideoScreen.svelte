<script lang="ts">
  import { onMount } from "svelte";
  import { save } from "@tauri-apps/plugin-dialog";
  import { listen } from "@tauri-apps/api/event";
  import { video } from "../../lib/state.svelte.js";
  import * as T from "../../lib/tauri.js";

  // Simple (default) hides every tunable behind Auto; Advanced shows the full panel.
  const ADV_KEY = "saient.video.advanced";
  let advanced = $state(localStorage.getItem(ADV_KEY) === "1");

  // For extend chaining: the previous full clip b64 we will append the next generated segment to.
  // Set by the Extend button; cleared after the generate roundtrip.
  let appendFromB64 = $state("");
  let stationFrame = $state(0);
  let stationFrameB64 = $state("");
  let stationFrameName = $state("");
  let stationCapturedFrame = $state<number | null>(null);
  let stationAnchorFrame = $state<number | null>(null);
  let stationPlaybackRate = $state(1);
  function setAdvanced(on: boolean) {
    advanced = on;
    localStorage.setItem(ADV_KEY, on ? "1" : "0");
    if (!on) autoSet();   // returning to Simple snaps everything back to known-good
  }

  const loaded = $derived(!!video.loadedPath && video.loadedPath === video.modelPath);
  const pct = $derived(video.progressTotal > 0 ? Math.round((video.progress / video.progressTotal) * 100) : 0);

  const selectedModel = $derived(video.models.find((m) => m.path === video.modelPath));
  const selectedModelText = $derived(
    `${video.modelPath || ""} ${video.loadedPath || ""} ${selectedModel?.label || ""} ${(selectedModel as any)?.pipeline || ""}`.toLowerCase()
  );
  function isTi2v5bText(text: string) {
    const t = text.toLowerCase();
    return t.includes("wan2.2") && t.includes("ti2v") && t.includes("5b");
  }
  // Strict 14B detection. 5B is now its own 480p-first path, not a fallback.
  const is14bSelected = $derived((selectedModelText.includes("14b") || selectedModelText.includes("a14b")) && !selectedModelText.includes("5b"));
  const isTi2v5bSelected = $derived(isTi2v5bText(selectedModelText));
  const isWanVideoSelected = $derived(is14bSelected || isTi2v5bSelected);
  const canTailCondition = $derived(
    selectedModelText.includes("i2v") ||
    selectedModelText.includes("ti2v") ||
    selectedModelText.includes("imagetovideo")
  );
  // ── Activity log (non-typable mini terminal under Params) ──────────────────
  let termEl: HTMLDivElement | undefined = $state();

  // Wan/SVI NSFW failure mode: vulva gets rendered as a mouth/lips, or genitals fuse
  // into a penis+vagina hybrid. Positive lock + hard negatives fix most of it; refine
  // with the same language pulls multi-chunk 30s chains back into shape.
  const ANATOMY_NEG =
    "bad anatomy, deformed genitals, fused genitals, ambiguous genitals, hermaphrodite, futanari, " +
    "mouth between legs, lips instead of vagina, labia as lips, oral opening as genitals, " +
    "teeth on crotch, face on genitals, penis growing from vagina, vagina and penis fused, " +
    "dick-vagina hybrid, inverted genitals, missing genitals, malformed labia, " +
    "anatomically incorrect genitals, wrong sex organs";
  const ANATOMY_POS =
    "anatomically correct female genitalia, clear detailed vulva with distinct labia majora " +
    "and labia minora and clitoris, natural vaginal opening (not a mouth, not lips, no teeth), " +
    "no penis, no fused sex organs, realistic intimate anatomy";
  const QUALITY_LONG_POS =
    "sharp high-fidelity detail, consistent identity, stable lighting and exposure, " +
    "no quality drop, no blur creep, smooth natural motion";

  function looksExplicitAnatomy(p: string) {
    return /\b(pussy|vagina|vulva|labia|clitoris|clit|genital|cunnilingus|nude|naked|nsfw|sex|erotic|breast|nipple|penis|cock|cum|creampie|masturbat)\b/i.test(p || "");
  }
  function mergeCsv(base: string, extra: string) {
    const seen = new Set<string>();
    const out: string[] = [];
    for (const part of `${base || ""}, ${extra || ""}`.split(",")) {
      const t = part.trim();
      if (!t) continue;
      const key = t.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(t);
    }
    return out.join(", ");
  }
  function withAnatomyPos(prompt: string, longClip = false) {
    let p = (prompt || "").trim();
    if (!p) return p;
    if (longClip && !/high-fidelity|no quality drop|consistent identity/i.test(p)) {
      p = `${p}, ${QUALITY_LONG_POS}`;
    }
    if (!video.anatomyLock || !looksExplicitAnatomy(p)) return p;
    if (/anatomically correct|labia majora|vulva with distinct/i.test(p)) return p;
    return `${p}, ${ANATOMY_POS}`;
  }
  function withAnatomyNeg(neg: string, prompt = "") {
    let n = neg || "";
    // Always keep the core anti-confusion terms for explicit runs (and when lock is on).
    if (video.anatomyLock || looksExplicitAnatomy(prompt) || looksExplicitAnatomy(n)) {
      n = mergeCsv(n, ANATOMY_NEG);
    }
    return n;
  }
  function ensureLongClipQualitySettings() {
    if (isTi2v5bSelected) {
      const bumped = video.steps < 50 || video.cfg !== 5 || video.fps !== 24;
      video.steps = Math.max(video.steps, 50);
      video.cfg = 5;
      video.fps = 24;
      video.scheduler = "auto";
      video.shift = 5;
      if (bumped) logln("quality floor: TI2V-5B uses official 50 steps · CFG 5 · 24 FPS timing");
      return;
    }
    // SPEED PATH: Lightning is distilled for 4 steps. Bumping to 20 on a 4-chunk 30s
    // run turned ~12 min into ~66 min (80 dual-expert steps). Keep the distill recipe.
    // Quality lever for long Lightning is Refine after, not more denoise steps.
    const ll = (video.loras.find((l) => l.path === video.loraPath)?.label || video.loraPath || "").toLowerCase();
    const isLightning = /lightning|4.?step/.test(ll) && !/svi/.test(ll);
    if (isLightning) {
      if (video.steps !== 4 || video.cfg !== 1) {
        video.steps = 4;
        video.cfg = 1;
        video.shift = Math.max(video.shift || 5, 5);
        if (is14bSelected) {
          video.loraProfile = "high_low";
          video.loraHighStrength = 1.0;
          video.loraLowStrength = 1.0;
          video.loraSplitStep = 2;
        }
        logln("speed path: Lightning long clip stays at 4 steps · CFG 1 (use Refine after for quality)");
      }
      return;
    }
    // Non-distill long clips: a soft floor so ultra-low custom settings don't smear,
    // but never force a multi-× slowdown like the old 4→20 Lightning bump.
    if (video.steps < 8 && is14bSelected) {
      video.steps = 8;
      video.cfg = Math.max(video.cfg, 1.5);
      logln("quality floor: long clip steps raised to 8 (non-Lightning)");
    }
  }

  function compactGpuMessage(raw: unknown) {
    const text = String(raw);
    if (/cuda out of memory|out of memory|cuda.*memory/i.test(text)) {
      const marker = text.trimStart().startsWith("✗") || /failed/i.test(text)
        ? "✗ "
        : text.trimStart().startsWith("⚠")
          ? "⚠ "
          : "";
      const tried = text.match(/Tried to allocate\s+([0-9.]+\s*(?:MiB|GiB))/i)?.[1];
      const free = text.match(/([0-9.]+\s*(?:MiB|GiB))\s+is free/i)?.[1];
      const total = text.match(/total capacity of\s+([0-9.]+\s*GiB)/i)?.[1];
      const details = [
        tried ? `needed ${tried}` : "",
        free ? `${free} free` : "",
        total ? `${total} GPU` : "",
      ].filter(Boolean).join(" · ");
      return `${marker}GPU out of memory${details ? ` — ${details}` : ""}. Lower frames/resolution, unload other models, or close other GPU apps.`;
    }
    return text;
  }

  function logln(s: string) {
    const t = new Date().toLocaleTimeString([], { hour12: false });
    const line = compactGpuMessage(s);
    const last = video.log[video.log.length - 1];
    if (last && last.endsWith(line)) return;       // skip repeated identical lines
    video.log = [...video.log, `${t}  ${line}`].slice(-300);
  }
  // Auto-scroll to the newest line whenever the log grows.
  $effect(() => {
    void video.log.length;
    if (termEl) termEl.scrollTop = termEl.scrollHeight;
  });

  // Optional dev-only auto-run. Keep it off by default so loading a heavy 14B model
  // never starts an expensive generation without the user pressing Generate.
  $effect(() => {
    if (is14bSelected) {
      // Dual-expert 14B never supports quality bf16 mode (would error). Force fast 4-bit + Lightning.
      if (video.qualityMode) video.qualityMode = false;
    }
    if (localStorage.getItem("saient.video.autorun") !== "1") return;
    if (video.loadedPath && !video.generating && !(window as any).__saientTestRan) {
      const modelL = (video.modelPath || '').toLowerCase();
      const is14bT2v = is14bSelected || modelL.includes('a14b') || modelL.includes('14b') || modelL.includes('wan2.2-t2v');
      const isAutoRunWan = is14bT2v || isTi2v5bSelected;
      if (isAutoRunWan) {
        (window as any).__saientTestRan = true;
        setTimeout(() => {
          if (isTi2v5bSelected) {
            if (!video.resLocked) { video.width = 832; video.height = 480; }
            video.steps = 50; video.cfg = 5; video.fps = 24; video.shift = 5;
          } else if (!video.resLocked) {
            video.width = 640; video.height = 480; // proven safe high-quality for 14B on 16GB
          }
          if (video.numFrames > 200) {
            video.numFrames = isTi2v5bSelected ? 121 : 81; // native debug segment for the selected model
          }
          const autoLabel = isTi2v5bSelected ? "TI2V-5B" : "14B (SVI/T2V)";
          if (video.storyboardPrompts && video.storyboardPrompts.some((p: string) => p && p.trim())) {
            logln(`Auto-running storyboard with ${autoLabel}...`);
            generateStoryboard();
          } else if (video.prompt && video.prompt.trim()) {
            logln(`Auto-running generate with ${autoLabel}...`);
            generate();
          }
        }, 300);
      }
    }
  });

  onMount(async () => {
    video.models = await T.videoScanModels().catch(() => []) as typeof video.models;
    video.loras = await T.videoScanLoras().catch(() => []) as typeof video.loras;
    const [cur, curLora] = await Promise.all([
      T.videoLoadedModel().catch(() => null),
      T.videoLoadedLora().catch(() => null),
    ]);
    if (cur) {
      video.loadedPath = cur;
      video.modelPath = cur;
      video.loraPath = curLora || "";
    }
    else if (video.models.length && !video.modelPath) video.modelPath = video.models[0].path;
    autoSet();   // snap params to the initially-selected model
    seedBuiltinRecipes();
  });

  async function refresh() {
    video.models = await T.videoScanModels().catch(() => video.models) as typeof video.models;
    video.loras = await T.videoScanLoras().catch(() => video.loras) as typeof video.loras;
    const [cur, curLora] = await Promise.all([
      T.videoLoadedModel().catch(() => null),
      T.videoLoadedLora().catch(() => null),
    ]);
    video.loadedPath = cur || "";
    if (cur) {
      video.modelPath = cur;
      video.loraPath = curLora || "";
    }
    seedBuiltinRecipes();
  }

  function seedBuiltinRecipes() {
    // Delete any old non-working "dp" / dp_reverse recipes or loras from saved list.
    const before = recipes.length;
    recipes = recipes.filter((r) => !/dp|dp_reverse|dreamprompt/i.test(r.name || r.loraPath || ''));
    if (recipes.length < before) {
      persistRecipes();
      logln('✓ removed old non-working dp/dp_reverse recipe(s)');
    }

    // Seed a SVI recipe if the files are present and not already in user's saved recipes.
    const hasSVI = video.loras.some((l: any) => /svi.*(high|low)|SVI_v2_PRO/i.test(l.label || l.path || ''));
    if (!hasSVI) return;
    const already = recipes.some((r) => /SVI v2 PRO/i.test(r.name));
    if (already) return;
    const sviHigh = video.loras.find((l: any) => /SVI.*HIGH|SVI_v2_PRO.*HIGH/i.test(l.label || l.path || ''))?.path || '';
    if (!sviHigh) return;
    const builtin: VideoRecipe = {
      name: 'SVI v2 PRO (I2V-A14B High/Low)',
      modelPath: video.modelPath || '',
      loraPath: sviHigh,
      loraProfile: 'high_low',
      loraStrength: 1,
      loraHighStrength: 1.0,
      loraLowStrength: 1.0,
      loraSplitStep: 10,
      steps: 20,
      cfg: 2.5,
      scheduler: 'auto',
      shift: 8,
      width: 640,
      height: 480,
      numFrames: 81,
      fps: 16,
      qualityMode: false,
    };
    recipes = [...recipes, builtin];
    persistRecipes();
    logln('✓ seeded built-in SVI v2 PRO recipe (click to load, then Load Model)');
  }

  // Per-model best/required settings. Different families want very different params
  // (FastWan: 3 steps/CFG1; CogVideoX: hard-locked 720×480/50/CFG6/8fps). "Auto"
  // snaps everything to the loaded model's sweet spot so you don't fight it.
  function presetFor(label: string, pipeline: string) {
    const l = label.toLowerCase(), p = pipeline.toLowerCase();
    if (p.includes("cogvideo"))
      // 49 frames is NATIVE for CogVideoX-5b-I2V — it MUST be 49 (the model is trained at
      // 49 and its temporal position embeddings assume it). A previous "25 frames to reduce
      // drift" tweak actually CORRUPTED the temporal embeddings → the whole clip diverged
      // into a rainbow/waffle field by mid-clip. 49 is both coherent AND fits 16 GB (~14.5 GB
      // peak with VAE tiling). Do not lower this below 49.
      return { width: 720, height: 480, numFrames: 49, steps: 50, cfg: 6, fps: 8, scheduler: "auto" as const, shift: 5, resLocked: true };
    if (l.includes("instax"))   // dp/uncensored bake needs more passes for fine detail
      return { width: 832, height: 480, numFrames: 49, steps: 8,  cfg: 1, fps: 16, scheduler: "auto" as const, shift: 8, resLocked: false };
    if (l.includes("fastwan"))
      return { width: 832, height: 480, numFrames: 49, steps: 3,  cfg: 1, fps: 16, scheduler: "auto" as const, shift: 5, resLocked: false };
    if (l.includes("wan2.2") && l.includes("t2v") && l.includes("a14b"))
      // Base 14B dual-expert recipe. Native First Block Cache is selected separately and
      // leaves the adapter slot available for the user's own LoRA.
      return { width: 832, height: 480, numFrames: 81, steps: 40, cfg: 4, fps: 16, scheduler: "auto" as const, shift: 5, resLocked: false };
    if (l.includes("ti2v-5b"))
      // Official TI2V-5B recipe is 121f @ 24 FPS, 50 steps, CFG 5, shift 5. Keep
      // the app 480p-first for this 16 GB display GPU; use Refine/Upscale after.
      return { width: 832, height: 480, numFrames: 121, steps: 50, cfg: 5, fps: 24, scheduler: "auto" as const, shift: 5, resLocked: false };
    if (l.includes("wan2.2"))
      return { width: 832, height: 480, numFrames: 49, steps: 30, cfg: 5, fps: 16, scheduler: "auto" as const, shift: 5, resLocked: false };
    return { width: 832, height: 480, numFrames: 49, steps: 30, cfg: 5, fps: 16, scheduler: "auto" as const, shift: 5, resLocked: false }; // Wan2.1 default
  }
  function autoSet() {
    const m = video.models.find((x) => x.path === video.modelPath);
    if (!m) return;
    const pr = presetFor(m.label, (m as any).pipeline ?? "");
    const ti2v5b = isTi2v5bText(`${m.path} ${m.label} ${(m as any).pipeline ?? ""}`);
    video.width = pr.width; video.height = pr.height; video.numFrames = pr.numFrames;
    video.steps = pr.steps; video.cfg = pr.cfg; video.fps = pr.fps;
    video.scheduler = pr.scheduler; video.shift = pr.shift; video.resLocked = pr.resLocked;
    if (ti2v5b) {
      video.width = 832;
      video.height = 480;
      video.numFrames = 121;
      video.steps = 50;
      video.cfg = 5;
      video.fps = 24;
      video.scheduler = "auto";
      video.shift = 5;
      video.resLocked = false;
      video.qualityMode = true;
      video.lowVramMode = false;
      video.blockOffload = false;
      video.doRefine = true;
      video.doUpscale = true;
      video.doInterpolate = false;
    }
    adjustResolutionForLength();
    video.denoiseCache = "off";
    // Auto adjusts model parameters only. The selected LoRA is user-owned state and must
    // never be replaced with a distillation adapter behind their back.
    loraAutoTune();
  }

  // A speed-distill LoRA dictates steps/CFG no matter which model it sits on — running
  // one at the model's default (e.g. 40 steps · CFG 4) is both slower AND worse: CFG > 1
  // doubles the passes and pushes the distilled trajectory past where it was trained.
  function loraAutoTune() {
    const ll = video.loras.find((l) => l.path === video.loraPath)?.label ?? "";
    if (!ll) return;
    if (/lightning|4.?step/i.test(ll)) {
      video.steps = 4; video.cfg = 1; video.scheduler = "auto"; video.shift = 5;
      // For A14B dual use high_low profile so it applies HIGH then LOW expert correctly
      if (is14bSelected) {
        video.loraProfile = "high_low";
        video.loraHighStrength = 1.0;
        video.loraLowStrength = 1.0;
        video.loraSplitStep = 2;
      } else {
        video.loraProfile = "single"; video.loraStrength = 1;
      }
      logln("auto: Lightning 4-step → 4 steps · CFG 1 (14B uses high/low split). 14B highest quality only.");
    } else if (/lightx2v|stepdistill|8.?step/i.test(ll)) {
      applyLightx2vRecipe();
    } else if (/svi|svi_v2|stable-video-infinity/i.test(ll)) {
      applySVIv2Recipe();
    }
  }
  function setScheduler(mode: "auto" | "euler_beta") { video.scheduler = mode; }

  // ── Recipes: name and restore a whole working setup (model + LoRA + every param) ──
  type VideoRecipe = {
    name: string; modelPath: string; loraPath: string; loraProfile: "single" | "high_low";
    loraStrength: number; loraHighStrength: number; loraLowStrength: number; loraSplitStep: number;
    steps: number; cfg: number; scheduler: "auto" | "euler_beta"; shift: number;
    width: number; height: number; numFrames: number; fps: number; qualityMode: boolean;
    denoiseCache?: "off" | "balanced";
  };
  const RECIPES_KEY = "saient.video.recipes";
  let recipes = $state<VideoRecipe[]>([]);
  try { recipes = JSON.parse(localStorage.getItem(RECIPES_KEY) ?? "[]"); } catch { recipes = []; }
  let recipeName = $state("");
  function persistRecipes() { localStorage.setItem(RECIPES_KEY, JSON.stringify(recipes)); }
  function saveRecipe() {
    const modelLabel = video.models.find((m) => m.path === video.modelPath)?.label ?? "model";
    const name = recipeName.trim() || `${modelLabel} · ${video.steps}st cfg${video.cfg}`;
    const r: VideoRecipe = {
      name, modelPath: video.modelPath, loraPath: video.loraPath, loraProfile: video.loraProfile,
      loraStrength: video.loraStrength, loraHighStrength: video.loraHighStrength,
      loraLowStrength: video.loraLowStrength, loraSplitStep: video.loraSplitStep,
      steps: video.steps, cfg: video.cfg, scheduler: video.scheduler, shift: video.shift,
      width: video.width, height: video.height, numFrames: video.numFrames, fps: video.fps,
      qualityMode: video.qualityMode,
      denoiseCache: video.denoiseCache,
    };
    const i = recipes.findIndex((x) => x.name === name);
    if (i >= 0) recipes[i] = r; else recipes.push(r);
    persistRecipes(); recipeName = "";
    logln(`✓ recipe saved: ${name}`);
  }
  function applyRecipe(r: VideoRecipe) {
    const needsReload = r.modelPath !== video.loadedPath || r.loraPath !== video.loraPath
      || r.qualityMode !== video.qualityMode;
    video.modelPath = r.modelPath; video.loraPath = r.loraPath; video.loraProfile = r.loraProfile;
    video.loraStrength = r.loraStrength; video.loraHighStrength = r.loraHighStrength;
    video.loraLowStrength = r.loraLowStrength; video.loraSplitStep = r.loraSplitStep;
    video.steps = r.steps; video.cfg = r.cfg; video.scheduler = r.scheduler; video.shift = r.shift;
    video.width = r.width; video.height = r.height; video.numFrames = r.numFrames; video.fps = r.fps;
    video.qualityMode = r.qualityMode; video.resLocked = false;
    video.denoiseCache = r.denoiseCache ?? "off";
    logln(`recipe: ${r.name}${needsReload ? " → hit Load to apply the model/LoRA" : ""}`);
  }
  function deleteRecipe(name: string) {
    recipes = recipes.filter((r) => r.name !== name);
    persistRecipes();
  }

  // ── Simple-tab one-click presets: pick → Load → Generate ─────────────────
  type SimplePresetId = "t2v_30s_fast" | "t2v_5s_fast" | "t2v_hd_3s" | "t2v_hd_5s_max" | "ti2v_5b_480p_hq" | "ti2v_5b_480p_10s" | "i2v_svi_5s";
  let activeSimplePreset = $state<SimplePresetId | "">("");
  // Max native 720p (1280×704) frame count that fits FastWan-14B on the 16 GB card.
  // Measured on the 5060 Ti: 49f (~3s) peak ~15.1 GB with safe margin; 65f (~4s) completes
  // but peaks ~15.8 GB (ragged edge — risky once the app's own UI takes GPU); 73f+ (≥4.5s)
  // OOMs at denoise. Kept at 49f so a preset can never OOM-lock the app. For 5s+ in HD,
  // gen 480p 5s then Enhance → RealESRGAN ×2 (≈960p+).
  const HD_MAX_FRAMES = 49;

  function findT2vA14b() {
    return video.models.find((m) => {
      const t = `${m.path} ${m.label}`.toLowerCase();
      return t.includes("t2v") && (t.includes("a14b") || t.includes("14b")) && !t.includes("5b");
    }) ?? video.models.find((m) => {
      const t = `${m.path} ${m.label}`.toLowerCase();
      return t.includes("wan2.2") && t.includes("t2v") && !t.includes("5b");
    }) ?? null;
  }
  function findI2vA14b() {
    return video.models.find((m) => {
      const t = `${m.path} ${m.label}`.toLowerCase();
      return (t.includes("i2v") || t.includes("ti2v")) && (t.includes("a14b") || t.includes("14b")) && !t.includes("5b");
    }) ?? null;
  }
  // Single-transformer FastWan-14B: the HD workhorse. Runs full fp32 at 720p (dual-expert
  // A14B is jammed at the 16 GB ceiling there and needs bf16 rope/latents that soften it),
  // no expert-boundary swap, baked 4-step distill so no external LoRA needed. Measured
  // 720p 49f fits at ~13.8 GB live and is razor sharp.
  function findFastWan14b() {
    return video.models.find((m) => {
      const t = `${m.path} ${m.label}`.toLowerCase();
      return t.includes("fastwan") && t.includes("t2v") && t.includes("14b")
        && !t.includes("a14b") && !t.includes("insta") && !t.includes("i2v");
    }) ?? null;
  }
  function findTi2v5b() {
    return video.models.find((m) => isTi2v5bText(`${m.path} ${m.label} ${(m as any).pipeline ?? ""}`)) ?? null;
  }
  function findLightningLora() {
    return video.loras.find((l) => /lightning/i.test(l.label) && /high/i.test(l.label))
      ?? video.loras.find((l) => /lightning|4.?step/i.test(l.label))
      ?? null;
  }
  function findSviHighLora() {
    return video.loras.find((l) => /SVI.*HIGH|SVI_v2_PRO.*HIGH/i.test(l.label))
      ?? video.loras.find((l) => /svi/i.test(l.label) && /high/i.test(l.label))
      ?? null;
  }

  function applySimplePreset(id: SimplePresetId) {
    activeSimplePreset = id;
    video.qualityMode = false;
    video.resLocked = false;
    video.lowVramMode = false;
    video.blockOffload = false;
    video.denoiseCache = "off";
    video.anatomyLock = true;
    video.fps = 16;
    video.scheduler = "auto";
    video.seed = -1;

    if (id === "ti2v_5b_480p_hq" || id === "ti2v_5b_480p_10s") {
      const m = findTi2v5b();
      if (!m) {
        video.error = "No Wan2.2 TI2V-5B model found. Drop it in and Refresh.";
        logln(`✗ preset: ${video.error}`);
        return;
      }
      video.modelPath = m.path;
      video.loraPath = "";
      video.loraProfile = "single";
      video.loraStrength = 1;
      video.loraHighStrength = 1;
      video.loraLowStrength = 1;
      video.loraSplitStep = 4;
      video.qualityMode = true;
      video.lowVramMode = id === "ti2v_5b_480p_10s";
      video.blockOffload = false;
      video.doRefine = true;
      video.doUpscale = true;
      video.doInterpolate = false;
      video.fps = 24;
      video.steps = 50;
      video.cfg = 5;
      video.shift = 5;
      video.width = 832;
      video.height = 480;
      video.numFrames = id === "ti2v_5b_480p_10s" ? framesFor(10) : 121;
      const label = id === "ti2v_5b_480p_10s"
        ? "5B HQ 10s 480p (2x native chunks)"
        : "5B HQ 5s 480p";
      logln(`preset: ${label} · ${m.label} · bf16 quality · 50 steps · CFG 5 · 24 FPS · ${video.numFrames}f`);
      logln("480p-first path: Generate, then Quality Pass Refine + Upscale to hand off toward 720p.");
      logln(loaded && video.loadedPath === m.path
        ? "✓ model already loaded — paste prompt → Generate"
        : "→ hit Load Model, then paste prompt → Generate");
      return;
    }

    // HD 3s / HD 5s (max) / short 5s → single-transformer FastWan-14B (full fp32, sharp, no
    // boundary). 30s stays on A14B below — only the dual-expert path chunks the long clip.
    if (id === "t2v_hd_3s" || id === "t2v_hd_5s_max" || id === "t2v_5s_fast") {
      const fast = findFastWan14b();
      if (fast) {
        video.modelPath = fast.path;
        video.loraPath = "";
        video.loraProfile = "single";
        video.loraStrength = 1;
        video.steps = 4;
        video.cfg = 1;
        video.shift = 5;
        if (id === "t2v_hd_3s") {
          video.width = 1280;
          video.height = 704;
          video.numFrames = HD_MAX_FRAMES; // native 720p ceiling on the 16 GB card (5s/81f OOMs)
        } else if (id === "t2v_hd_5s_max") {
          // SAFE 5s@720p: native 5s@480p (fits VRAM cleanly) then the 2× ESRGAN upscale
          // pass lifts it to ~1664×960 HD. Parking fp32 (54 GB) into 39 GB RAM froze the
          // whole PC — never again. This route is freeze-proof and still HD-class.
          video.width = 832;
          video.height = 480;
          video.numFrames = framesFor(5);
          video.blockOffload = false;
          video.doUpscale = true;   // pre-arm the Enhance → 2× upscale to reach HD
          video.doRefine = true;
        } else {
          video.width = 832;
          video.height = 480;
          video.numFrames = framesFor(5); // 480p 5s fits FastWan-14B easily
        }
        const secs = clipSecondsFromFrames(video.numFrames, 16);
        const label = id === "t2v_hd_3s" ? `HD ${secs}s T2V`
          : id === "t2v_hd_5s_max" ? "5s T2V (480p → Enhance ×2 to HD)" : "5s T2V";
        logln(`preset: ${label} · ${fast.label} · FastWan 4-step (fp32) · ${video.width}×${video.height} · ${video.numFrames}f`
          + (id === "t2v_hd_5s_max" ? " · then click Enhance for 2× HD upscale (no PC-freezing RAM park)" : ""));
        logln(loaded && video.loadedPath === fast.path
          ? "✓ model already loaded — paste prompt → Generate"
          : "→ hit Load Model, then paste prompt → Generate");
        return;
      }
      logln("⚠ FastWan-14B not found — falling back to A14B + Lightning (720p tighter/softer)");
    }

    if (id === "t2v_30s_fast" || id === "t2v_5s_fast" || id === "t2v_hd_3s" || id === "t2v_hd_5s_max") {
      const m = findT2vA14b();
      if (!m) {
        video.error = "No Wan2.2 T2V-A14B model found. Drop the model in and Refresh.";
        logln(`✗ preset: ${video.error}`);
        return;
      }
      const lora = findLightningLora();
      video.modelPath = m.path;
      video.loraPath = lora?.path ?? "";
      video.loraProfile = "high_low";
      video.loraStrength = 1;
      video.loraHighStrength = 1;
      video.loraLowStrength = 1;
      video.loraSplitStep = 2;
      video.steps = 4;
      video.cfg = 1;
      video.shift = 5;
      if (id === "t2v_hd_3s") {
        video.width = 1280;
        video.height = 704;
        video.numFrames = 49; // ~3s @16fps, native HD unit on 16GB
      } else {
        video.width = 832;
        video.height = 480;
        video.numFrames = framesFor(id === "t2v_30s_fast" ? 30 : 5);
      }
      const label = id === "t2v_30s_fast" ? "30s Fast T2V"
        : id === "t2v_5s_fast" ? "5s Fast T2V"
        : id === "t2v_hd_5s_max" ? "5s T2V (FastWan absent → 480p fallback)"
        : "HD 3s T2V";
      logln(`preset: ${label} · ${m.label} · Lightning 4-step · ${video.width}×${video.height} · ${video.numFrames}f`);
      if (!lora) logln("⚠ no Lightning LoRA found — gen will be slow without it");
      logln(loaded && video.loadedPath === m.path
        ? "✓ model already loaded — paste prompt → Generate"
        : "→ hit Load Model, then paste prompt → Generate");
      return;
    }

    if (id === "i2v_svi_5s") {
      const m = findI2vA14b();
      if (!m) {
        video.error = "No Wan I2V-A14B model found for SVI preset. Drop it in and Refresh.";
        logln(`✗ preset: ${video.error}`);
        return;
      }
      const svi = findSviHighLora();
      video.modelPath = m.path;
      video.loraPath = svi?.path ?? "";
      video.loraProfile = "high_low";
      video.loraStrength = 1;
      video.loraHighStrength = 1;
      video.loraLowStrength = 1;
      video.loraSplitStep = 10;
      video.steps = 20;
      video.cfg = 2.5;
      video.shift = 8;
      video.width = 640;
      video.height = 480;
      video.numFrames = framesFor(5);
      logln(`preset: SVI 5s I2V · ${m.label} · steps 20 · CFG 2.5 · add a start image → Load → Generate`);
      if (!svi) logln("⚠ no SVI HIGH LoRA found — install SVI v2 PRO high/low for this preset");
      logln(loaded && video.loadedPath === m.path
        ? "✓ model already loaded — add image + prompt → Generate"
        : "→ hit Load Model, add image, then Generate");
    }
  }
  function setLoraProfile(mode: "single" | "high_low") { video.loraProfile = mode; }
  function clampLoraSplit() {
    video.loraSplitStep = Math.max(1, Math.min(Math.round(video.loraSplitStep), Math.max(1, video.steps - 1)));
  }
  function applyLightx2vRecipe() {
    const candidate =
      video.loras.find((l) => /lightx2v/i.test(l.label)) ??
      video.loras.find((l) => /highnoise/i.test(l.label));
    if (!video.loraPath && candidate) video.loraPath = candidate.path;
    video.steps = 8;
    video.cfg = 1;
    video.scheduler = "euler_beta";
    video.shift = 8;
    video.loraProfile = "high_low";
    video.loraHighStrength = 2.2;
    video.loraLowStrength = 0.8;
    video.loraSplitStep = 4;
    logln("preset: LightX2V · 8 steps (4/4) · CFG 1 · Euler Beta · shift 8 · LoRA 2.2/0.8");
  }

  function applySVIv2Recipe() {
    // SVI v2 PRO High + Low for Wan2.2 I2V-A14B — best for long coherent videos + excellent segment transitions.
    // Lighting and pop can be dull with SVI alone at very low steps/CFG.
    // Recommended starting point for better lighting/contrast: higher steps, moderate CFG, good shift.
    const high = video.loras.find((l) => /SVI.*HIGH| SVI_v2.*HIGH/i.test(l.label)) ||
                 video.loras.find((l) => /SVI_v2_PRO.*HIGH/i.test(l.label)) ||
                 video.loras.find((l) => l.label.toLowerCase().includes('svi') && l.label.toLowerCase().includes('high'));
    if (high) {
      video.loraPath = high.path;
    }
    video.loraProfile = "high_low";
    video.loraHighStrength = 1.0;
    video.loraLowStrength = 1.0;
    video.loraSplitStep = 10;  // reasonable switch point for SVI high/low noise phases on ~20+ step gens; adjust in advanced if needed

    // Better defaults for lighting and quality with SVI (not the ultra-low 4-step Lightning numbers)
    video.steps = Math.max(video.steps, 20);
    video.cfg = Math.max(video.cfg, 2.5);
    video.scheduler = "auto";
    video.shift = 8;

    // Try to auto-select a suitable I2V 14B model if none or if a T2V one is selected
    if (video.models && video.models.length > 0) {
      const current = (video.modelPath || '').toLowerCase();
      if (!current.includes('i2v') || current.includes('t2v')) {
        const i2v = video.models.find((m: any) => (m.path || m.label || '').toLowerCase().includes('i2v') && (m.path || m.label || '').toLowerCase().includes('14b'));
        if (i2v) {
          video.modelPath = i2v.path;
          logln(`auto: switched to I2V model for SVI: ${i2v.label || i2v.path}`);
        }
      }
    }

    // SVI alone can mute lighting; pairing with anatomy lock defaults helps the common
    // "vagina looks like a mouth" failure mode without needing a custom LoRA.
    video.negPrompt = withAnatomyNeg(video.negPrompt, "pussy vagina");
    logln("preset: SVI v2 PRO (I2V-A14B) High/Low @1.0 — steps 20+ / cfg 2.5+ / shift 8 for better lighting & contrast. Use with I2V model. Anatomy lock negatives applied. Add Lightning combo if you want more pop/motion (see SVI card).");
  }
  // One-click SD/HD. 720p rounds to 704 because Wan requires multiples of 32 (1280×720
  // would be silently adjusted to 1280×704 anyway) — set the real value so there's no
  // surprise. On this 16 GB display GPU, HD 14B uses 49f chunks; 81f HD still OOMs.
  function setRes(w: number, h: number) {
    if (video.resLocked) return;
    video.width = w; video.height = h;
    adjustResolutionForLength();
    noteNativeSegmentLimit();
  }

  // Clip length in seconds → frame count. Wan needs 4n+1 frames, so snap; depends on fps.
  // No hard cap anymore — user can target 30s+ (subject to VRAM/time/model limits).
  function framesFor(sec: number) {
    let f = Math.round(sec * video.fps);
    f = Math.round((f - 1) / 4) * 4 + 1;        // nearest 4n+1
    return Math.max(9, f);
  }
  function clipSecondsFromFrames(frames = video.numFrames, fps = video.fps) {
    return Math.max(1, Math.round((Math.max(frames, 1) - 1) / Math.max(fps, 1)));
  }
  function isWan14bHd() {
    return is14bSelected && !video.resLocked && video.width * video.height >= 500_000;
  }
  function isTi2v5bHd() {
    return isTi2v5bSelected && !video.resLocked && video.width * video.height >= 500_000;
  }
  function nativeSegmentFrames() {
    if (isTi2v5bSelected) return 121;
    // 720p 14B on this 16 GB display GPU does not fit the old 81f single pass.
    // 49f is the proven native-quality HD unit; SD can still use ~5s / 81f chunks.
    return isWan14bHd() ? 49 : 81;
  }
  let _lengthNoteTimer: ReturnType<typeof setTimeout> | null = null;
  function noteNativeSegmentLimit(immediate = false) {
    const run = () => {
      if (!isWanVideoSelected || video.fps < (isTi2v5bSelected ? 24 : 16) || video.numFrames <= nativeSegmentFrames()) return;
      if (canTailCondition) {
        logln(`${isTi2v5bSelected ? "TI2V-5B" : isWan14bHd() ? "HD native" : "Native"} long: ${video.numFrames}f will chain as ${nativeSegmentFrames()}f tail-conditioned chunks.`);
      } else {
        // Pure T2V-A14B: one prompt, backend low-VRAM chunked dual-expert. Keep SD res —
        // chunking is the VRAM strategy, not downscaling to 512×288.
        logln(
          `Pure T2V long: ${video.numFrames}f (~${clipSecondsFromFrames()}s) @ ${video.width}×${video.height} → one prompt, ` +
          `backend chunked dual-expert (Lightning 4-step = fast path). Refine after for quality.`
        );
      }
    };
    if (immediate) {
      if (_lengthNoteTimer) { clearTimeout(_lengthNoteTimer); _lengthNoteTimer = null; }
      run();
      return;
    }
    // Slider fires every pixel — debounce so we don't spam the log (and don't thrash res).
    if (_lengthNoteTimer) clearTimeout(_lengthNoteTimer);
    _lengthNoteTimer = setTimeout(run, 200);
  }
  function isNativeLongClip() {
    const nativeFps = isTi2v5bSelected ? 24 : 16;
    return isWanVideoSelected && video.fps >= nativeFps && video.numFrames > nativeSegmentFrames();
  }
  /** Pure T2V long (the path that actually delivered single-prompt 30s). */
  function isPureT2vLongClip() {
    return isNativeLongClip() && !canTailCondition;
  }
  function setSeconds(sec: number, fromSlider = false) {
    if (is14bSelected && video.fps < 16) {
      video.fps = 16;
      logln("native timing restored: 16 FPS. Use the Keyframes buttons only when you explicitly want slow/keyframe mode.");
    } else if (isTi2v5bSelected && video.fps < 24) {
      video.fps = 24;
      logln("native timing restored: TI2V-5B official timing is 24 FPS.");
    }
    video.numFrames = framesFor(sec);
    adjustResolutionForLength();
    noteNativeSegmentLimit(!fromSlider);
  }

  // Derived current approximate seconds for the slider (reflects direct Frames changes too).
  const currentSec = $derived(clipSecondsFromFrames());

  // Pure T2V long already chunks denoise for VRAM — do NOT death-spiral res to 512×288
  // while the user drags to 30s (that cost an hour for a postage-stamp clip). Only nudge
  // oversized custom widths; leave SD 480p / 832×480 alone.
  function adjustResolutionForLength() {
    if (video.resLocked) return;
    if (isWan14bHd()) return;
    if (isTi2v5bSelected) return;
    // I2V AR long can still soft-cap very wide custom sizes; pure T2V keeps selected size.
    if (isPureT2vLongClip() || (!canTailCondition && is14bSelected)) return;
    const secs = clipSecondsFromFrames();
    if (secs <= 12) return;
    const targetMaxW = secs > 25 ? 640 : 768;
    if (video.width > targetMaxW) {
      const ratio = targetMaxW / video.width;
      const newH = Math.round(video.height * ratio / 16) * 16;
      video.width = targetMaxW;
      video.height = newH;
      logln(`⚠ Long I2V clip (${secs.toFixed(0)}s) — auto-lowered res to ${targetMaxW}×${newH} for 16GB headroom.`);
    }
  }

  // React to direct changes in numFrames (e.g. typing in Advanced Frames field)
  $effect(() => {
    if (video.numFrames > 0) adjustResolutionForLength();
  });

  // i2v: pick a still image, read it to base64 (strip the data: prefix for the daemon)
  function pickImage(e: Event) {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;
    const r = new FileReader();
    r.onload = () => {
      video.imageB64 = String(r.result).split(",")[1] || "";
      video.imageName = file.name;
    };
    r.readAsDataURL(file);
  }
  function clearImage() { video.imageB64 = ""; video.imageName = ""; }

  function frameMax() {
    return Math.max(0, (video.frames || video.numFrames || 1) - 1);
  }

  function clampFrame(n: number) {
    return Math.max(0, Math.min(Math.round(n || 0), frameMax()));
  }

  function videoEl() {
    return (document.getElementById("vplayer") as HTMLVideoElement | null)
      || (document.querySelector("video.vplayer") as HTMLVideoElement | null);
  }

  async function waitForVideoReady(vid: HTMLVideoElement) {
    if (vid.readyState >= 1 && vid.videoWidth > 0) return;
    await new Promise<void>((resolve) => {
      const done = () => { vid.removeEventListener("loadedmetadata", done); resolve(); };
      vid.addEventListener("loadedmetadata", done, { once: true });
      setTimeout(() => { vid.removeEventListener("loadedmetadata", done); resolve(); }, 1000);
    });
  }

  async function seekVideoFrame(frame: number) {
    const vid = videoEl();
    if (!vid) return null;
    await waitForVideoReady(vid);
    const idx = clampFrame(frame);
    const frameTime = idx / Math.max(video.fps || 1, 1);
    const endGuard = vid.duration && isFinite(vid.duration) ? Math.max(0, vid.duration - 0.04) : frameTime;
    const target = Math.min(frameTime, endGuard);
    if (Math.abs(vid.currentTime - target) > 0.015) {
      await new Promise<void>((resolve) => {
        const done = () => { vid.removeEventListener("seeked", done); resolve(); };
        vid.addEventListener("seeked", done, { once: true });
        vid.currentTime = target;
        setTimeout(() => { vid.removeEventListener("seeked", done); resolve(); }, 900);
      });
    }
    stationFrame = idx;
    return vid;
  }

  async function scrubFrame(frame: number) {
    await seekVideoFrame(frame);
    if (stationCapturedFrame !== stationFrame) {
      stationFrameB64 = "";
      stationFrameName = "";
    }
  }

  function syncFrameFromPlayer() {
    const vid = videoEl();
    if (!vid) return;
    stationFrame = clampFrame(Math.round(vid.currentTime * Math.max(video.fps || 1, 1)));
  }

  async function captureFrame(frame = stationFrame) {
    const vid = await seekVideoFrame(frame);
    if (!vid || !vid.videoWidth || !vid.videoHeight) throw new Error("video frame not ready");
    const canvas = document.createElement("canvas");
    canvas.width = vid.videoWidth;
    canvas.height = vid.videoHeight;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("canvas unavailable");
    ctx.drawImage(vid, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/png");
    stationFrameB64 = dataUrl.split(",")[1] || "";
    stationFrameName = `frame-${String(stationFrame).padStart(4, "0")}.png`;
    stationCapturedFrame = stationFrame;
    logln(`✓ captured ${stationFrameName}`);
    return stationFrameB64;
  }

  async function captureCurrentFrame() {
    syncFrameFromPlayer();
    await captureFrame(stationFrame).catch((e) => logln(`✗ frame capture failed: ${e}`));
  }

  async function saveSelectedFrame() {
    const b64 = stationCapturedFrame === stationFrame && stationFrameB64
      ? stationFrameB64
      : await captureFrame(stationFrame);
    const path = await save({ filters: [{ name: "PNG", extensions: ["png"] }], defaultPath: stationFrameName || "frame.png" }).catch(() => null);
    if (path) {
      await T.writeBinaryB64(path, b64).catch((e) => logln(`✗ frame save failed: ${e}`));
      logln(`✓ saved ${stationFrameName || "frame.png"}`);
    }
  }

  async function useSelectedFrameAsSeed(anchor = false) {
    const b64 = stationCapturedFrame === stationFrame && stationFrameB64
      ? stationFrameB64
      : await captureFrame(stationFrame);
    video.imageB64 = b64;
    video.imageName = anchor ? `anchor-${stationFrame}.png` : (stationFrameName || `frame-${stationFrame}.png`);
    if (anchor) {
      stationAnchorFrame = stationFrame;
      if (video.prompt && !/same character|same face|same outfit|visual anchor/i.test(video.prompt)) {
        video.prompt = video.prompt.trim() + ", same character, same face, same outfit, consistent visual anchor";
      }
      logln(`✓ frame ${stationFrame} set as visual anchor / i2v seed`);
    } else {
      logln(`✓ frame ${stationFrame} set as image seed`);
    }
  }

  function setPlaybackRate(rate: number) {
    stationPlaybackRate = rate;
    const vid = videoEl();
    if (vid) vid.playbackRate = rate;
  }

  function setKeyframeFps(fps: number) {
    const seconds = clipSecondsFromFrames(video.numFrames || video.frames || 49, video.fps || 1);
    video.fps = fps;
    video.numFrames = framesFor(seconds);
    logln(`keyframe mode: ${fps} FPS native generation · ${video.numFrames} frames for ~${seconds}s`);
  }

  async function runStationPass(stages: string[], label: string, interpFactor = 2, fpsMultiplier = 1) {
    if (!video.resultB64 || video.enhancing) return;
    video.enhancing = true; video.error = ""; video.loadStatus = label;
    video.progress = 0; video.progressTotal = 1;
    logln(label);
    const unStatus = await listen<string>("vidload-progress", (e) => { video.loadStatus = e.payload; logln(e.payload); });
    const un = await listen<{ step: number; total: number }>("video_progress", (e) => {
      video.progress = e.payload.step; video.progressTotal = e.payload.total;
    });
    try {
      const r = await T.videoEnhance({
        video_b64: video.resultB64, fps: video.fps, stages, model_path: video.modelPath,
        prompt: video.prompt, neg_prompt: video.negPrompt, cfg_scale: video.cfg,
        refine_strength: video.refineStrength, refine_steps: video.refineSteps, interp_factor: interpFactor,
      });
      video.resultB64 = r.enhanced_b64; video.frames = r.frames; video.enhanced = true;
      if (fpsMultiplier !== 1) video.fps = Math.max(1, Math.round(video.fps * fpsMultiplier));
      video.loadedPath = "";
      stationFrame = 0; stationFrameB64 = ""; stationFrameName = ""; stationCapturedFrame = null;
      logln(`✓ ${label} complete · ${r.frames} frames in ${r.elapsed}s`);
    } catch (e) { video.error = compactGpuMessage(e); logln(`✗ ${e}`); }
    finally { un(); unStatus(); video.enhancing = false; video.loadStatus = ""; }
  }

  async function bakeSlowMo(factor = 2) {
    await runStationPass(["slow"], `slow motion: baking ${factor}× duration`, factor, 1);
  }

  async function smoothToFps(targetFps = 16) {
    const factor = Math.max(2, Math.round(targetFps / Math.max(video.fps || 1, 1)));
    await runStationPass(["interpolate"], `smooth: ${video.fps} FPS → ~${video.fps * factor} FPS`, factor, factor);
  }

  async function load() {
    if (!video.modelPath || video.loading) return;
    video.loading = true; video.error = ""; video.loadStatus = "Starting…";
    logln(`▶ loading ${video.models.find(m => m.path === video.modelPath)?.label ?? "model"}…`);
    const un = await listen<string>("vidload-progress", (e) => { video.loadStatus = e.payload; logln(e.payload); });
    try {
      // One GPU, one model. Drop the chat LLM first so its VRAM is gone before the video
      // model loads — otherwise HD generation OOMs on top of a resident chat server. (The
      // backend also kills stray servers and waits for the memory, but unloading here keeps
      // the chat engine's own state clean so it reloads correctly next time.)
      await T.unloadModel().catch(() => null);
      const loadLoraStrength = video.loraProfile === "high_low" ? video.loraHighStrength : video.loraStrength;
      await T.videoLoad(video.modelPath, video.loraPath, loadLoraStrength, video.numFrames, video.qualityMode ? "quality" : "fast");
      video.loadedPath = video.modelPath; video.loadStatus = "";
      logln(video.loraPath ? `✓ model ready (+ LoRA @ ${loadLoraStrength})` : "✓ model ready");
    } catch (e) { video.error = compactGpuMessage(e); logln(`✗ load failed: ${e}`); }
    finally { un(); video.loading = false; }
  }

  async function unload() {
    await T.videoUnload().catch(() => null);
    video.loadedPath = "";
    logln("✓ unloaded · VRAM freed");
  }

  async function generate() {
    // Two long paths:
    //  1) I2V/SVI  — UI autoregressive: capture last frame → next chunk (best identity)
    //  2) Pure T2V — single prompt, full frame count; backend dual-expert low-VRAM
    //     chunks denoise (~121f) with first-frame seam conditioning. This is the path
    //     that already produced single-prompt ~30s clips. Do NOT refuse it.
    if (isNativeLongClip() && canTailCondition) {
      await generateAutoregressiveLong();
      return;
    }
    if (isPureT2vLongClip()) {
      ensureLongClipQualitySettings();
      if (video.anatomyLock && looksExplicitAnatomy(video.prompt)) {
        logln("anatomy lock: pure T2V long — pos/neg anti mouth↔vulva / fused-genital guards on the single prompt");
      }
      logln(
        `pure T2V long: ${video.numFrames}f @ ${video.fps} FPS · one prompt · ` +
        `backend will chunk denoise (seam-conditioned) — no starting image required`
      );
      if (isWan14bHd()) {
        logln("⚠ HD pure T2V long is VRAM-tight on 16GB; if it OOMs drop to SD 480p or shorten the clip.");
      }
    }
    await generateSingleSegment();
  }

  async function generateSingleSegment() {
    if (!loaded || video.generating || !video.prompt.trim()) return;
    if (isWanVideoSelected) {
      const requestedFrames = Math.max(1, Math.round(video.numFrames));
      const supportedFrames = Math.max(9, Math.floor((requestedFrames - 1) / 4) * 4 + 1);
      if (supportedFrames !== requestedFrames) {
        video.numFrames = supportedFrames;
        logln(`Wan frame count: ${requestedFrames} snapped to supported ${supportedFrames} (4n+1)`);
      }
    }
    const prevBase = appendFromB64;
    const isExtension = !!prevBase;

    video.generating = true; video.error = ""; video.enhanced = false;
    video.previewB64 = ""; video.previewStep = 0; video.previewFrames = [];
    if (!isExtension) {
      video.resultB64 = "";
    } else {
      // Keep the prior full clip visible while we generate + stitch the extension.
      video.loadStatus = "Generating extension…";
    }
    video.progress = 0; video.progressTotal = video.steps;
    if (!isExtension) video.loadStatus = "Preparing…";
    clampLoraSplit();
    const sched = video.scheduler === "euler_beta" ? ` · Euler Beta s${video.shift}` : ` · model scheduler s${video.shift}`;
    const cache = video.denoiseCache === "balanced" ? " · cached denoise" : " · exact denoise";
    const split = video.loraPath && video.loraProfile === "high_low"
      ? ` · LoRA ${video.loraHighStrength}/${video.loraLowStrength} (${video.loraSplitStep}/${Math.max(video.steps - video.loraSplitStep, 0)})`
      : "";
    const wanSafeChunkFrames = 121;
    const wanHighResPixels = 500_000;
    const isWanHighRes = video.width * video.height >= wanHighResPixels;
    // Pure T2V long + any 14B gen over the safe chunk / 20s mark forces backend low-VRAM
    // chunking (this is the single-prompt 30s path).
    const pureT2vLong = !isExtension && isPureT2vLongClip();
    const useLowVram =
      video.lowVramMode ||
      pureT2vLong ||
      (is14bSelected && (video.numFrames > wanSafeChunkFrames || isWanHighRes || video.numFrames / Math.max(video.fps, 1) >= 20)) ||
      (isTi2v5bSelected && (video.numFrames > 121 || isTi2v5bHd()));
    logln(isExtension ? `➕ extend · appending ${video.numFrames}f to prior clip` : `🎬 generate · ${video.width}×${video.height} · ${video.numFrames}f · ${video.steps} steps${sched}${cache}${split}`);
    if (useLowVram) {
      if (pureT2vLong || video.numFrames > wanSafeChunkFrames) {
        logln(
          `low-VRAM T2V chunking: ${video.numFrames}f split above ~${wanSafeChunkFrames}f with seam-frame continuity ` +
          `(single prompt — not I2V AR)`
        );
      } else if (isWanHighRes) {
        logln(`low-VRAM mode active: ${video.width}×${video.height} needs high-res ${isTi2v5bSelected ? "5B" : "14B"} headroom`);
      } else {
        logln("low-VRAM mode active for this generation");
      }
    }
    if (isWan14bHd() && video.fps < 16) {
      logln(`manual keyframe path: ${video.fps} FPS native · ${video.numFrames} frames. Temporal quality is intentionally traded for duration.`);
    }
    // Phase messages (text-encoder load, then denoise) come over vidload-progress;
    // step progress comes over video_progress. The first real step clears the phase text.
    const unStatus = await listen<string>("vidload-progress", (e) => { video.loadStatus = e.payload; logln(e.payload); });
    const un = await listen<{ step: number; total: number; step_seconds?: number; elapsed_seconds?: number }>("video_progress", (e) => {
      video.progress = e.payload.step; video.progressTotal = e.payload.total;
      video.loadStatus = "";
      const timing = typeof e.payload.step_seconds === "number"
        ? ` · ${e.payload.step_seconds.toFixed(1)}s step · ${Math.round(e.payload.elapsed_seconds ?? 0)}s elapsed`
        : "";
      logln(`step ${e.payload.step}/${e.payload.total}${timing}`);
    });
    const unPreview = await listen<{
      base64_jpeg: string;
      step: number;
      total: number;
      frames: number[];
      decode_seconds?: number;
    }>("video-preview", (e) => {
      video.previewB64 = e.payload.base64_jpeg;
      video.previewStep = e.payload.step;
      video.previewFrames = e.payload.frames;
      const timing = typeof e.payload.decode_seconds === "number"
        ? ` · ${e.payload.decode_seconds.toFixed(2)}s decode`
        : "";
      logln(`preview · step ${e.payload.step}/${e.payload.total} · ${e.payload.frames.length} frames${timing}`);
    });
    // Anatomy + long-clip locks applied at request time so the textarea stays what the user typed
    // (except AR/storyboard which temporarily rewrite video.prompt per chunk).
    const genPrompt = withAnatomyPos(video.prompt, isExtension || isNativeLongClip() || pureT2vLong || !!prevBase);
    const genNeg = withAnatomyNeg(video.negPrompt, video.prompt);
    if (video.anatomyLock && looksExplicitAnatomy(video.prompt) && genPrompt !== video.prompt) {
      logln("anatomy lock: positive vulva descriptors injected (anti mouth/lips/fused-genital collapse)");
    }
    try {
      const r = await T.videoGenerate({
        prompt: genPrompt, neg_prompt: genNeg, model_path: video.modelPath,
        num_frames: video.numFrames, steps: video.steps, cfg_scale: video.cfg,
        scheduler: video.scheduler, shift: video.shift,
        lora_profile: video.loraProfile,
        lora_strength_high: video.loraHighStrength,
        lora_strength_low: video.loraLowStrength,
        lora_split_step: video.loraSplitStep,
        width: video.width, height: video.height, fps: video.fps, seed: video.seed,
        image_b64: video.imageB64 || undefined,
        previous_video_b64: prevBase || undefined,
        force_seam_blend: !!(prevBase && video.storyboardPrompts && video.storyboardPrompts.some(p => p && p.trim())),
        low_vram: useLowVram,
        block_offload: video.blockOffload,
        denoise_cache: video.denoiseCache,
        cache_threshold: 0.10,
        preview: video.livePreview,
        preview_every: 5,
        preview_max_width: 256,
      });
      video.resultB64 = r.base64_mp4; video.frames = r.frames; video.elapsed = r.elapsed;
      if (isExtension) {
        logln(`✓ extended + stitched · now ${r.frames} frames total · ${r.elapsed}s for segment`);
        logln('   Tip: for long chains, keep the prompt locked on identity ("stable face, same clothes, same lighting") and use the ✨ Refine bf16 v2v pass afterward to pull the combined clip back toward the original description.');
      } else if (pureT2vLong) {
        logln(`✓ pure T2V long done · ${r.frames} frames in ${r.elapsed}s (backend-chunked single prompt)`);
        logln("Tip: run ✨ Quality Pass (Refine) — best lever for anatomy + lighting drift across T2V seam chunks.");
      } else {
        logln(`✓ done · ${r.frames} frames in ${r.elapsed}s`);
      }
      logln('Click "Save MP4" to save the video (nothing is auto-saved to disk).');
    } catch (e) { video.error = compactGpuMessage(e); logln(`✗ ${e}`); }
    finally {
      un(); unStatus(); unPreview(); video.generating = false; video.loadStatus = "";
      appendFromB64 = "";  // always clear after the round
      // After a successful extension, clear the temporary "last frame" seed image chip.
      // The player now shows the full stitched clip; hit Extend again to chain further.
      if (isExtension) {
        video.imageB64 = "";
        video.imageName = "";
      }
    }
  }

  function continuationPrompt(base: string, longClip = false) {
    const text = base.trim();
    const explicit = looksExplicitAnatomy(text);
    let cont =
      ", continuing directly from the previous frame, exact same character identity, face, body proportions, outfit, camera, lighting and exposure, seamless motion continuity, no reset, no jump cut, no identity drift";
    if (longClip) {
      cont += ", " + QUALITY_LONG_POS;
    }
    if (video.anatomyLock && explicit) {
      cont +=
        ", exact same genitals as previous frame, anatomically correct vulva with distinct labia (not a mouth, not lips, no teeth, no penis, no fused sex organs), no anatomy drift across the cut";
    }
    if (/continu|seamless|previous|from the last|directly from|maintain.*continuity|exact same shot/i.test(text)) {
      // Still inject anatomy/quality locks if the user wrote continuity language but omitted them.
      let p = text;
      if (longClip && !/high-fidelity|no quality drop/i.test(p)) p = `${p}, ${QUALITY_LONG_POS}`;
      if (video.anatomyLock && explicit && !/anatomically correct|not a mouth|labia majora/i.test(p)) {
        p = `${p}, ${ANATOMY_POS}, exact same genitals as previous frame, no anatomy drift`;
      }
      return p;
    }
    return `${text}${cont}`;
  }

  async function generateAutoregressiveLong() {
    if (!loaded || video.generating || !video.prompt.trim()) return;
    const startsFromText = isTi2v5bSelected && !video.imageB64;
    if (!video.imageB64 && !startsFromText) {
      video.error = "Native autoregressive long mode needs a starting image on this I2V/SVI model. Add an image first, then Generate.";
      logln(`✗ ${video.error}`);
      return;
    }

    const targetFrames = video.numFrames;
    const originalPrompt = video.prompt;
    const originalFrames = video.numFrames;
    const originalFps = video.fps;
    const originalSteps = video.steps;
    const originalCfg = video.cfg;
    const originalShift = video.shift;
    const segmentFrames = nativeSegmentFrames();
    const stride = segmentFrames - 1;
    const chunks = Math.ceil(Math.max(targetFrames - 1, 1) / stride);
    let produced = 0;

    appendFromB64 = "";
    video.frames = 0;
    video.error = "";
    ensureLongClipQualitySettings();
    if (video.anatomyLock && looksExplicitAnatomy(originalPrompt)) {
      logln("anatomy lock: positive vulva descriptors + anti mouth/lips/fused-genital negatives on every chunk");
    }
    logln(
      `native autoregressive: ${targetFrames} frames @ ${video.fps} FPS as ${chunks}×${segmentFrames}f ` +
      (startsFromText ? "TI2V chunks (first T2V, then tail-image conditioned)" : "tail-conditioned chunks")
    );

    for (let i = 0; i < chunks; i++) {
      const remaining = Math.max(targetFrames - produced, 1);
      let segFrames = i === 0 ? Math.min(segmentFrames, remaining) : Math.min(segmentFrames, remaining + 1);
      segFrames = Math.max(9, Math.round((segFrames - 1) / 4) * 4 + 1);
      video.numFrames = segFrames;
      // Chunk 0 also gets anatomy/quality locks; later chunks get continuity language too.
      video.prompt = i === 0
        ? withAnatomyPos(originalPrompt, true)
        : withAnatomyPos(continuationPrompt(originalPrompt, true), true);

      if (i > 0) {
        await captureLastFrameForChain();
        if (!video.imageB64 || !appendFromB64) {
          video.error = "Could not capture the previous tail frame for the next autoregressive chunk.";
          logln(`✗ ${video.error}`);
          break;
        }
      }

      logln(`native chunk ${i + 1}/${chunks}: ${segFrames}f @ ${video.fps} FPS${
        i > 0 ? " · tail image conditioned" : startsFromText ? " · text-only opening" : " · starting image conditioned"
      }`);
      await generateSingleSegment();
      if (video.error) break;

      produced = i === 0 ? segFrames : produced + Math.max(segFrames - 1, 0);
      logln(`native autoregressive progress: ${Math.min(produced, targetFrames)}/${targetFrames} unique frames`);
    }

    video.prompt = originalPrompt;
    video.numFrames = originalFrames;
    video.fps = originalFps;
    video.steps = originalSteps;
    video.cfg = originalCfg;
    video.shift = originalShift;
    appendFromB64 = "";
    if (!video.error && video.resultB64) {
      logln("✓ long clip assembled. Tip: run ✨ Quality Pass (Refine + Face) — refine re-reads your prompt and cleans anatomy/lighting drift across seams.");
      if (video.anatomyLock && looksExplicitAnatomy(originalPrompt) && video.doRefine) {
        logln("anatomy tip: Refine uses the same anatomy lock language so mouth↔vulva mistakes get corrected without re-rolling the whole 30s.");
      }
    }
  }

  async function enhance() {
    if (!video.resultB64 || video.enhancing) return;
    const stages: string[] = [];
    if (video.doRefine) stages.push("refine");
    if (video.doFace) stages.push("face");
    if (video.doUpscale) stages.push("upscale");
    if (video.doInterpolate) stages.push("interpolate");
    if (stages.length === 0) return;
    video.enhancing = true; video.error = ""; video.loadStatus = "Preparing…";
    video.progress = 0; video.progressTotal = 1;
    logln(`✨ quality pass: ${stages.join(" → ")}`);
    // Long / explicit clips: push anatomy + fidelity into the v2v refine prompt so
    // the pass actually fixes mouth↔vulva and fused-genital artifacts.
    const refinePrompt = withAnatomyPos(video.prompt, video.frames > 100);
    const refineNeg = withAnatomyNeg(video.negPrompt, video.prompt);
    if (refinePrompt !== video.prompt || refineNeg !== video.negPrompt) {
      logln("quality pass: anatomy/fidelity language injected into refine embeds");
    }
    // Slightly stronger refine on long clips (still motion-preserving).
    const longClip = video.frames > 100;
    const strength = longClip
      ? Math.min(0.55, Math.max(video.refineStrength, 0.4))
      : video.refineStrength;
    const steps = longClip
      ? Math.max(video.refineSteps, 24)
      : video.refineSteps;
    if (longClip && (strength !== video.refineStrength || steps !== video.refineSteps)) {
      logln(`quality pass: long-clip refine bumped to strength ${strength} / ${steps} steps`);
    }
    const unStatus = await listen<string>("vidload-progress", (e) => { video.loadStatus = e.payload; logln(e.payload); });
    const un = await listen<{ step: number; total: number }>("video_progress", (e) => {
      video.progress = e.payload.step; video.progressTotal = e.payload.total;
    });
    try {
      const r = await T.videoEnhance({
        video_b64: video.resultB64, fps: video.fps, stages, model_path: video.modelPath,
        prompt: refinePrompt, neg_prompt: refineNeg, cfg_scale: video.cfg,
        refine_strength: strength, refine_steps: steps, interp_factor: 2,
      });
      video.resultB64 = r.enhanced_b64; video.frames = r.frames; video.enhanced = true;
      video.loadedPath = "";   // the generator was unloaded to free VRAM for the pass
      logln(`✓ enhanced · ${r.width}×${r.height} · ${r.frames} frames in ${r.elapsed}s`);
    } catch (e) { video.error = compactGpuMessage(e); logln(`✗ ${e}`); }
    finally { un(); unStatus(); video.enhancing = false; video.loadStatus = ""; }
  }

  async function saveMp4() {
    if (!video.resultB64) return;
    const path = await save({ filters: [{ name: "MP4", extensions: ["mp4"] }], defaultPath: "clip.mp4" }).catch(() => null);
    if (path) await T.writeBinaryB64(path, video.resultB64).catch(() => {});
  }

  // Extend: capture the final frame of the current clip and feed it as an image-to-video
  // seed. Lets you chain short high-quality generations into longer clips without one giant
  // 100+ frame run. The target per-segment gen time (with quality bf16) is 30s–60s.
  async function extendClip() {
    if (!video.resultB64) return;
    await captureLastFrameForChain();
    if (video.prompt && !/continuation|continue|extend|next|smooth follow/i.test(video.prompt)) {
      video.prompt = video.prompt.trim() + ", continuation of the motion, seamless from previous frame";
    }
    logln('✓ captured last frame for extend — will append to prior clip. Tweak prompt and hit 🖼 Animate image (or Generate)');
  }

  // Reusable: capture last frame of current resultB64 into imageB64 + set appendFromB64 for chaining.
  // Used by both manual Extend and auto storyboard.
  async function captureLastFrameForChain() {
    if (!video.resultB64) return;
    const vid = (document.getElementById('vplayer') as HTMLVideoElement | null) || document.querySelector('video.vplayer') as HTMLVideoElement | null;
    if (!vid) {
      appendFromB64 = video.resultB64 || "";
      logln('⚠ chaining with previous result (no video player available for frame capture)');
      return;
    }
    const wasPaused = vid.paused;
    const prevTime = vid.currentTime;
    try {
      if (vid.duration && isFinite(vid.duration)) {
        vid.currentTime = Math.max(0, vid.duration - 0.08);
      } else {
        vid.currentTime = 999999;
      }
      await new Promise<void>((resolve) => {
        const done = () => { vid.removeEventListener('seeked', done); resolve(); };
        vid.addEventListener('seeked', done, { once: true });
        setTimeout(() => { vid.removeEventListener('seeked', done); resolve(); }, 800);
      });
      const w = vid.videoWidth || 832;
      const h = vid.videoHeight || 480;
      const canvas = document.createElement('canvas');
      canvas.width = w; canvas.height = h;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (ctx) {
        ctx.drawImage(vid, 0, 0, w, h);
        const dataUrl = canvas.toDataURL('image/png');
        video.imageB64 = dataUrl.split(',')[1] || '';
        video.imageName = 'chain-frame.png';
      }
      appendFromB64 = video.resultB64 || "";
      if (!wasPaused) vid.play().catch(() => {});
    } catch (e) {
      logln('⚠ frame capture for chain failed: ' + e + ' (will still stitch)');
      appendFromB64 = video.resultB64 || "";
    } finally {
      try { vid.currentTime = prevTime; } catch {}
    }
  }

  // Auto-generate storyboardPrompts as sequential native segments. I2V-capable models
  // get real tail-frame conditioning; pure T2V only gets prompt continuity + seam blend.
  async function generateStoryboard() {
    const prompts = video.storyboardPrompts.map(p => (p || '').trim()).filter(Boolean);
    if (prompts.length === 0) {
      await generateSingleSegment();
      return;
    }
    logln(`📖 Storyboard (${isTi2v5bSelected ? "TI2V-5B 480p" : "14B SVI/T2V"}): ${prompts.length} segments — single load, coherence across clips + seam blend`);
    logln(`Pre-test VRAM clear check: ensuring clean state (<500MiB) for ${isTi2v5bSelected ? "TI2V-5B" : "14B"} quality run.`);
    video.resultB64 = "";
    video.error = "";
    video.frames = 0;
    appendFromB64 = "";
    // Do NOT clear video.imageB64 here.
    // If the user added an initial image before clicking Storyboard (common for I2V),
    // we must preserve it for segment 1. For segments 2+, captureLastFrameForChain()
    // will overwrite imageB64 with the previous segment's last frame.
    if (video.imageB64) {
      logln("Using your added image as the starting frame for segment 1.");
    }

    const isPureT2V = !canTailCondition;
    const originalFrames = video.numFrames;
    if (is14bSelected && video.fps >= 16 && video.numFrames > nativeSegmentFrames()) {
      video.numFrames = nativeSegmentFrames();
      logln(`Storyboard uses native ${nativeSegmentFrames()}f segments; add more storyboard boxes for longer video.`);
    }

    ensureLongClipQualitySettings();
    if (video.anatomyLock && prompts.some(looksExplicitAnatomy)) {
      logln("anatomy lock: storyboard segments get vulva-correct descriptors + anti mouth/lips/fused-genital negatives");
    }

    for (let i = 0; i < prompts.length; i++) {
      // withAnatomyPos is applied again inside generateSingleSegment — here we only add continuity.
      video.prompt = i === 0
        ? prompts[i]
        : continuationPrompt(prompts[i], true);
      if (i > 0) {
        appendFromB64 = video.resultB64 || "";
        if (!isPureT2V) {
          // Only attempt image capture/seed if the model can use it (I2V or TI2V).
          // For pure T2V A14B we skip to avoid the "T2V does not accept image" rejection.
          await captureLastFrameForChain();
        } else {
          logln(`ℹ Pure T2V-A14B segment ${i+1} (no image seed allowed). Prompt + force-seam + blend. 14B only.`);
        }
      }
      await generateSingleSegment();
      if (video.error) {
        logln(`✗ storyboard halted at segment ${i + 1}`);
        break;
      }
      logln(`✓ segment ${i + 1}/${prompts.length} complete`);
      if (i < prompts.length - 1 && video.resultB64) {
        appendFromB64 = video.resultB64 || "";
      }
    }
    video.numFrames = originalFrames;
    logln('📖 Storyboard finished — full stitched video ready');
    if (!video.error && video.resultB64) {
      logln("Tip: run ✨ Quality Pass (Refine) — cleans anatomy/lighting drift across storyboard seams.");
    }
    logln('Click "Save MP4" to save the final video (nothing auto-saved without your explicit action).');
  }
</script>

<div class="ig-layout">
  <div class="ig-sidebar">
    <div class="vrow" style="margin-bottom:10px"><span>Setup</span>
      <div class="vsize">
        <button class="vsize-btn" class:on={!advanced} onclick={() => setAdvanced(false)} title="Just pick a model and generate — everything else is chosen for you">Simple</button>
        <button class="vsize-btn" class:on={advanced} onclick={() => setAdvanced(true)} title="Show every parameter (LoRA, steps, CFG, scheduler, recipes…)">Advanced</button>
      </div>
    </div>
    <div class="vlabel">Model</div>
    <select class="vsel" bind:value={video.modelPath} onchange={autoSet}>
      <option value="">— select a video model —</option>
      {#each video.models as m}<option value={m.path}>{m.label}</option>{/each}
    </select>
    <button class="vbtn-ghost" onclick={refresh}>⟳ Refresh</button>
    <button class="load-btn" onclick={load} disabled={!video.modelPath || video.loading}>
      {video.loading ? `⟳ ${video.loadStatus || "Loading…"}` : loaded ? "✓ Loaded · Reload" : "▶ Load Model"}
    </button>
    {#if loaded && !video.loading}
      <button class="vbtn-ghost" onclick={unload} disabled={video.generating || video.enhancing}>⏏ Unload · Free VRAM</button>
    {/if}
    {#if video.models.length === 0}
      <div class="vhint">No video models found. Drop a diffusers Wan/LTX folder into your models dir, then Refresh.</div>
    {/if}

    {#if !advanced}
      <div class="vlabel" style="margin-top:14px">Presets</div>
      <div class="vsize" style="flex-wrap:wrap; gap:4px;">
        <button class="vsize-btn" class:on={activeSimplePreset === "t2v_30s_fast"}
          onclick={() => applySimplePreset("t2v_30s_fast")}
          title="Wan2.2 T2V-A14B + Lightning 4-step · 832×480 · 30s · anatomy lock. Load → Generate.">
          30s Fast T2V
        </button>
        <button class="vsize-btn" class:on={activeSimplePreset === "t2v_5s_fast"}
          onclick={() => applySimplePreset("t2v_5s_fast")}
          title="FastWan-14B (fp32, sharp) · 832×480 · ~5s · 4-step. Sharper than the A14B path for short clips. Load → Generate.">
          5s Fast T2V
        </button>
        <button class="vsize-btn" class:on={activeSimplePreset === "t2v_hd_3s"}
          onclick={() => applySimplePreset("t2v_hd_3s")}
          title="FastWan-14B (full fp32, no 720p bf16 softening) · 1280×704 · 49f (~3s) · 4-step. Native 720p tops out ~3-4s on 16 GB; for 5s HD use HD 5s (max) or gen 480p then Enhance ×2. Load → Generate.">
          HD 3s T2V
        </button>
        <button class="vsize-btn" class:on={activeSimplePreset === "t2v_hd_5s_max"}
          onclick={() => applySimplePreset("t2v_hd_5s_max")}
          title="Freeze-proof HD 5s: FastWan-14B renders native 5s@480p (fits the 16 GB card), then click Enhance for the 2× ESRGAN upscale to ~960p HD. Avoids the RAM-parking route that froze the PC. Load → Generate → Enhance.">
          HD 5s (safe)
        </button>
        <button class="vsize-btn" class:on={activeSimplePreset === "ti2v_5b_480p_hq"}
          onclick={() => applySimplePreset("ti2v_5b_480p_hq")}
          title="Wan2.2 TI2V-5B · official 121f @ 24 FPS · 50 steps · CFG 5 · 832×480. Quality-first 5s base for the refiner/upscaler. Load → Generate → Enhance.">
          5B HQ 5s
        </button>
        <button class="vsize-btn" class:on={activeSimplePreset === "ti2v_5b_480p_10s"}
          onclick={() => applySimplePreset("ti2v_5b_480p_10s")}
          title="Wan2.2 TI2V-5B · 10s as 2 native 121f chunks, first from text then tail-frame conditioned · 832×480. Load → Generate → Enhance.">
          5B HQ 10s
        </button>
        <button class="vsize-btn" class:on={activeSimplePreset === "i2v_svi_5s"}
          onclick={() => applySimplePreset("i2v_svi_5s")}
          title="I2V-A14B + SVI v2 PRO · 640×480 · 5s. Needs a start image. Load → Generate.">
          SVI 5s I2V
        </button>
      </div>
      <div class="vhint">
        One click sets model + LoRA + length + size. Then <strong>Load Model</strong> → paste prompt → <strong>Generate</strong>.
        {activeSimplePreset === "t2v_30s_fast" ? " Active: 30s Lightning (speed path)."
          : activeSimplePreset === "t2v_5s_fast" ? " Active: 5s Lightning smoke."
          : activeSimplePreset === "t2v_hd_3s" ? " Active: HD 3s Lightning."
          : activeSimplePreset === "ti2v_5b_480p_hq" ? " Active: 5B 480p quality baseline."
          : activeSimplePreset === "ti2v_5b_480p_10s" ? " Active: 5B 480p long test, chained 121f chunks."
          : activeSimplePreset === "i2v_svi_5s" ? " Active: SVI I2V — add an image first."
          : ""}
      </div>

      <!-- Simple mode: the only two choices that are actually taste, not tuning. -->
      <div class="vlabel" style="margin-top:14px">Clip</div>
      <div class="vrow"><span>Length</span>
        <div style="flex:1; display:flex; flex-direction:column; gap:2px;">
          <div style="display:flex; align-items:center; gap:8px;">
            <input type="range" min="1" max="120" step="1"
              value={currentSec}
              oninput={(e) => setSeconds(parseInt((e.target as HTMLInputElement).value), true)}
              style="flex:1; accent-color: var(--accent);" />
            <span style="font-family:var(--mono); font-size:11px; min-width:32px; text-align:right;">{currentSec}s</span>
          </div>
          <div class="vsize" style="margin-top:2px;">
            <button class="vsize-btn" class:on={video.numFrames === framesFor(2)} onclick={() => setSeconds(2)} title="~2 second clip">2s</button>
            <button class="vsize-btn" class:on={video.numFrames === framesFor(3)} onclick={() => setSeconds(3)} title="~3 second clip">3s</button>
            <button class="vsize-btn" class:on={video.numFrames === framesFor(5)} onclick={() => setSeconds(5)} title="~5 second native target: 5B uses 121f @24 FPS; 14B uses 16 FPS chunks">5s</button>
          </div>
          <div class="vsize" style="margin-top:2px;">
            <button class="vsize-btn" class:on={currentSec === 10} onclick={() => setSeconds(10)} title="~10s. TI2V-5B chains 121f @24 FPS chunks; A14B pure T2V uses backend chunked denoise; I2V uses tail-frame chunks.">10s</button>
            <button class="vsize-btn" class:on={currentSec === 20} onclick={() => setSeconds(20)} title="~20s. Long 5B/I2V routes chain tail-conditioned native chunks; pure A14B T2V uses single-prompt backend chunks.">20s</button>
            <button class="vsize-btn" class:on={currentSec === 30} onclick={() => setSeconds(30)} title="~30s. Best for 480p-first chained tests; 14B pure T2V keeps a single prompt, 5B starts from text then tail-conditions later chunks.">30s</button>
          </div>
        </div>
      </div>
      <div class="vrow"><span>Size</span>
        <div class="vsize">
          <button class="vsize-btn" class:on={!video.resLocked && video.width === 832 && video.height === 480}
                  disabled={video.resLocked} onclick={() => setRes(832, 480)}
                  title="832×480 — fastest and rock-solid stable">SD 480p</button>
          <button class="vsize-btn" class:on={!video.resLocked && video.width === 1280 && video.height === 704}
                  disabled={video.resLocked} onclick={() => setRes(1280, 704)}
                  title="1280×704 HD — 14B uses safe 49-frame chunks on 16GB. TI2V-5B can be tested short, but the stable path is 480p then Refine/Upscale.">HD 720p</button>
        </div>
      </div>
      <div class="vhint">
        Auto setup: {video.steps} steps · CFG {video.cfg}{video.loraPath ? ` · ${video.loras.find((l) => l.path === video.loraPath)?.label ?? "LoRA"}` : ""}.
        {isNativeLongClip() && canTailCondition
          ? `${isTi2v5bSelected ? "TI2V-5B long" : "I2V long"}: ${video.fps} FPS · tail-conditioned ${nativeSegmentFrames()}f chunks.`
          : isPureT2vLongClip()
            ? `Pure T2V long: one prompt · backend seam-chunked denoise (~${clipSecondsFromFrames()}s). Anatomy lock + Refine for quality.`
            : "Everything is picked for this model — switch to Advanced to override."}
      </div>
    {/if}

    <div class="vlabel" style="margin-top:14px">Denoise</div>
    <div class="vrow"><span>Execution</span>
      <div class="vsize">
        <button class="vsize-btn" class:on={video.denoiseCache === "off"}
          onclick={() => video.denoiseCache = "off"}
          title="Run every transformer block on every step">Exact</button>
        <button class="vsize-btn" class:on={video.denoiseCache === "balanced"}
          onclick={() => video.denoiseCache = "balanced"}
          title="Reuse similar block outputs between denoise steps. Leaves the LoRA slot free and may slightly change the result.">Cached</button>
      </div>
    </div>
    <div class="vhint">
      {video.denoiseCache === "balanced"
        ? "Native block cache is active. Your selected LoRA remains the only adapter."
        : "Exact runs every block and is the slow reference path."}
    </div>
    <label class="vrow" style="cursor:pointer">
      <span>Live latent preview</span>
      <input type="checkbox" bind:checked={video.livePreview} disabled={video.generating} />
    </label>

    {#if advanced}
    <div class="vlabel" style="margin-top:14px">LoRA <span style="font-weight:400;text-transform:none;color:var(--text3)">(optional, applied at load)</span></div>
    <select class="vsel" bind:value={video.loraPath} onchange={loraAutoTune} disabled={video.loading}>
      <option value="">— none —</option>
      {#each video.loras as l}<option value={l.path}>{l.label}</option>{/each}
    </select>
    {#if video.loraPath}
      <div class="vrow"><span>Mode</span>
        <div class="vsize">
          <button class="vsize-btn" class:on={video.loraProfile === "single"} onclick={() => setLoraProfile("single")}>Single</button>
          <button class="vsize-btn" class:on={video.loraProfile === "high_low"} onclick={() => setLoraProfile("high_low")}>High/Low</button>
        </div>
      </div>
      {#if video.loraProfile === "high_low"}
        <div class="vrow"><span>High</span><input type="number" bind:value={video.loraHighStrength} min="0" max="4" step="0.05" disabled={video.loading} /></div>
        <div class="vrow"><span>Low</span><input type="number" bind:value={video.loraLowStrength} min="0" max="4" step="0.05" disabled={video.loading} /></div>
        <div class="vrow"><span>Split</span><input type="number" bind:value={video.loraSplitStep} min="1" max={Math.max(video.steps - 1, 1)} step="1" onblur={clampLoraSplit} disabled={video.loading} /></div>
        <div class="vhint">Runs LoRA high strength for {video.loraSplitStep} steps, then low strength for {Math.max(video.steps - video.loraSplitStep, 0)}.</div>
      {:else}
        <div class="vrow"><span>Strength</span><input type="number" bind:value={video.loraStrength} min="0" max="4" step="0.05" disabled={video.loading} /></div>
      {/if}
      <button class="vbtn-ghost" onclick={applyLightx2vRecipe} disabled={video.loading}>LightX2V 8-step recipe</button>
      <button class="vbtn-ghost" onclick={applySVIv2Recipe} disabled={video.loading} title="SVI v2 PRO High+Low for Wan2.2-I2V-A14B. Excellent for long multi-segment videos and character coherence across storyboard clips. Load an I2V 14B model first. Optional Lightning for speed.">SVI v2 PRO (I2V long/coherent)</button>
      <div class="vhint">LoRA stays as a sidecar adapter on the cached 4-bit transformer. Re-Load after changing the LoRA file.</div>
    {/if}

    <div class="vlabel" style="margin-top:14px">Recipes</div>
    {#each recipes as r (r.name)}
      <div style="display:flex; gap:6px; align-items:center; margin-bottom:4px">
        <button class="vbtn-ghost" style="flex:1; text-align:left; margin:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap"
          onclick={() => applyRecipe(r)}
          title={`${video.models.find((m) => m.path === r.modelPath)?.label ?? r.modelPath} · ${r.steps} steps · CFG ${r.cfg} · ${r.width}×${r.height} · ${r.numFrames}f${r.loraPath ? " · +LoRA" : ""}`}>
          {r.name}
        </button>
        <button class="vauto" onclick={() => deleteRecipe(r.name)} title="Delete recipe">✕</button>
      </div>
    {/each}
    <div style="display:flex; gap:6px; align-items:center">
      <input type="text" placeholder="name this setup…" bind:value={recipeName}
        style="flex:1; min-width:0" onkeydown={(e) => e.key === "Enter" && saveRecipe()} />
      <button class="vauto" onclick={saveRecipe} title="Save the current model + LoRA + all params as a recipe">＋ Save</button>
    </div>
    <div class="vhint">Saves model + LoRA + every param. Click a recipe to restore it — then Load if the model or LoRA changed.</div>

    <div class="vlabel" style="margin-top:14px">Quality <span style="font-weight:400;text-transform:none;color:var(--text3)">(applied at load)</span></div>
    <label class="vrow" style="cursor:pointer">
      <span>bf16 transformer <span style="font-size:0.85em;color:var(--accent2,#e0a060)">· experimental</span></span>
      <input type="checkbox" bind:checked={video.qualityMode} disabled={video.loading || is14bSelected} />
    </label>
    <label class="vrow" style="cursor:pointer">
      <span>Low VRAM</span>
      <input type="checkbox" bind:checked={video.lowVramMode} disabled={video.loading} />
    </label>
    <label class="vrow" style="cursor:pointer"
      title="Streams the transformer from system RAM to fit bigger clips. SAFE-GUARDED: refused with a clear error if the model won't fit physical RAM (fp32 14B = 54 GB won't; it froze the PC before). Only works for models that fit RAM (e.g. 4-bit A14B). Slow.">
      <span>Park to RAM <span style="font-size:0.85em;color:var(--accent2,#e0a060)">· only if model fits RAM · slow</span></span>
      <input type="checkbox" bind:checked={video.blockOffload} disabled={video.loading} />
    </label>
    <div class="vhint">
      {#if is14bSelected}
        <strong style="color:#e0a060">14B high-quality. Long native quality means 16 FPS chunks with I2V tail-frame conditioning. Pure Wan2.2 T2V cannot condition long chunks.</strong>
      {:else if isTi2v5bSelected}
        <strong style="color:#e0a060">TI2V-5B quality route. Use 480p, 121f @ 24 FPS, 50 steps, CFG 5; longer clips chain native chunks and then go through Refine/Upscale.</strong>
      {:else if video.qualityMode}
        Experimental bf16: full-precision weights streamed from RAM (sharper). Optimized offload aims for 30s–1m gens. If a gen OOMs/errors, untick + Re-Load.
      {:else}
        Fast 4-bit transformer (recommended). Low VRAM frees inactive modules between stages; slower next run, same model/output path.
      {/if}
    </div>

    <div class="vlabel" style="margin-top:14px; display:flex; justify-content:space-between; align-items:center">
      <span>Params</span>
      <button class="vauto" onclick={autoSet} title="Snap all params to this model's best/required settings">⚙ Auto</button>
    </div>
    <div class="vrow"><span>Frames</span><input type="number" bind:value={video.numFrames} min="9" max="2000" step="4" /></div>
    <div class="vrow"><span>Length</span>
      <div style="flex:1; display:flex; flex-direction:column; gap:2px;">
        <div style="display:flex; align-items:center; gap:8px;">
          <input type="range" min="1" max="120" step="1"
            value={currentSec}
            oninput={(e) => setSeconds(parseInt((e.target as HTMLInputElement).value), true)}
            style="flex:1; accent-color: var(--accent);" />
          <span style="font-family:var(--mono); font-size:11px; min-width:32px; text-align:right;">{currentSec}s</span>
        </div>
        <div class="vsize" style="margin-top:2px;">
          <button class="vsize-btn" class:on={video.numFrames === framesFor(2)} onclick={() => setSeconds(2)} title="~2 second clip">2s</button>
          <button class="vsize-btn" class:on={video.numFrames === framesFor(3)} onclick={() => setSeconds(3)} title="~3 second clip">3s</button>
          <button class="vsize-btn" class:on={video.numFrames === framesFor(5)} onclick={() => setSeconds(5)} title="~5 second native target: 5B uses 121f @24 FPS; 14B uses 16 FPS chunks">5s</button>
        </div>
        <div class="vsize" style="margin-top:2px;">
          <button class="vsize-btn" class:on={currentSec === 10} onclick={() => setSeconds(10)} title="~10s. TI2V-5B chains 121f @24 FPS chunks; A14B pure T2V uses backend chunked denoise; I2V uses tail-frame chunks.">10s</button>
          <button class="vsize-btn" class:on={currentSec === 20} onclick={() => setSeconds(20)} title="~20s. Long 5B/I2V routes chain tail-conditioned native chunks; pure A14B T2V uses single-prompt backend chunks.">20s</button>
          <button class="vsize-btn" class:on={currentSec === 30} onclick={() => setSeconds(30)} title="~30s. Best for 480p-first chained tests; 14B pure T2V keeps a single prompt, 5B starts from text then tail-conditions later chunks.">30s</button>
        </div>
      </div>
    </div>
    <div class="vrow"><span>Steps</span><input type="number" bind:value={video.steps} min="1" max="60" /></div>
    <div class="vrow"><span>CFG</span><input type="number" bind:value={video.cfg} min="1" max="15" step="0.5" /></div>
    <div class="vrow"><span>Scheduler</span>
      <div class="vsize">
        <button class="vsize-btn" class:on={video.scheduler === "auto"} onclick={() => setScheduler("auto")} title="Use the scheduler shipped with the model folder">Model</button>
        <button class="vsize-btn" class:on={video.scheduler === "euler_beta"} onclick={() => setScheduler("euler_beta")} title="FlowMatch Euler with beta sigmas">Euler Beta</button>
      </div>
    </div>
    <div class="vrow"><span>Shift</span><input type="number" bind:value={video.shift} min="1" max="16" step="0.5" /></div>
    <div class="vrow"><span>Size</span>
      <div class="vsize">
        <button class="vsize-btn" class:on={!video.resLocked && video.width === 832 && video.height === 480}
                disabled={video.resLocked} onclick={() => setRes(832, 480)}
                title="832×480 — fastest and rock-solid stable">SD 480p</button>
        <button class="vsize-btn" class:on={!video.resLocked && video.width === 1280 && video.height === 704}
                disabled={video.resLocked} onclick={() => setRes(1280, 704)}
                title="1280×704 HD — 14B uses safe 49-frame chunks on 16GB. TI2V-5B can be tested short, but the stable path is 480p then Refine/Upscale.">HD 720p</button>
      </div>
    </div>
    <div class="vrow"><span>Width</span><input type="number" bind:value={video.width} min="256" max="1360" step="16" disabled={video.resLocked} /></div>
    <div class="vrow"><span>Height</span><input type="number" bind:value={video.height} min="256" max="768" step="16" disabled={video.resLocked} /></div>
    <div class="vrow"><span>FPS</span><input type="number" bind:value={video.fps} min="2" max="30" /></div>
    <div class="vrow"><span>Seed</span><input type="number" bind:value={video.seed} /></div>
    <div class="vhint" style="margin-top:4px">
      {#if video.resLocked}🔒 This model locks resolution — Width/Height fixed.{:else}−1 seed = random. HD 14B uses 49f chunks on 16GB; TI2V-5B targets 121f @24 FPS at 480p. Long native quality needs a model that can take a tail image. Keyframe FPS is manual and trades motion quality for duration.{/if}
    </div>
    {/if}

    <div class="vlabel" style="margin-top:14px; display:flex; justify-content:space-between; align-items:center">
      <span>Activity</span>
      {#if (video.loading || video.generating)}<span class="vdot"></span>{/if}
    </div>
    <div class="vterm" bind:this={termEl} aria-readonly="true">
      {#if video.log.length === 0}
        <div class="vterm-line vterm-idle">— idle — load a model to begin</div>
      {:else}
        {#each video.log as line}<div class="vterm-line" title={line}>{line}</div>{/each}
      {/if}
    </div>
  </div>

  <div class="ig-main">
    <textarea class="vprompt" placeholder="Describe the video…" bind:value={video.prompt}></textarea>
    <textarea class="vprompt vneg" placeholder="Negative prompt (what to avoid)" bind:value={video.negPrompt}></textarea>
    <label class="vcheck" style="margin-top:6px; font-size:12px; display:flex; align-items:center; gap:8px; color:var(--text2)">
      <input type="checkbox" bind:checked={video.anatomyLock} />
      Anatomy lock
      <span style="color:var(--text3); font-size:11px">
        — stops vulva→mouth/lips and fused genitals (auto pos+neg on explicit prompts; used by Refine too)
      </span>
    </label>

    <!-- Storyboard: multiple prompts for segmented long video on 14B (T2V or SVI I2V) or TI2V-5B.
         Auto-chains + force seam blend. SVI I2V path uses image seeding for best continuity.
         Target longer total video (20s–minutes) with good character coherence. -->
    <div class="vlabel" style="margin-top:8px">Storyboard Prompts (1-4 segments)</div>
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px; font-size:11px;">
      {#each video.storyboardPrompts as _, i}
        <div>
          <span style="color:var(--text3)">Seg {i+1}:</span>
          <textarea class="vprompt" style="min-height:42px; font-size:11px; padding:4px 6px; margin-top:2px"
            bind:value={video.storyboardPrompts[i]}
            placeholder={i===0 ? "Opening action..." : "Next beat / continuation..."}></textarea>
        </div>
      {/each}
    </div>
    <div style="font-size:10px; color:var(--text3); margin-top:2px">
      <strong>14B or TI2V-5B storyboard.</strong> One load only. 5B starts from text at 480p, then uses the last frame as the seed for continuity. For I2V/SVI: uses last frame as seed for continuity. Fill later boxes with strong "continues the exact same shot, anatomy, camera, lighting seamlessly...". Use Refine after.
    </div>

    <div class="i2v-row">
      <label class="i2v-pick">
        🖼 {video.imageB64 ? "Change image" : "Add image → animate (i2v)"}
        <input type="file" accept="image/*" onchange={pickImage} hidden />
      </label>
      {#if video.imageB64}
        <img class="i2v-thumb" src="data:image/*;base64,{video.imageB64}" alt="i2v source" />
        <span class="i2v-name">{video.imageName}</span>
        <button class="vbtn-ghost i2v-clear" onclick={clearImage}>✕</button>
        <span class="i2v-tag">image-to-video</span>
      {/if}
    </div>

    <button class="vgen" onclick={generate} disabled={!loaded || video.generating || !video.prompt.trim()}>
      {video.generating ? (video.progress > 0 ? `Generating… ${pct}%` : video.loadStatus || "Preparing…") : loaded ? (video.imageB64 ? "🖼 Animate image" : "🎬 Generate") : "Load a model first"}
    </button>
    <button class="vbtn-ghost" onclick={generateStoryboard} disabled={!loaded || video.generating} title="Chain segments using the storyboard prompts above (each uses current Length e.g. 5s + auto last-frame stitch + next prompt)">
      📖 Storyboard
    </button>

    {#if video.generating}
      {#if video.progress > 0}
        <div class="vbar"><div class="vbar-fill" style="width:{pct}%"></div></div>
        <div class="vbar-label">Step {video.progress} / {video.progressTotal}</div>
      {:else}
        <div class="vbar vbar-indet"><div class="vbar-fill"></div></div>
        <div class="vbar-label">{video.loadStatus || "Preparing…"}</div>
      {/if}
      {#if video.previewB64}
        <div class="vlive-preview">
          <img src="data:image/jpeg;base64,{video.previewB64}" alt="Live latent preview" />
          <div class="vbar-label">
            Preview step {video.previewStep} · frames {video.previewFrames.join(", ")}
          </div>
        </div>
      {/if}
    {/if}

    {#if video.error}
      <div class="verr" title={video.error}>{video.error}</div>
    {:else if video.resultB64}
      <div class="vresult">
        <!-- svelte-ignore a11y_media_has_caption -->
        <video id="vplayer" class="vplayer" controls autoplay loop src="data:video/mp4;base64,{video.resultB64}"></video>
        <div class="vmeta">
          {video.frames ?? ""} frames · generated in {video.elapsed}s
          {#if video.enhanced}<span class="qpass-badge">✨ enhanced</span>{/if}
        </div>
        <button class="vbtn-ghost" onclick={saveMp4}>💾 Save MP4</button>
        <button class="vbtn-ghost" onclick={extendClip} disabled={!video.resultB64} title="Append a new segment starting from the current clip's last frame. Backend will stitch + lightly blend the seam. Use a strong consistent prompt (or Refine pass) to limit identity drift.">➕ Extend (stitch + blend)</button>

        <div class="station">
          <div class="station-head">
            <span>Video Station</span>
            <span class="station-count">Frame {stationFrame} / {frameMax()}</span>
          </div>

          <div class="station-body">
            <div class="station-preview">
              {#if stationFrameB64}
                <img src="data:image/png;base64,{stationFrameB64}" alt="selected frame" />
              {:else}
                <div class="station-empty">Select a frame</div>
              {/if}
            </div>

            <div class="station-tools">
              <div class="station-row">
                <span>Frame</span>
                <input
                  type="range"
                  min="0"
                  max={frameMax()}
                  step="1"
                  value={stationFrame}
                  oninput={(e) => scrubFrame(parseInt((e.target as HTMLInputElement).value))}
                />
                <input
                  class="station-frame-num"
                  type="number"
                  min="0"
                  max={frameMax()}
                  step="1"
                  value={stationFrame}
                  onchange={(e) => scrubFrame(parseInt((e.target as HTMLInputElement).value))}
                />
              </div>

              <div class="station-actions">
                <button class="vbtn-ghost" onclick={captureCurrentFrame} disabled={video.enhancing}>Capture</button>
                <button class="vbtn-ghost" onclick={saveSelectedFrame} disabled={video.enhancing}>Save PNG</button>
                <button class="vbtn-ghost" onclick={() => useSelectedFrameAsSeed(false)} disabled={video.enhancing}>Use as seed</button>
                <button class="vbtn-ghost" onclick={() => useSelectedFrameAsSeed(true)} disabled={video.enhancing}>
                  {stationAnchorFrame === stationFrame ? "Anchor set" : "Set anchor"}
                </button>
              </div>

              <div class="station-split">
                <div>
                  <div class="station-sub">Preview</div>
                  <div class="vsize">
                    <button class="vsize-btn" class:on={stationPlaybackRate === 0.25} onclick={() => setPlaybackRate(0.25)}>¼×</button>
                    <button class="vsize-btn" class:on={stationPlaybackRate === 0.5} onclick={() => setPlaybackRate(0.5)}>½×</button>
                    <button class="vsize-btn" class:on={stationPlaybackRate === 1} onclick={() => setPlaybackRate(1)}>1×</button>
                  </div>
                </div>
                <div>
                  <div class="station-sub">Keyframes</div>
                  <div class="vsize">
                    <button class="vsize-btn" onclick={() => setKeyframeFps(2)} disabled={video.generating || video.enhancing}>2 FPS</button>
                    <button class="vsize-btn" onclick={() => setKeyframeFps(4)} disabled={video.generating || video.enhancing}>4 FPS</button>
                    <button class="vsize-btn" onclick={() => setKeyframeFps(8)} disabled={video.generating || video.enhancing}>8 FPS</button>
                  </div>
                </div>
              </div>

              <div class="station-actions">
                <button class="vbtn-ghost" onclick={() => bakeSlowMo(2)} disabled={video.enhancing}>Bake 2× slow</button>
                <button class="vbtn-ghost" onclick={() => bakeSlowMo(4)} disabled={video.enhancing}>Bake 4× slow</button>
                <button class="vbtn-ghost" onclick={() => smoothToFps(16)} disabled={video.enhancing || video.fps >= 16}>Smooth to 16 FPS</button>
              </div>
              <div class="vhint">Keyframe FPS lowers Wan's native frame count. Smooth/bake uses the existing optical-flow pass, not another 14B generation.</div>
            </div>
          </div>
        </div>

        <div class="qpass">
          <div class="qpass-head">✨ Quality Pass <span class="qpass-sub">— frees the generator, runs each stage on its own</span></div>
          <div class="qpass-toggles">
            <label><input type="checkbox" bind:checked={video.doRefine} disabled={video.enhancing} /> Refine <span class="qpass-tag">bf16 v2v</span></label>
            <label><input type="checkbox" bind:checked={video.doFace} disabled={video.enhancing} /> Restore faces <span class="qpass-tag">CodeFormer</span></label>
            <label><input type="checkbox" bind:checked={video.doUpscale} disabled={video.enhancing} /> Upscale <span class="qpass-tag">2× ESRGAN</span></label>
            <label><input type="checkbox" bind:checked={video.doInterpolate} disabled={video.enhancing} /> Interpolate <span class="qpass-tag">2× flow</span></label>
          </div>
          {#if video.doRefine}
            <div class="vrow"><span>Strength</span><input type="number" bind:value={video.refineStrength} min="0.1" max="0.9" step="0.05" disabled={video.enhancing} /></div>
          {/if}
          <button class="qpass-run" onclick={enhance} disabled={video.enhancing || (!video.doRefine && !video.doFace && !video.doUpscale && !video.doInterpolate)}>
            {video.enhancing ? (video.loadStatus || "Enhancing…") : "✨ Run Quality Pass"}
          </button>
          {#if video.enhancing}
            <div class="vbar vbar-indet"><div class="vbar-fill"></div></div>
          {/if}
          <div class="vhint">Unloads the generator first (all VRAM goes to the pass). You'll re-Load the model to generate again.</div>
        </div>
      </div>
    {:else if !video.generating}
      <div class="vplaceholder">
        <div style="font-size:48px;opacity:0.15">🎬</div>
        <div style="color:var(--text3);font-size:13px;margin-top:8px">Your clip will appear here</div>
      </div>
    {/if}
  </div>
</div>

<style>
  .vlabel { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text3); margin-bottom: 6px; }
  .vsel { width: 100%; min-width: 0; max-width: 100%; margin-bottom: 4px; }
  .vbtn-ghost { background: transparent; border: 1px solid var(--border); color: var(--text2); border-radius: var(--radius-sm); padding: 4px 8px; font-size: 11px; cursor: pointer; width: 100%; }
  .vbtn-ghost:hover { background: var(--bg3); color: var(--text); }
  .vauto { font-size: 10px; font-weight: 600; background: var(--bg3); border: 1px solid var(--border); color: var(--text2); border-radius: var(--radius-sm); padding: 2px 8px; cursor: pointer; text-transform: none; letter-spacing: 0; }
  .vauto:hover { background: var(--accent); color: #fff; border-color: var(--accent); }
  .vhint { font-size: 9px; color: var(--text3); margin-top: 4px; line-height: 1.5; }
  .vrow { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; min-width: 0; }
  .vrow span { font-size: 11px; color: var(--text2); min-width: 48px; }
  .vrow input { flex: 1; min-width: 0; font-family: var(--mono); }
  .vsize { flex: 1; min-width: 0; display: flex; gap: 6px; }
  .vsize-btn { flex: 1; font-size: 11px; font-weight: 600; background: var(--bg3); border: 1px solid var(--border); color: var(--text2); border-radius: var(--radius-sm); padding: 4px 0; cursor: pointer; }
  .vsize-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--text); }
  .vsize-btn.on { background: var(--accent); color: #fff; border-color: var(--accent); }
  .vsize-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  /* Mini non-typable terminal — live activity log. */
  .vterm {
    background: #0a0c10; border: 1px solid var(--border); border-radius: var(--radius-sm);
    font-family: var(--mono); font-size: 10.5px; line-height: 1.5; color: #8fb98f;
    padding: 6px 8px; height: 150px; width: 100%; max-width: 100%; overflow-y: auto; overflow-x: hidden; user-select: text;
    white-space: pre-wrap; word-break: break-word; overflow-wrap: anywhere;
  }
  .vterm-line { opacity: 0.92; max-width: 100%; overflow-wrap: anywhere; word-break: break-word; }
  .vterm-line:last-child { color: #b6e3b6; opacity: 1; }   /* highlight newest */
  .vterm-idle { color: var(--text3); font-style: italic; }
  .vdot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); animation: vdot-pulse 1s ease-in-out infinite; }
  @keyframes vdot-pulse { 0%,100% { opacity: 0.3; } 50% { opacity: 1; } }

  .ig-main { gap: 10px; min-width: 0; overflow-x: hidden; }
  .vprompt { width: 100%; min-height: 70px; resize: vertical; font-family: var(--sans); font-size: 13px; line-height: 1.6; padding: 10px 12px; }
  .vneg { min-height: 42px; color: var(--text2); }

  .i2v-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .i2v-pick { font-size: 12px; color: var(--text2); border: 1px dashed var(--border); border-radius: var(--radius-sm); padding: 6px 10px; cursor: pointer; }
  .i2v-pick:hover { background: var(--bg3); color: var(--text); border-color: var(--accent); }
  .i2v-thumb { height: 38px; border-radius: var(--radius-sm); border: 1px solid var(--border); }
  .i2v-name { font-size: 11px; color: var(--text3); font-family: var(--mono); max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .i2v-clear { width: auto; padding: 2px 8px; }
  .i2v-tag { font-size: 9px; text-transform: uppercase; letter-spacing: 0.05em; color: #c4a3fb; background: rgba(168,85,247,0.18); border-radius: 4px; padding: 2px 6px; }
  .vgen { align-self: flex-start; padding: 9px 22px; font-size: 14px; font-weight: 600; background: var(--accent); border: none; color: #fff; border-radius: var(--radius); cursor: pointer; }
  .vgen:hover:not(:disabled) { background: #7a9cf7; }
  .vgen:disabled { opacity: 0.5; cursor: not-allowed; }

  .vbar { height: 5px; background: var(--bg3); border-radius: 3px; overflow: hidden; }
  .vbar-fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 0.3s; }
  /* Indeterminate (no step count yet — e.g. text-encoder loading): sliding sweep. */
  .vbar-indet .vbar-fill { width: 35%; transition: none; animation: vbar-sweep 1.1s ease-in-out infinite; }
  @keyframes vbar-sweep { 0% { margin-left: -35%; } 100% { margin-left: 100%; } }
  .vbar-label { font-size: 11px; color: var(--text3); font-family: var(--mono); overflow-wrap: anywhere; }
  .vlive-preview { width: min(100%, 768px); display: flex; flex-direction: column; gap: 5px; }
  .vlive-preview img { display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: #05060a; border: 1px solid var(--border); border-radius: var(--radius-sm); }
  .verr { background: rgba(248,113,113,0.08); border: 1px solid rgba(248,113,113,0.3); color: var(--red); font-size: 12px; padding: 10px 12px; border-radius: var(--radius-sm); line-height: 1.5; white-space: pre-wrap; overflow-wrap: anywhere; max-height: 30vh; max-width: 100%; overflow-y: auto; overflow-x: hidden; }

  .vresult { display: flex; flex-direction: column; gap: 8px; align-items: flex-start; }

  .station {
    width: 100%;
    padding: 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--bg2);
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .station-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    font-size: 12px;
    font-weight: 700;
    color: var(--text);
  }
  .station-count {
    font-family: var(--mono);
    font-size: 10px;
    font-weight: 500;
    color: var(--text3);
    white-space: nowrap;
  }
  .station-body {
    display: grid;
    grid-template-columns: minmax(160px, 240px) minmax(0, 1fr);
    gap: 12px;
    align-items: start;
  }
  .station-preview {
    width: 100%;
    aspect-ratio: 16 / 9;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: #050608;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }
  .station-preview img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }
  .station-empty {
    color: var(--text3);
    font-size: 11px;
    font-family: var(--mono);
  }
  .station-tools {
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .station-row {
    display: grid;
    grid-template-columns: 44px minmax(0, 1fr) 72px;
    gap: 8px;
    align-items: center;
  }
  .station-row span,
  .station-sub {
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text3);
  }
  .station-row input[type="range"] {
    min-width: 0;
    accent-color: var(--accent);
  }
  .station-frame-num {
    width: 72px;
    min-width: 0;
    font-family: var(--mono);
    font-size: 11px;
  }
  .station-actions {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 6px;
  }
  .station-actions .vbtn-ghost {
    margin: 0;
    min-height: 28px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .station-split {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
  }
  .station-split .vsize {
    margin-top: 4px;
  }

  /* Quality Pass panel */
  .qpass { width: 100%; margin-top: 8px; padding: 12px; border: 1px solid var(--border); border-radius: var(--radius); background: var(--bg2); display: flex; flex-direction: column; gap: 8px; }
  .qpass-head { font-size: 12px; font-weight: 700; color: var(--text); }
  .qpass-sub { font-weight: 400; color: var(--text3); font-size: 11px; }
  .qpass-toggles { display: flex; flex-wrap: wrap; gap: 14px; }
  .qpass-toggles label { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text2); cursor: pointer; }
  .qpass-tag { font-size: 9px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text3); border: 1px solid var(--border); border-radius: 4px; padding: 1px 4px; }
  .qpass-run { align-self: flex-start; padding: 8px 18px; font-size: 13px; font-weight: 600; background: linear-gradient(135deg, #a855f7, #6366f1); border: none; color: #fff; border-radius: var(--radius); cursor: pointer; }
  .qpass-run:hover:not(:disabled) { filter: brightness(1.1); }
  .qpass-run:disabled { opacity: 0.5; cursor: not-allowed; }
  .qpass-badge { font-size: 10px; background: rgba(168,85,247,0.18); color: #c4a3fb; border-radius: 4px; padding: 1px 6px; margin-left: 6px; }
  .vplayer { max-width: 100%; max-height: 60vh; border-radius: var(--radius); border: 1px solid var(--border); background: #000; }
  .vmeta { font-size: 11px; color: var(--text3); font-family: var(--mono); }
  .vplaceholder { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 280px; }

  @media (max-width: 860px) {
    .station-body,
    .station-split {
      grid-template-columns: 1fr;
    }
    .station-actions {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
</style>
