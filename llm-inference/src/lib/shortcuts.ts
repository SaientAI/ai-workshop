// Global keyboard shortcuts.
//
// Registered on the WINDOW CAPTURE phase (see App.svelte) so app shortcuts fire
// before the xterm terminal or any input sees the event. When we claim a key we
// call stopImmediatePropagation() so the terminal never receives it; unclaimed
// keys fall through untouched, so typing and the shell keep working normally.

import { ui, chat, agent } from "./state.svelte.js";
import type { Screen, AgentTab, ChatTab } from "./types.js";
import * as T from "./tauri.js";

const SCREENS: Screen[]       = ["chat", "agent", "imggen", "assets", "video", "tts", "lora", "merge"];
const AGENT_TABS: AgentTab[]  = ["files", "terminal", "planner", "memory"];

export interface Shortcut { keys: string[]; desc: string; group: string; }

// Shown in the help overlay (ShortcutsHelp.svelte).
export const SHORTCUTS: Shortcut[] = [
  { group: "Navigation", keys: ["Ctrl", "1 – 8"], desc: "Switch screen — Chat · Agent · Image · Assets · Video · TTS · LoRA · Merge" },
  { group: "Navigation", keys: ["Ctrl", "Tab"],   desc: "Cycle tabs (chat ⇄ system · agent files/terminal/planner/memory)" },
  { group: "Chat",       keys: ["Enter"],          desc: "Send message" },
  { group: "Chat",       keys: ["Shift", "Enter"], desc: "Newline in the message box" },
  { group: "Chat",       keys: ["Ctrl", "K"],      desc: "Clear the conversation" },
  { group: "Chat",       keys: ["Esc"],            desc: "Stop generating" },
  { group: "Agent",      keys: ["Ctrl", "Shift", "K"], desc: "Toggle Saient on/off" },
  { group: "Agent",      keys: ["Ctrl", "Shift", "W"], desc: "Toggle agent write mode" },
  { group: "General",    keys: ["?"],              desc: "Show / hide this help" },
  { group: "General",    keys: ["Ctrl", "/"],      desc: "Show / hide this help" },
];

function typing(): boolean {
  const el = document.activeElement as HTMLElement | null;
  if (!el) return false;
  return el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable;
}

async function stopGen() {
  if (!chat.streaming) return;
  try { await T.stopGenerate(); } catch { /* ignore */ }
  chat.streaming = false;
  chat.pendingUserText = "";
  const last = chat.messages[chat.messages.length - 1];
  if (last?.streaming) { last.streaming = false; last.stopped = true; last.ts = Date.now(); }
}

export function handleKey(e: KeyboardEvent) {
  const k = e.key;
  const ctrl = e.ctrlKey || e.metaKey;
  // Claim a shortcut: stop the terminal/inputs from also acting on it.
  const claim = () => { e.preventDefault(); e.stopImmediatePropagation(); };

  // ── Always-on (work even while typing or with the terminal focused) ──────────

  // Ctrl+1..8 → switch screen
  if (ctrl && !e.shiftKey && !e.altKey && k >= "1" && k <= "8") {
    ui.screen = SCREENS[+k - 1]; claim(); return;
  }

  // Ctrl+/ → help
  if (ctrl && k === "/") { ui.showShortcuts = !ui.showShortcuts; claim(); return; }

  // Ctrl+Shift+K → toggle Saient
  if (ctrl && e.shiftKey && (k === "K" || k === "k")) {
    ui.saientEnabled = !ui.saientEnabled;
    localStorage.setItem("saient_enabled", String(ui.saientEnabled));
    claim(); return;
  }

  // Ctrl+Shift+W → toggle agent write mode
  if (ctrl && e.shiftKey && (k === "W" || k === "w")) {
    ui.agentWriteMode = !ui.agentWriteMode;
    localStorage.setItem("agent_write_mode", String(ui.agentWriteMode));
    T.setAgentWriteMode(ui.agentWriteMode).catch(() => {});
    claim(); return;
  }

  // Ctrl+Tab → cycle tabs within the current screen
  if (ctrl && k === "Tab") {
    if (ui.screen === "agent") {
      const i = AGENT_TABS.indexOf(agent.tab as AgentTab);
      const n = AGENT_TABS.length;
      agent.tab = AGENT_TABS[(i + (e.shiftKey ? n - 1 : 1)) % n];
    } else if (ui.screen === "chat") {
      chat.tab = (chat.tab === "chat" ? "system" : "chat") as ChatTab;
    }
    claim(); return;
  }

  // Esc → stop generating (only consume it when actually streaming)
  if (k === "Escape") {
    if (ui.showShortcuts) { ui.showShortcuts = false; claim(); return; }
    if (chat.streaming) { stopGen(); claim(); return; }
    return; // let Esc behave normally otherwise
  }

  // Ctrl+K → clear chat. Chat screen only; on the agent screen Ctrl+K belongs to
  // the terminal (kill-line), so we don't claim it there.
  if (ctrl && !e.shiftKey && (k === "k" || k === "K") && ui.screen === "chat") {
    chat.messages = []; claim(); return;
  }

  // ── Only when NOT typing in an input / terminal ──────────────────────────────
  if (typing()) return;
  if (k === "?") { ui.showShortcuts = !ui.showShortcuts; claim(); return; }
}
