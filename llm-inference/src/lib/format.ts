// Text formatting and display utilities

export function escHtml(s: unknown): string {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

export function escAttr(s: unknown): string {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function prettyPath(path: string): string {
  return path.replace(/^\/home\/[^/]+/, "~");
}

export function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const min = Math.floor(total / 60);
  const sec = String(total % 60).padStart(2, "0");
  return `${min}:${sec}`;
}

export function formatBytes(b: number): string {
  if (b < 1024) return `${b} B`;
  if (b < 1024 ** 2) return `${(b / 1024).toFixed(1)} KB`;
  if (b < 1024 ** 3) return `${(b / 1024 ** 2).toFixed(1)} MB`;
  return `${(b / 1024 ** 3).toFixed(2)} GB`;
}

export function stripThinkTags(text: string): string {
  return text.replace(/<think>[\s\S]*?<\/think>/g, "").replace(/<think>[\s\S]*/g, "");
}

export function splitReasoningAnswer(text: string): { reasoning: string; answer: string } {
  const thinkMatch = text.match(/^<think>([\s\S]*?)<\/think>([\s\S]*)$/);
  if (thinkMatch) {
    return { reasoning: thinkMatch[1].trim(), answer: thinkMatch[2].trim() };
  }
  return { reasoning: "", answer: text };
}

/** Remove leaked <think> blocks that appeared mid-stream. */
export function splitLeakedThinking(text: string): { display: string; leaked: string } {
  const leakMatch = text.match(/^([\s\S]*?)<think>([\s\S]*)$/);
  if (leakMatch && !text.includes("</think>")) {
    return { display: leakMatch[1], leaked: leakMatch[2] };
  }
  return { display: text, leaked: "" };
}


export function isCasualGreeting(text: string): boolean {
  return /^(hi|hello|hey|sup|yo|greetings|howdy|hola|salut|bonjour|hiya|what'?s up|good (morning|afternoon|evening))[.!?]?\s*$/i.test(
    text.trim()
  );
}

export function isArtifactRequest(text: string): boolean {
  const hasVerb =
    /\b(build|create|design|implement|make|write|generate|render|draw)\b/i.test(text);
  const hasNoun =
    /\b(ui|interface|component|page|site|app|tool|game|calculator|timer|clock|counter|widget|viewer|editor|player|simulator|demo|canvas|chart|graph|form|dashboard|visuali[sz]ation|html|css|javascript)\b/i.test(
      text
    );
  return hasVerb && hasNoun;
}

export function friendlyLoadError(raw: string): string {
  if (raw.includes("No such file")) return "Model file not found. Check the path.";
  if (raw.includes("CUDA") || raw.includes("cuda")) return "GPU error. Try GPU layers = 0 for CPU mode.";
  if (raw.includes("out of memory") || raw.includes("OOM")) return "Out of memory. Reduce GPU layers or use a smaller model.";
  if (raw.includes("permission denied")) return "Permission denied reading model file.";
  return raw.length > 200 ? raw.slice(0, 200) + "…" : raw;
}

export function isRefusal(text: string): boolean {
  const start = text.trim().slice(0, 250);
  return (
    /^I(?:'m| am) sorry,? but I (?:can't|cannot|am not able to)/i.test(start) ||
    /^I(?:'m| am) sorry,? but (?:that|this)/i.test(start) ||
    /^I (?:can't|cannot) (?:help|assist) with that/i.test(start) ||
    /^(?:I'm sorry|I apologize)[.,]? (?:but )?I(?:'m| am) (?:not able|unable) to/i.test(start) ||
    /^(?:I'm sorry|I apologize)[.,]?\s*(?:but )?I (?:can't|cannot) (?:provide|help|assist)/i.test(start)
  );
}

export function friendlyGenerateError(raw: string): string {
  if (raw.includes("No model loaded")) return "No model loaded. Start a server first.";
  if (raw.includes("context")) return "Context limit reached. Clear chat or reduce max tokens.";
  return raw.length > 200 ? raw.slice(0, 200) + "…" : raw;
}
