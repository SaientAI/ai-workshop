// Markdown renderer with syntax highlighting.
// Used by Message.svelte for AI response content.

import { marked, Renderer } from "marked";
import hljs from "highlight.js/lib/core";

// Register the languages we actually see in LLM output — import only what
// we need so tree-shaking keeps the bundle small.
import langBash     from "highlight.js/lib/languages/bash";
import langCss      from "highlight.js/lib/languages/css";
import langHtml     from "highlight.js/lib/languages/xml";   // hljs uses xml for html
import langJs       from "highlight.js/lib/languages/javascript";
import langJson     from "highlight.js/lib/languages/json";
import langPython   from "highlight.js/lib/languages/python";
import langRust     from "highlight.js/lib/languages/rust";
import langTs       from "highlight.js/lib/languages/typescript";
import langYaml     from "highlight.js/lib/languages/yaml";
import langSql      from "highlight.js/lib/languages/sql";
import langMarkdown from "highlight.js/lib/languages/markdown";

hljs.registerLanguage("bash",       langBash);
hljs.registerLanguage("sh",         langBash);
hljs.registerLanguage("shell",      langBash);
hljs.registerLanguage("css",        langCss);
hljs.registerLanguage("html",       langHtml);
hljs.registerLanguage("xml",        langHtml);
hljs.registerLanguage("javascript", langJs);
hljs.registerLanguage("js",         langJs);
hljs.registerLanguage("json",       langJson);
hljs.registerLanguage("python",     langPython);
hljs.registerLanguage("py",         langPython);
hljs.registerLanguage("rust",       langRust);
hljs.registerLanguage("rs",         langRust);
hljs.registerLanguage("typescript", langTs);
hljs.registerLanguage("ts",         langTs);
hljs.registerLanguage("yaml",       langYaml);
hljs.registerLanguage("yml",        langYaml);
hljs.registerLanguage("sql",        langSql);
hljs.registerLanguage("markdown",   langMarkdown);
hljs.registerLanguage("md",         langMarkdown);

const renderer = new Renderer();

renderer.code = function ({ text, lang }) {
  const validLang = lang && hljs.getLanguage(lang) ? lang : "";
  const highlighted = validLang
    ? hljs.highlight(text, { language: validLang }).value
    : hljs.highlightAuto(text).value;
  const cls = validLang ? `hljs language-${validLang}` : "hljs";
  const label = validLang
    ? `<span class="code-lang">${validLang}</span>`
    : "";
  return `<div class="code-block">${label}<pre><code class="${cls}">${highlighted}</code></pre></div>`;
};

marked.use({
  renderer,
  gfm: true,
  breaks: true,
});

export function renderMarkdown(text: string): string {
  // marked() is synchronous when no async extensions are registered
  return marked(text) as string;
}
