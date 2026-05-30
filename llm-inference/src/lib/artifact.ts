// Artifact parsing and rendering logic

export interface ArtifactData {
  type: string;
  title: string;
  content: string;
  complete: boolean;
}

export interface SplitResult {
  chatText: string;
  artifact: ArtifactData | null;
}

export function splitArtifact(text: string): SplitResult {
  const openIdx = text.indexOf("<artifact");
  if (openIdx === -1) return { chatText: text, artifact: null };

  const tagEnd = text.indexOf(">", openIdx);
  if (tagEnd === -1) return { chatText: text.substring(0, openIdx).trim(), artifact: null };

  const tag = text.substring(openIdx, tagEnd + 1);
  const typeM = tag.match(/type="([^"]*)"/i);
  const titleM = tag.match(/title="([^"]*)"/i);
  const type = typeM ? typeM[1] : "html";
  const title = titleM ? titleM[1] : "Artifact";

  const chatBefore = text.substring(0, openIdx).trim();
  const afterOpen = text.substring(tagEnd + 1);

  const closeMatch = afterOpen.match(/<;?\/artifact\s*>/i);
  const closeIdx = closeMatch?.index ?? -1;
  if (closeIdx === -1) {
    return { chatText: chatBefore, artifact: { type, title, content: afterOpen, complete: false } };
  }
  const content = afterOpen.substring(0, closeIdx);
  const chatAfter = afterOpen.substring(closeIdx + (closeMatch?.[0].length ?? 0)).trim();
  const chatText = [chatBefore, chatAfter].filter(Boolean).join("\n\n");
  return { chatText, artifact: { type, title, content, complete: true } };
}

export function stripArtifactTag(html: string): string {
  // Repair malformed closing tags (e.g. <;/style> → </style>)
  html = html.replace(/<;\/(\w+)/g, "</$1");
  // Remove HTML-encoded artifact tags visible as text in WebKitGTK
  html = html.replace(/&lt;\/artifact[^&>]*(?:&gt;)?/gi, "");
  // Truncate at </artifact> — everything after is outside the HTML doc
  const idx = html.toLowerCase().indexOf("</artifact");
  return idx === -1 ? html : html.substring(0, idx).trimEnd();
}

const CSP_META = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:;">`;

const ERR_OVERLAY = `<style>#_err{position:fixed;bottom:0;left:0;right:0;background:#7f1d1d;color:#fca5a5;font:11px monospace;padding:4px 8px;display:none;z-index:9999;white-space:pre-wrap;}</style>
<div id="_err"></div>
<script>
function _postErr(msg,stack){
  window.parent.postMessage({type:'artifact-error',message:msg,stack:stack||''},'*');
  var d=document.getElementById('_err');
  if(d){d.style.display='block';d.textContent=stack||msg;}
}
window.onerror=function(m,s,l,c,e){_postErr(m,e?.stack);return true;};
window.addEventListener('unhandledrejection',function(ev){
  _postErr(String(ev.reason?.message||ev.reason||'Unhandled rejection'),ev.reason?.stack);
});
<\/script>`;

export function buildSrcdoc(html: string): string {
  let clean = stripArtifactTag(html).trim();
  if (!clean) return "";
  // Insert CSP meta tag after <head> opener, or prepend if no head element
  const headMatch = clean.match(/<head[^>]*>/i);
  if (headMatch) {
    const idx = (clean.toLowerCase().indexOf(headMatch[0].toLowerCase())) + headMatch[0].length;
    clean = clean.slice(0, idx) + CSP_META + clean.slice(idx);
  } else {
    clean = CSP_META + clean;
  }
  const bi = clean.lastIndexOf("</body>");
  return bi !== -1
    ? clean.slice(0, bi) + ERR_OVERLAY + clean.slice(bi)
    : clean + ERR_OVERLAY;
}
