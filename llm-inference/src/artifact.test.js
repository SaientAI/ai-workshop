// artifact.test.js — run with: node src/artifact.test.js
// Inline copies of the parser functions so this file is self-contained.

function splitArtifact(text) {
  const openIdx = text.indexOf('<artifact');
  if (openIdx === -1) return { chatText: text, artifact: null };
  const tagEnd = text.indexOf('>', openIdx);
  if (tagEnd === -1) return { chatText: text.substring(0, openIdx).trim(), artifact: null };
  const tag = text.substring(openIdx, tagEnd + 1);
  const typeM  = tag.match(/type="([^"]*)"/i);
  const titleM = tag.match(/title="([^"]*)"/i);
  const type  = typeM  ? typeM[1]  : 'html';
  const title = titleM ? titleM[1] : 'Artifact';
  const chatBefore = text.substring(0, openIdx).trim();
  const afterOpen  = text.substring(tagEnd + 1);
  const closeMatch = afterOpen.match(/<;?\/artifact\s*>/i);
  const closeIdx = closeMatch ? closeMatch.index : -1;
  if (closeIdx === -1) {
    return { chatText: chatBefore, artifact: { type, title, content: afterOpen, complete: false } };
  }
  const content   = afterOpen.substring(0, closeIdx);
  const chatAfter = afterOpen.substring(closeIdx + closeMatch[0].length).trim();
  const chatText  = [chatBefore, chatAfter].filter(Boolean).join('\n\n');
  return { chatText, artifact: { type, title, content, complete: true } };
}

function stripArtifactTag(html) {
  html = html.replace(/<;\/(\w+)/g, '</$1');
  html = html.replace(/&lt;\/artifact[^&>]*(?:&gt;)?/gi, '');
  const idx = html.toLowerCase().indexOf('</artifact');
  return idx === -1 ? html : html.substring(0, idx).trimEnd();
}

// ── Test harness ──────────────────────────────────────────────────────────────
let passed = 0, failed = 0;
function test(name, fn) {
  try {
    fn();
    console.log(`  ✓ ${name}`);
    passed++;
  } catch (e) {
    console.error(`  ✗ ${name}: ${e.message}`);
    failed++;
  }
}
function assert(cond, msg) { if (!cond) throw new Error(msg || 'assertion failed'); }
function assertEqual(a, b) {
  if (a !== b) throw new Error(`expected ${JSON.stringify(b)}, got ${JSON.stringify(a)}`);
}

// ── splitArtifact ─────────────────────────────────────────────────────────────
console.log('\nsplitArtifact');

test('no artifact tag → artifact is null', () => {
  const { chatText, artifact } = splitArtifact('Hello world');
  assertEqual(chatText, 'Hello world');
  assertEqual(artifact, null);
});

test('complete artifact extracted', () => {
  const input = 'Here it is:\n<artifact type="html" title="Demo"><h1>hi</h1></artifact>';
  const { chatText, artifact } = splitArtifact(input);
  assertEqual(chatText, 'Here it is:');
  assertEqual(artifact.type, 'html');
  assertEqual(artifact.title, 'Demo');
  assert(artifact.content.includes('<h1>hi</h1>'), 'content missing');
  assertEqual(artifact.complete, true);
});

test('partial artifact (no close tag) marked incomplete', () => {
  const input = '<artifact type="html" title="WIP"><script>let x = ';
  const { artifact } = splitArtifact(input);
  assert(artifact !== null);
  assertEqual(artifact.complete, false);
  assert(artifact.content.includes('<script>'), 'partial content missing');
});

test('malformed close tag <;/artifact> accepted', () => {
  const input = '<artifact type="html" title="T"><b>bold</b><;/artifact>';
  const { artifact } = splitArtifact(input);
  assert(artifact !== null);
  assertEqual(artifact.complete, true);
});

test('artifact without title defaults to Artifact', () => {
  const { artifact } = splitArtifact('<artifact type="html"><p>x</p></artifact>');
  assertEqual(artifact.title, 'Artifact');
});

test('artifact without type defaults to html', () => {
  const { artifact } = splitArtifact('<artifact title="T"><p>x</p></artifact>');
  assertEqual(artifact.type, 'html');
});

// ── stripArtifactTag ──────────────────────────────────────────────────────────
console.log('\nstripArtifactTag');

test('repairs <;/style> to </style>', () => {
  const result = stripArtifactTag('<style>body{}<;/style><p>hi</p>');
  assert(result.includes('</style>'), 'not repaired: ' + result);
  assert(!result.includes('<;/'), 'semicolon not removed: ' + result);
});

test('truncates at </artifact>', () => {
  const html = '<h1>hi</h1></artifact><p>should be gone</p>';
  const result = stripArtifactTag(html);
  assert(result.includes('<h1>hi</h1>'), 'content lost: ' + result);
  assert(!result.includes('should be gone'), 'not truncated: ' + result);
});

test('removes HTML-encoded artifact closing tag', () => {
  const html = '<p>content</p>&lt;/artifact&gt;';
  const result = stripArtifactTag(html);
  assert(!result.includes('&lt;/artifact'), 'encoded tag not removed: ' + result);
});

test('no-op on clean HTML', () => {
  const html = '<html><body><h1>OK</h1></body></html>';
  const result = stripArtifactTag(html);
  assertEqual(result, html);
});

// ── Summary ───────────────────────────────────────────────────────────────────
console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
