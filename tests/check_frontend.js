// Parse the inline <script> of the chat page, and exercise the markdown renderer.
const fs = require("fs");
const vm = require("vm");

const html = fs.readFileSync("D:/RAG/frontend/index.html", "utf8");
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) { console.error("FAIL: no inline script found"); process.exit(1); }
const src = m[1];

// 1. syntax check
try { new vm.Script(src); console.log("PASS  inline script parses"); }
catch (e) { console.error("FAIL  syntax error:", e.message); process.exit(1); }

// 2. run the markdown renderer standalone (extract the pure functions)
const pure = src.slice(src.indexOf("function esc"), src.indexOf("const stream = "));
const ctx = { console };
vm.createContext(ctx);
vm.runInContext(pure, ctx);

const cases = [
  ["**bold** and *em*", "<strong>bold</strong>"],
  ["Revenue was 4.7M [1] and Berlin 1.2M [2].", 'class="cite" data-n="1"'],
  ["Total due 12450 EUR【1】", 'class="cite" data-n="1"'],
  ["# Title\n\ntext", "<h1>Title</h1>"],
  ["- one\n- two", "<li>one</li><li>two</li>"],
  ["1. first\n2. second", "<ol><li>first</li>"],
  ["| A | B |\n|---|---|\n| 1 | 2 |", "<table><thead>"],
  ["```py\nx = 1\n```", "<pre><code"],
  ["> quoted", "<blockquote>"],
  ["<script>alert(1)</script>", "&lt;script&gt;"],
  ["a | b", "<p>a | b</p>"],
];
let bad = 0;
for (const [input, expect] of cases) {
  const out = ctx.renderMd(input);
  const ok = out.includes(expect);
  if (!ok) { bad++; console.log("FAIL  " + JSON.stringify(input) + "\n      got " + out); }
}
console.log(bad ? `FAIL  ${bad}/${cases.length} markdown cases` : `PASS  ${cases.length} markdown cases`);

// 3. XSS: a citation must not let raw html through
const xss = ctx.renderMd('<img src=x onerror="alert(1)"> [1]');
console.log(xss.includes("<img") ? "FAIL  raw html leaked" : "PASS  html is escaped");
if (xss.includes("<img")) bad++;

// 4. Streaming safety.
//
// paint() re-renders the whole answer on every animation frame, so renderMd sees
// each *prefix* of the text, including ones that end mid-construct. If any prefix
// makes the loop stop advancing, the browser tab locks up entirely - that is not a
// rendering glitch, it is Chrome's "page unresponsive" dialog. A markdown table
// used to do exactly that: the header row arrives one frame before the |---| line
// under it, and nothing consumed it.
const STREAMED = [
  "Segment PG has 175,447 claims.\n\n| segment | claims |\n|---|---|\n| PG | 175,447 |\n| IND | 24,528 |",
  "| a |\n| b |\n| c |",
  "Totals:\n\n| zone | amount | claims | status | date |\n|---|---|---|---|---|\n| East | 1.2 | 3 | ok | 2025-10-01 |",
  "```sql\nSELECT * FROM t\n```\n\ndone",
  "```unterminated fence\nstill going",
  "# Heading\n\n- one\n- two\n\n> quote\n\n---\n\n1. a\n2. b",
  "a | b | c\nplain text with | pipes | inside",
  "|",
  "|||\n|",
  "> quote\n| table |\n# head\n- list\n1. num\n```",
  "text\n\n\n\n| x |",
];
let hangs = 0, checked = 0;
for (const full of STREAMED) {
  for (let n = 1; n <= full.length; n++) {
    checked++;
    try {
      vm.runInContext("renderMd(__p)", Object.assign(ctx, { __p: full.slice(0, n) }),
                      { timeout: 500 });
    } catch (e) {
      hangs++;
      console.log(`FAIL  renderMd stalled on prefix ${n} of ${JSON.stringify(full.slice(0, 60))}`);
      break;
    }
  }
}
bad += hangs;
console.log(hangs ? `FAIL  ${hangs} streaming prefix(es) hang the renderer`
                  : `PASS  ${checked} streamed prefixes render without stalling`);

process.exit(bad ? 1 : 0);
