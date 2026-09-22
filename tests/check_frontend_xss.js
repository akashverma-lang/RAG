// Throw XSS payloads through the page's real renderers.
//
// Everything a model writes, and every snippet lifted out of a document, reaches
// innerHTML through these two functions. A document is untrusted input: it is
// whatever happened to land in the data folder.
//
// The detector scans real TAGS, not raw substrings. A first attempt flagged
// "&lt;img src=x onerror=alert(1)&gt;" as a hole, which is precisely the output that
// proves escaping worked -- it renders as visible text and executes nothing.
const fs = require("fs");
const vm = require("vm");
const path = require("path");

const html = fs.readFileSync(path.join(__dirname, "..", "frontend", "index.html"), "utf8");
const src = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const pure = src.slice(src.indexOf("function esc"), src.indexOf("const stream = "));
const ctx = { console: { log(){}, warn(){}, error(){} } };
vm.createContext(ctx);
vm.runInContext(pure, ctx);
const render = (fn, a) => vm.runInContext(`${fn}(__a)`, Object.assign(ctx, { __a: a }));

// Tags the renderers are supposed to emit. Anything else appearing as a real tag
// came from the payload.
const ALLOWED = new Set([
  "p", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li",
  "strong", "em", "del", "code", "pre", "blockquote", "table", "thead", "tbody",
  "tr", "th", "td", "a", "span", "div", "details", "summary", "button",
  // the diagram renderer's own output
  "svg", "defs", "marker", "path", "rect", "circle", "text", "line", "g",
]);

function findings(out) {
  const bad = [];
  const tag = /<\s*\/?\s*([a-zA-Z][\w:-]*)((?:"[^"]*"|'[^']*'|[^>"'])*)>/g;
  let m;
  while ((m = tag.exec(out)) !== null) {
    const name = m[1].toLowerCase();
    const attrs = m[2] || "";
    if (!ALLOWED.has(name)) bad.push(`unexpected <${name}>`);
    // A real event-handler attribute, i.e. one the PARSER would create. Entities do
    // not terminate an attribute: data-lang="&quot; onload=&quot;x" is one attribute
    // whose value happens to contain quotes, verified against a real browser parser.
    // Only literal quotes close an attribute, so entity text inside a value is data.
    const attrOnly = attrs.replace(/=\s*"[^"]*"/g, "=V").replace(/=\s*'[^']*'/g, "=V");
    const on = /(^|\s)on[a-z]+\s*=/i.exec(attrOnly);
    if (on) bad.push(`event handler in <${name}>: ${attrs.trim().slice(0, 40)}`);
    // A navigable attribute pointing somewhere executable.
    const url = /\b(href|src|xlink:href|action|formaction|data)\s*=\s*("([^"]*)"|'([^']*)'|([^\s>]+))/gi;
    let u;
    while ((u = url.exec(attrs)) !== null) {
      const val = (u[3] || u[4] || u[5] || "").trim().toLowerCase()
        .replace(/&#(\d+);/g, (_, d) => String.fromCharCode(+d));
      if (/^(javascript|data|vbscript):/.test(val))
        bad.push(`${u[1]}=${val.slice(0, 40)} in <${name}>`);
    }
  }
  return bad;
}

const PAYLOADS = [
  ["script tag",            '<script>alert(1)</script>'],
  ["img onerror",           '<img src=x onerror=alert(1)>'],
  ["svg onload",            '<svg onload=alert(1)>'],
  ["iframe",                '<iframe src="javascript:alert(1)"></iframe>'],
  ["js link",               '[click](javascript:alert(1))'],
  ["js link uppercase",     '[click](JaVaScRiPt:alert(1))'],
  ["js link entity",        '[click](java&#115;cript:alert(1))'],
  ["data url link",         '[click](data:text/html,<script>alert(1)</script>)'],
  ["attr break in link",    '[x](https://a.com" onmouseover="alert(1))'],
  ["code fence html",       '```\n<img src=x onerror=alert(1)>\n```'],
  ["fence lang injection",  '```" onload="alert(1)\nx\n```'],
  ["table cell",            '| a |\n| - |\n| <img src=x onerror=alert(1)> |'],
  ["table header",          '| <img src=x onerror=alert(1)> |\n| - |\n| b |'],
  ["heading",               '# <script>alert(1)</script>'],
  ["blockquote",            '> <img src=x onerror=alert(1)>'],
  ["list item",             '- <svg onload=alert(1)>'],
  ["bold wrapper",          '**<img src=x onerror=alert(1)>**'],
  ["inline code",           '`<img src=x onerror=alert(1)>`'],
  ["autolink-ish",          'https://a.com"><script>alert(1)</script>'],
  ["entity double-encode",  '&lt;script&gt;alert(1)&lt;/script&gt;'],
  ["broken tag",            '<scr<script>ipt>alert(1)</script>'],
  ["backtick attr",         '`x` <b onclick=alert(1)>y</b>'],
  ["cite marker forge",     '[1]<img src=x onerror=alert(1)>'],
  ["null byte",             '<img\u0000 src=x onerror=alert(1)>'],
];

const DIAGRAMS = [
  ["node label",      'flowchart TD\n  A[<img src=x onerror=alert(1)>] --> B[ok]'],
  ["edge label",      'flowchart TD\n  A[a] -->|<svg onload=alert(1)>| B[b]'],
  ["node id",         'flowchart TD\n  A"><script>alert(1)</script>[x] --> B[y]'],
  ["quote in label",  'flowchart TD\n  A[" onload="alert(1)] --> B[b]'],
  ["header line",     'flowchart TD<script>alert(1)</script>\n  A[a] --> B[b]'],
  ["marker id forge", 'flowchart TD\n  A[a"/><script>alert(1)</script>] --> B[b]'],
];

let bad = 0;
const scan = (label, out) => {
  const f = findings(out || "");
  if (f.length) { console.log(`  FAIL  ${label}  ->  ${f[0]}`); bad++; }
  else console.log(`  ok    ${label}`);
};

console.log("-".repeat(64));
console.log("markdown: what a model writes, and what a document contains");
console.log("-".repeat(64));
for (const [n, p] of PAYLOADS) scan(n.padEnd(22), render("renderMd", p));

console.log("\n" + "-".repeat(64));
console.log("diagrams");
console.log("-".repeat(64));
for (const [n, p] of DIAGRAMS) {
  scan((n + " direct").padEnd(22), (() => {
    try { return render("renderDiagram", p); } catch { return ""; }
  })());
  scan((n + " via fence").padEnd(22), render("renderMd", "```mermaid\n" + p + "\n```"));
}

console.log("\n" + "-".repeat(64));
console.log("a file name and a path, which the locate answer prints verbatim");
console.log("-".repeat(64));
scan("path in backticks     ", render("renderMd", 'Found at `D:\\data\\<img src=x onerror=alert(1)>.pdf`'));
scan("bold file name        ", render("renderMd", '**<script>alert(1)</script>.pdf** is in:'));
scan("path with quote       ", render("renderMd", 'Found at `D:\\a" onmouseover="alert(1)\\b.pdf`'));

console.log("");
console.log(bad ? `${bad} PROBLEM(S)` : "no payload produced executable markup");
process.exit(bad ? 1 : 0);
