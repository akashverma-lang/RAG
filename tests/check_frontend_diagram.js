// The flowchart renderer: does it draw, does it refuse, does it stay sane.
//
// Runs the page's own pure renderers in a VM, the same slice check_frontend.js uses,
// so this tests the shipped code rather than a copy of it.
const fs = require("fs");
const vm = require("vm");
const path = require("path");

const html = fs.readFileSync(path.join(__dirname, "..", "frontend", "index.html"), "utf8");
const src = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const pure = src.slice(src.indexOf("function esc"), src.indexOf("const stream = "));

const ctx = { console: { log(){}, warn(){}, error(){} } };
vm.createContext(ctx);
try { vm.runInContext(pure, ctx); }
catch (e) { console.log("FAIL  the renderers do not parse: " + e.message); process.exit(1); }

const run = (fn, arg) => vm.runInContext(`${fn}(__a)`, Object.assign(ctx, { __a: arg }));

let bad = 0;
const check = (name, ok, detail) => {
  console.log((ok ? "PASS  " : "FAIL  ") + name + (detail ? `   ${detail}` : ""));
  if (!ok) bad++;
};

// ---------------------------------------------------------------- draws
const flow = "flowchart TD\n  A[Ingest files] --> B{Spreadsheet?}\n"
           + "  B -->|yes| C[(SQL tables)]\n  B -->|no| D[Chunk and embed]\n"
           + "  C --> E[Answer]\n  D --> E";
const svg = run("renderDiagram", flow);
check("a flowchart becomes svg", !!svg && svg.includes("<svg"));
check("every node is drawn", (svg.match(/class="dia-node/g) || []).length === 5,
      `${(svg.match(/class="dia-node/g) || []).length} of 5`);
check("every edge is drawn", (svg.match(/class="dia-edge/g) || []).length === 5,
      `${(svg.match(/class="dia-edge/g) || []).length} of 5`);
check("edge labels are kept", svg.includes(">yes<") && svg.includes(">no<"));
check("a decision gets the diamond", svg.includes("dia-node diamond"));
check("geometry is finite", !/NaN|Infinity|undefined/.test(svg));

// mermaid writes a datastore as A[(name)]; the parens must not end up in the label.
check("a datastore label loses its parens",
      svg.includes(">SQL tables<") && !svg.includes(">(SQL tables)<"));

// ---------------------------------------------------------------- refuses
check("a non-flowchart is left alone",
      run("renderDiagram", "sequenceDiagram\n  A->>B: hi") === null);
check("prose is left alone", run("renderDiagram", "just some words") === null);
check("an empty block is left alone", run("renderDiagram", "") === null);

// ---------------------------------------------------------------- shape
// A long left-to-right chain is unreadable at the width of a reading column, so it
// is drawn downward however it was authored.
const chain = "graph LR\n" + Array.from({ length: 7 },
  (_, i) => `  N${i}[Step number ${i}] --> N${i + 1}[Step number ${i + 1}]`).join("\n");
const tall = run("renderDiagram", chain);
const vb = (tall.match(/viewBox="[^ ]+ [^ ]+ ([\d.]+) ([\d.]+)"/) || []);
check("a long LR chain is drawn top-down instead",
      Number(vb[2]) > Number(vb[1]), `${vb[1]} x ${vb[2]}`);

// ---------------------------------------------------------------- through markdown
const md = run("renderMd", "Before.\n\n```mermaid\n" + flow + "\n```\n\nAfter.");
check("renderMd swaps the fence for the drawing",
      md.includes("<svg") && !md.includes("<code data-lang=\"mermaid\""));

// A fence that has not finished streaming must stay a code block: re-laying out a
// half-written graph on every animation frame flickers through nonsense.
const partial = run("renderMd", "Before.\n\n```mermaid\nflowchart TD\n  A[Ingest] --> B{Spread");
check("an unterminated fence stays a code block",
      partial.includes("<pre><code") && !partial.includes("<svg"));

// ---------------------------------------------------------------- hostile input
check("a cycle terminates", !!run("renderDiagram", "flowchart TD\n A[x] --> B[y]\n B --> A"));
check("a self-loop terminates", !!run("renderDiagram", "flowchart TD\n A[x] --> A"));
const huge = "flowchart TD\n" + Array.from({ length: 300 },
  (_, i) => `  A${i}[n${i}] --> A${i + 1}[n${i + 1}]`).join("\n");
check("an oversized graph is refused rather than drawn",
      run("renderDiagram", huge) === null);
check("a label cannot inject markup",
      !run("renderDiagram", 'flowchart TD\n A[&lt;img src=x onerror=alert(1)&gt;] --> B[ok]')
        .includes("<img"));

console.log(bad ? `\n${bad} PROBLEM(S)` : "\nPASS  diagrams render, refuse and stay sane");
process.exit(bad ? 1 : 0);
