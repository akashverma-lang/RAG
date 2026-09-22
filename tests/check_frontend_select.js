// Dropdown lists have to stay readable in both themes.
//
// The bug this exists to stop coming back: #dashbar select -- the Analyse page's
// file picker -- set a background but never a colour, so `select{color:inherit}`
// handed its options the page's own --fg, #f0ede9. The list itself is drawn by the
// operating system, which knew nothing about our dark ground and painted it light,
// so near-white filenames landed on a near-white list. Measured below at 1.09:1,
// which is what "the names are almost transparent" looks like as a number.
//
// Two declarations fix it and both are checked here, because neither is enough
// alone: `color-scheme` asks the browser to draw the list dark, and explicit
// option colours hold the line on browsers that ignore it.
const fs = require("fs");
const path = require("path");

const html = fs.readFileSync(path.join(__dirname, "..", "frontend", "index.html"), "utf8");
const css = (html.match(/<style>([\s\S]*?)<\/style>/) || [])[1] || "";
const clean = css.replace(/\/\*[\s\S]*?\*\//g, "");

// Declarations for a selector. Collects every matching rule and anchors at a rule
// boundary, so "select" cannot match the tail of "#dashbar select".
function rule(selector) {
  const esc = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp("(?:^|[}\\n;])\\s*" + esc + "\\s*\\{([^}]*)\\}", "g");
  const found = [];
  let m;
  while ((m = re.exec(clean)) !== null) found.push(m[1]);
  return found.length ? found.join(";").replace(/\s+/g, " ").trim() : null;
}

function decl(body, prop) {
  if (!body) return null;
  const re = new RegExp("(?:^|;)\\s*" + prop + "\\s*:\\s*([^;]+)", "i");
  const m = body.match(re);
  return m ? m[1].trim() : null;
}

// --- colour ---------------------------------------------------------------------
function tokens(block) {
  const out = {};
  const re = /--([a-z0-9-]+)\s*:\s*([^;]+)/gi;
  let m;
  while ((m = re.exec(block)) !== null) out["--" + m[1]] = m[2].trim();
  return out;
}

const darkTokens = tokens(rule(":root") || "");
const lightTokens = Object.assign({}, darkTokens, tokens(rule('html[data-theme="light"]') || ""));

function resolve(value, tok) {
  // A missing declaration is the failure this file exists to catch, so it has to
  // arrive as a readable FAIL rather than String(null) becoming the truthy "null"
  // and crashing the contrast maths further down.
  if (!value) return null;
  let v = String(value).trim();
  for (let i = 0; i < 5 && v.startsWith("var("); i++) {
    const name = v.slice(4, v.indexOf(")")).split(",")[0].trim();
    if (!tok[name]) return null;
    v = tok[name].trim();
  }
  return v;
}

function rgb(hex) {
  const h = String(hex).trim().replace("#", "");
  if (!/^[0-9a-f]{6}$/i.test(h)) return null;
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
}

function luminance(c) {
  const a = c.map((v) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * a[0] + 0.7152 * a[1] + 0.0722 * a[2];
}

function contrast(fg, bg) {
  const a = luminance(rgb(fg));
  const b = luminance(rgb(bg));
  const [hi, lo] = a > b ? [a, b] : [b, a];
  return (hi + 0.05) / (lo + 0.05);
}

// --- checks ---------------------------------------------------------------------
let bad = 0;
function ok(label, cond, detail) {
  console.log(`${cond ? "PASS" : "FAIL"}  ${label}${detail ? "   " + detail : ""}`);
  if (!cond) bad++;
}

const optRule = rule("select option,select optgroup") || rule("select option") || "";
const optColor = decl(optRule, "color");
const optBg = decl(optRule, "background") || decl(optRule, "background-color");

ok("the option list names its own text colour", !!optColor, optColor || "missing");
ok("...and its own background, rather than the platform's", !!optBg, optBg || "missing");

// The whole point: legible in both themes, not merely declared.
for (const [name, tok] of [["dark", darkTokens], ["light", lightTokens]]) {
  const fg = resolve(optColor, tok);
  const bg = resolve(optBg, tok);
  const r = fg && bg ? contrast(fg, bg) : 0;
  ok(`filenames are readable in the ${name} theme`,
     r >= 4.5, `${fg} on ${bg} = ${r.toFixed(2)}:1`);
}

// color-scheme is what stops the browser drawing a light popup under dark text.
ok("select asks the browser for a matching popup",
   /dark|light/.test(decl(rule("select"), "color-scheme") || ""),
   decl(rule("select"), "color-scheme") || "missing");
ok("...and flips it with the light theme",
   /light/.test(decl(rule('html[data-theme="light"] select'), "color-scheme") || ""),
   decl(rule('html[data-theme="light"] select'), "color-scheme") || "missing");

// The regression itself: this select set a background but no colour.
ok("the Analyse file picker names a text colour",
   !!decl(rule("#dashbar select"), "color"),
   decl(rule("#dashbar select"), "color") || "missing");

// Scoping one file tints the closed control accent. Inherited onto the list, that
// paints every filename accent-on-light.
const scoped = rule("#anaTable.scoped");
if (scoped && /(^|;)\s*color\s*:/.test(scoped)) {
  ok("the scoped tint is kept off the option list",
     !!rule("#anaTable.scoped option"),
     rule("#anaTable.scoped option") || "no override");
}

console.log(bad ? `\n${bad} check(s) failed` : "\ndropdown lists stay readable in both themes");
process.exit(bad ? 1 : 0);
