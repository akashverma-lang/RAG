// Execute the chat page's inline script against a stubbed DOM.
//
// check_frontend.js proves the script PARSES and that the markdown renderer is
// correct. It cannot catch a reference to a function that was renamed, an element
// id that no longer exists, or a handler wired to a null - all of which throw only
// when the page actually boots, and all of which have happened here.
//
// So this runs the whole top-level script: every `$("#id").onclick = ...`, every
// querySelectorAll walk, every observer. The stub is deliberately dumb; it only has
// to be real enough that a genuine mistake throws.
const fs = require("fs");
const vm = require("vm");
const path = require("path");

const file = path.join(__dirname, "..", "frontend", "index.html");
const html = fs.readFileSync(file, "utf8");
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) { console.error("FAIL  no inline script found"); process.exit(1); }

// ---------------------------------------------------------------- the ids and
// classes the markup actually declares, so a typo in either direction is caught.
const IDS = new Set([...html.matchAll(/\bid="([A-Za-z0-9_-]+)"/g)].map(x => x[1]));
const CLASSES = new Set();
for (const c of html.matchAll(/\bclass="([^"]+)"/g))
  c[1].split(/\s+/).forEach(n => n && CLASSES.add(n));

// The tab strip is driven by data-tab, so the stub has to carry the real values or
// selectTab looks up "#tab-undefined" and reports a bug that does not exist.
const TAB_KEYS = [...html.matchAll(/data-tab="([A-Za-z0-9_-]+)"/g)].map(x => x[1]);

const missing = new Set();

// Every handler the page wires up, so they can be fired after boot. Assigning a
// handler proves nothing; calling it is what catches a reference to a function that
// was renamed out from under it.
const HANDLERS = [];

function makeEl(tag, id) {
  const kids = [];
  const el = {
    tagName: (tag || "div").toUpperCase(),
    id: id || "",
    dataset: {},
    style: new Proxy({ setProperty() {}, removeProperty() {},
                       getPropertyValue: () => "" }, {
      get: (t, k) => (t[k] === undefined ? "" : t[k]),
      set: (t, k, v) => (t[k] = v, true) }),
    className: "",
    textContent: "",
    innerHTML: "",
    value: "",
    checked: false,
    disabled: false,
    tabIndex: 0,
    offsetWidth: 80, offsetLeft: 0, offsetHeight: 20,
    scrollHeight: 100, scrollTop: 0, clientHeight: 100,
    children: kids,
    classList: {
      _s: new Set(),
      add(...n) { n.forEach(x => this._s.add(x)); },
      remove(...n) { n.forEach(x => this._s.delete(x)); },
      toggle(n, f) { const on = f === undefined ? !this._s.has(n) : !!f;
                     on ? this._s.add(n) : this._s.delete(n); return on; },
      contains(n) { return this._s.has(n); },
    },
    setAttribute() {}, getAttribute() { return null; }, removeAttribute() {},
    appendChild(c) { kids.push(c); return c; },
    append(...c) { kids.push(...c); },
    prepend(c) { kids.unshift(c); return c; },
    replaceWith() {}, remove() {}, insertBefore(c) { kids.push(c); return c; },
    addEventListener() {}, removeEventListener() {},
    focus() {}, blur() {}, click() {}, scrollIntoView() {},
    closest() { return null; },
    querySelector(sel) { return makeEl("div"); },
    querySelectorAll() { return []; },
    getBoundingClientRect() { return { top: 0, left: 0, width: 80, height: 20 }; },
    // Enough of a 2D context that the vector field actually runs here rather than
    // being skipped: a background that throws on boot takes the page with it.
    getContext: () => ({
      setTransform() {}, clearRect() {}, beginPath() {}, arc() {}, fill() {},
      stroke() {}, moveTo() {}, lineTo() {}, fillText() {}, save() {}, restore() {},
      closePath() {}, rect() {}, fillRect() {}, strokeRect() {}, translate() {},
      rotate() {}, scale() {}, createLinearGradient: () => ({ addColorStop() {} }),
      createRadialGradient: () => ({ addColorStop() {} }),
      measureText: () => ({ width: 10 }),
      set fillStyle(v) {}, set strokeStyle(v) {}, set lineWidth(v) {},
      set font(v) {}, set globalAlpha(v) {},
    }),
  };
  return new Proxy(el, {
    set(t, k, v) {
      if (typeof k === "string" && k.startsWith("on") && typeof v === "function")
        HANDLERS.push({ name: k + (id ? " on #" + id : ""), fn: v });
      t[k] = v;
      return true;
    },
  });
}

function resolve(sel) {
  const s = String(sel).trim();
  const idHit = s.match(/^#([A-Za-z0-9_-]+)$/);
  if (idHit) {
    if (!IDS.has(idHit[1])) { missing.add("#" + idHit[1]); return null; }
    return makeEl("div", idHit[1]);
  }
  const clsHit = s.match(/^\.([A-Za-z0-9_-]+)$/);
  if (clsHit && !CLASSES.has(clsHit[1])) { missing.add("." + clsHit[1]); return null; }
  return makeEl("div");
}

const doc = {
  documentElement: makeEl("html"),
  body: makeEl("body"),
  activeElement: makeEl("div"),
  querySelector: resolve,
  querySelectorAll(sel) {
    const s = String(sel).trim();
    const cls = s.match(/^\.([A-Za-z0-9_-]+)$/);
    if (cls && !CLASSES.has(cls[1])) { missing.add("." + cls[1]); return []; }
    if (s === ".tab" && TAB_KEYS.length)
      return TAB_KEYS.map(k => { const e = makeEl("button"); e.dataset.tab = k; return e; });
    // Enough elements that index arithmetic and wrap-around get exercised.
    return [makeEl("div"), makeEl("div"), makeEl("div")];
  },
  getElementById(id) { return IDS.has(id) ? makeEl("div", id) : null; },
  createElement: t => makeEl(t),
  addEventListener() {}, removeEventListener() {},
  createTextNode: () => makeEl("text"),
};
doc.documentElement.dataset = {};

const store = {};
const ctx = {
  console,
  document: doc,
  window: null,
  localStorage: {
    getItem: k => (k in store ? store[k] : null),
    setItem: (k, v) => { store[k] = String(v); },
    removeItem: k => { delete store[k]; },
  },
  location: { href: "http://127.0.0.1:8000/", origin: "http://127.0.0.1:8000",
              hash: "", pathname: "/", search: "", reload() {}, assign() {} },
  history: { pushState() {}, replaceState() {}, back() {}, forward() {}, state: null },
  fetch: () => Promise.resolve({ ok: true, status: 200, body: null,
                                 json: () => Promise.resolve({}),
                                 text: () => Promise.resolve("") }),
  setTimeout: () => 0, clearTimeout() {}, setInterval: () => 0, clearInterval() {},
  requestAnimationFrame: () => 0, cancelAnimationFrame() {},
  AbortController: class { constructor() { this.signal = {}; } abort() {} },
  TextDecoder: class { decode() { return ""; } },
  ResizeObserver: class { observe() {} disconnect() {} },
  MutationObserver: class { observe() {} disconnect() {} },
  matchMedia: () => ({ matches: false, addEventListener() {}, addListener() {} }),
  navigator: { clipboard: { writeText: () => Promise.resolve() }, userAgent: "node" },
  URL: { createObjectURL: () => "blob:x", revokeObjectURL() {} },
  Blob: class {},
  getComputedStyle: () => ({ getPropertyValue: () => "" }),
  devicePixelRatio: 2, innerWidth: 1440, innerHeight: 900,
  performance: { now: () => 0 },
  alert() {}, confirm: () => true, prompt: () => null,
  // The page calls these bare, as globals on window, which is legal in a browser.
  addEventListener() {}, removeEventListener() {}, dispatchEvent: () => true,
  scrollTo() {}, scrollY: 0, scrollX: 0,
};
ctx.window = ctx;
ctx.globalThis = ctx;
vm.createContext(ctx);

let bad = 0;
try {
  vm.runInContext(m[1], ctx, { timeout: 5000, filename: "index.html inline script" });
  console.log("PASS  the page boots (top-level script runs against a DOM)");
} catch (e) {
  console.error("FAIL  boot threw: " + (e && e.message));
  if (e && e.stack) console.error(e.stack.split("\n").slice(1, 4).join("\n"));
  bad++;
}

// Fire every handler. A stub this thin will make plenty of them throw for reasons
// that are the stub's fault, so only the errors that mean a real missing symbol are
// counted: a name that does not exist, or a call on something that is not callable.
let fired = 0, real = [];
for (const h of HANDLERS) {
  const ev = {
    preventDefault() {}, stopPropagation() {}, key: "Escape", target: makeEl("div"),
    currentTarget: makeEl("div"), clientX: 0, clientY: 0, shiftKey: false,
    dataTransfer: { files: [] },
  };
  try { h.fn(ev); fired++; }
  catch (e) {
    const msg = String((e && e.message) || e);
    if (e instanceof ReferenceError || /is not a function|is not defined/.test(msg))
      real.push(h.name + ": " + msg);
  }
}
if (real.length) {
  console.log("FAIL  handlers reference something that does not exist:");
  real.slice(0, 8).forEach(r => console.log("        " + r));
  bad++;
} else {
  console.log(`PASS  ${HANDLERS.length} handlers fire without a missing reference`);
}

if (missing.size) {
  console.log("FAIL  selectors with nothing to match in the markup: "
              + [...missing].join(", "));
  bad++;
} else {
  console.log("PASS  every id and class selector resolves against the markup");
}

// The whole point of this page is that it works with no network.
const external = [...html.matchAll(/(?:src|href)\s*=\s*["'](https?:\/\/[^"']+)/g)]
  .map(x => x[1])
  .concat([...html.matchAll(/@import\s+["']?(https?:\/\/[^"';]+)/g)].map(x => x[1]));
console.log(external.length
  ? "FAIL  page loads something external: " + external.join(", ")
  : "PASS  no external stylesheet, script or font is loaded");
if (external.length) bad++;

process.exit(bad ? 1 : 0);
