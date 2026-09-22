// Does the point cloud get a canvas with a real size on a plain page load?
//
// The console starts display:none and is switched on by the router. An element that
// is display:none reports clientWidth/clientHeight of 0, so anything that measures
// it before the router runs sizes the canvas to 0x0 and draws nothing.
//
// Headless Chrome hides this: it fires a resize during startup, which re-measures.
// A real browser load does not, so this models the element honestly instead.
const fs = require("fs");
const vm = require("vm");
const path = require("path");

const html = fs.readFileSync(path.join(__dirname, "..", "frontend", "index.html"), "utf8");
const src = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const IDS = new Set([...html.matchAll(/\bid="([A-Za-z0-9_-]+)"/g)].map(x => x[1]));
const nodes = new Map();
let resizeFired = false;

function el(tag, id, cls) {
  const kids = [];
  const set = new Set((cls || "").split(/\s+/).filter(Boolean));
  const n = {
    tagName: (tag || "div").toUpperCase(), id: id || "", dataset: {},
    _style: {}, className: cls || "", textContent: "", innerHTML: "", value: "",
    disabled: false, checked: false, tabIndex: 0, scrollHeight: 30,
    offsetWidth: 90, offsetLeft: 0, offsetHeight: 24, scrollTop: 0, children: kids,
    width: 0, height: 0,
    classList: {
      add: (...a) => a.forEach(x => set.add(x)), remove: (...a) => a.forEach(x => set.delete(x)),
      toggle: (x, f) => { const on = f === undefined ? !set.has(x) : !!f;
                          on ? set.add(x) : set.delete(x); return on; },
      contains: x => set.has(x), _set: set,
    },
    setAttribute(k, v) { this["attr_" + k] = v; }, getAttribute(k) { return this["attr_" + k] ?? null; },
    removeAttribute() {}, appendChild(c) { kids.push(c); return c; },
    append(...c) { kids.push(...c); }, prepend(c) { kids.unshift(c); return c; },
    replaceWith() {}, remove() {}, insertBefore(c) { kids.push(c); return c; },
    addEventListener(t, fn) { (this._h ||= {})[t] = fn; },
    removeEventListener() {}, focus() {}, blur() {}, click() {}, scrollIntoView() {},
    closest() { return null; }, querySelector() { return el("div"); }, querySelectorAll: () => [],
    getBoundingClientRect: () => ({ top: 0, left: 0, width: 90, height: 24 }),
    getContext: () => new Proxy({}, { get: () => () => ({ addColorStop() {} }) }),
  };
  // The honest part: a hidden box has no size. #console is only shown once the
  // router adds .on, exactly as the real stylesheet has it.
  Object.defineProperty(n, "clientWidth", {
    get() { return (this.id === "console" && !set.has("on")) ? 0 : 1440; },
  });
  Object.defineProperty(n, "clientHeight", {
    get() { return (this.id === "console" && !set.has("on")) ? 0 : 800; },
  });
  n.style = new Proxy(n._style, {
    get: (t, k) => (k === "setProperty" || k === "removeProperty" ? () => {} :
                    k === "getPropertyValue" ? () => "" : (t[k] === undefined ? "" : t[k])),
    set: (t, k, v) => (t[k] = v, true),
  });
  return new Proxy(n, {
    set(t, k, v) {
      if (typeof k === "string" && k.startsWith("on") && typeof v === "function")
        (t._h ||= {})[k.slice(2)] = v;
      t[k] = v; return true;
    },
  });
}
function pick(sel) {
  sel = String(sel).trim();
  if (nodes.has(sel)) return nodes.get(sel);
  const id = sel.match(/^#([A-Za-z0-9_-]+)$/);
  if (id && !IDS.has(id[1])) return null;
  const n = el("div", id ? id[1] : ""); nodes.set(sel, n); return n;
}
const body = el("body"), docEl = el("html"); docEl.dataset = {};
const doc = {
  documentElement: docEl, body, activeElement: el("div"), querySelector: pick,
  querySelectorAll: () => [], getElementById: id => (IDS.has(id) ? pick("#" + id) : null),
  createElement: t => el(t), addEventListener(t, fn) { (this._h ||= {})[t] = fn; },
  removeEventListener() {}, createTextNode: () => el("text"),
};

// Anything that observes an element for size changes gets told when one happens.
const observers = [];
const store = {};
const ctx = {
  console: { log(){}, warn(){}, error(){} }, document: doc, window: null,
  localStorage: { getItem: k => store[k] ?? null, setItem: (k, v) => { store[k] = String(v); },
                  removeItem: k => { delete store[k]; } },
  location: { href: "http://x/", origin: "http://x", hash: "", pathname: "/", search: "" },
  history: { pushState() {}, replaceState() {}, state: null },
  fetch: () => new Promise(() => {}),        // health never resolves; irrelevant here
  setTimeout: () => 0, clearTimeout() {}, setInterval: () => 0, clearInterval() {},
  requestAnimationFrame: () => 1, cancelAnimationFrame() {},
  AbortController: class { constructor(){ this.signal = {}; } abort(){} },
  TextDecoder: class { decode(){ return ""; } },
  ResizeObserver: class {
    constructor(fn){ this.fn = fn; observers.push(this); }
    observe(target){ (this.targets ||= []).push(target); }
    disconnect(){}
  },
  MutationObserver: class { observe(){} disconnect(){} },
  matchMedia: () => ({ matches: false, addEventListener(){}, addListener(){} }),
  navigator: { clipboard: { writeText: () => Promise.resolve() } },
  URL: { createObjectURL: () => "blob:x", revokeObjectURL(){} }, Blob: class {},
  getComputedStyle: () => ({ getPropertyValue: () => "" }),
  devicePixelRatio: 2, innerWidth: 1440, innerHeight: 900,
  performance: { now: () => 0 },
  addEventListener(t, fn) {
    if (t === "resize") resizeFired = true;   // registered, but never dispatched
    (this._h ||= {})[t] = fn;
  },
  removeEventListener() {},
  alert(){}, confirm: () => true, prompt: () => null,
};
ctx.window = ctx; ctx.globalThis = ctx;
vm.createContext(ctx);

try { vm.runInContext(src, ctx, { timeout: 8000 }); }
catch (e) { console.log("FAIL  boot threw: " + e.message); process.exit(1); }

// Deliberately NO resize event: that is the crutch headless Chrome provides and a
// real page load does not.
const consoleEl = pick("#console");
const cv = pick("#skull");

// A ResizeObserver watching the console is a legitimate fix, so let it fire the way
// a browser would when the box goes from hidden to shown.
for (const o of observers) {
  if ((o.targets || []).some(t => t && t.id === "console")) {
    try { o.fn([{ target: consoleEl }], o); } catch {}
  }
}

let bad = 0;
console.log(`  console shown after boot   ${consoleEl.classList.contains("on")}`);
console.log(`  canvas pixels              ${cv.width} x ${cv.height}`);
console.log(`  a resize was never fired   ${resizeFired ? "(listener exists, not dispatched)" : "(no listener)"}`);

if (!consoleEl.classList.contains("on")) {
  console.log("  FAIL  the console never became visible"); bad++;
}
if (!(cv.width > 0 && cv.height > 0)) {
  console.log("  FAIL  the canvas has no pixels, so the cloud draws nothing");
  bad++;
}
console.log("");
console.log(bad ? `  ${bad} PROBLEM(S)` : "  the cloud gets a real canvas on a plain load");
process.exit(bad ? 1 : 0);
