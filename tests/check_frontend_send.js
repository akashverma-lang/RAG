// Type into the composer and actually press Enter / click Send.
//
// The boot test fires all 66 handlers, but against empty inputs, so send() returned
// at its `if (!q) return` guard and everything past that guard was never executed.
// That is exactly where the ReferenceError lived. This drives the real path.
const fs = require("fs");
const vm = require("vm");
const path = require("path");

const html = fs.readFileSync(path.join(__dirname, "..", "frontend", "index.html"), "utf8");
const src = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const IDS = new Set([...html.matchAll(/\bid="([A-Za-z0-9_-]+)"/g)].map(x => x[1]));
const nodes = new Map();
const made = [];   // every element ever created, incl. chat bubbles

function el(tag, id, cls) {
  const kids = [];
  const set = new Set((cls || "").split(/\s+/).filter(Boolean));
  const n = {
    tagName: (tag || "div").toUpperCase(), id: id || "", dataset: {},
    _style: {}, className: cls || "", textContent: "", innerHTML: "", value: "",
    disabled: false, checked: false, tabIndex: 0, scrollHeight: 30,
    offsetWidth: 90, offsetLeft: 0, offsetHeight: 24, clientWidth: 1440, clientHeight: 800,
    scrollTop: 0, children: kids,
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
  n.style = new Proxy(n._style, {
    get: (t, k) => (k === "setProperty" || k === "removeProperty" ? () => {} :
                    k === "getPropertyValue" ? () => "" : (t[k] === undefined ? "" : t[k])),
    set: (t, k, v) => (t[k] = v, true),
  });
  const proxy = new Proxy(n, {
    set(t, k, v) {
      if (typeof k === "string" && k.startsWith("on") && typeof v === "function")
        (t._h ||= {})[k.slice(2)] = v;
      t[k] = v; return true;
    },
  });
  made.push(proxy);
  return proxy;
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
  querySelectorAll: () => [],
  getElementById: id => (id === "skull" ? null : IDS.has(id) ? pick("#" + id) : null),
  createElement: t => el(t), addEventListener(t, fn) { (this._h ||= {})[t] = fn; },
  // Deliberately absent, so CONSOLE falls back and the paint loop is uncontested.
  removeEventListener() {}, createTextNode: () => el("text"),
};

// The stream the server would send for a greeting: no sources, just tokens.
const SSE = ['data: {"type":"status","text":"thinking"}',
             'data: {"type":"token","text":"Hello"}',
             'data: {"type":"token","text":" there."}',
             'data: {"type":"done","took":0.4,"route":"chat"}'].join("\n\n") + "\n\n";

const store = {};
const ctx = {
  console: { log(){}, warn(){}, error(){} }, document: doc, window: null,
  localStorage: { getItem: k => store[k] ?? null, setItem: (k, v) => { store[k] = String(v); },
                  removeItem: k => { delete store[k]; } },
  location: { href: "http://127.0.0.1:8000/", origin: "http://127.0.0.1:8000",
              hash: "", pathname: "/", search: "" },
  history: { pushState() {}, replaceState() {}, state: null },
  // A realistic health payload, so the probe exercises the real footer path instead
  // of an artifact of an empty stub.
  fetch: () => Promise.resolve({
    ok: true, status: 200, text: () => Promise.resolve(""),
    json: () => Promise.resolve({
      llm: { model: "openai/gpt-oss-120b", model_ready: true, online: true,
             models: [], providers: [] },
      data_dir: "D:\data", data_dir_exists: true, files: 18, chunks: 147 }),
    body: { getReader(){ let done = false; return { read(){
      if (done) return Promise.resolve({ done: true, value: undefined });
      done = true;
      return Promise.resolve({ done: false, value: Buffer.from(SSE, "utf8") });
    }, releaseLock(){} }; } },
  }),
  setTimeout: (f) => { try { typeof f === "function" && f(); } catch {} return 0; },
  clearTimeout() {}, setInterval: () => 0, clearInterval() {},
  // Deferred, not synchronous: a synchronous rAF would recurse forever on any
  // self-rescheduling loop. Capped so the probe always terminates.
  requestAnimationFrame: (f) => { if (rafs++ < 300)
    setImmediate(() => { try { f(0); } catch (e) { uncaught.push(e); } }); return 0; },
  cancelAnimationFrame() {},
  AbortController: class { constructor(){ this.signal = {}; } abort(){} },
  TextDecoder: class { decode(b){ return b ? Buffer.from(b).toString("utf8") : ""; } },
  ResizeObserver: class { observe(){} disconnect(){} },
  MutationObserver: class { observe(){} disconnect(){} },
  matchMedia: () => ({ matches: false, addEventListener(){}, addListener(){} }),
  navigator: { clipboard: { writeText: () => Promise.resolve() } },
  URL: { createObjectURL: () => "blob:x", revokeObjectURL(){} }, Blob: class {},
  getComputedStyle: () => ({ getPropertyValue: () => "" }),
  devicePixelRatio: 2, innerWidth: 1440, innerHeight: 900,
  performance: { now: () => 0 },
  addEventListener(t, fn) { (this._h ||= {})[t] = fn; }, removeEventListener() {},
  alert(){}, confirm: () => true, prompt: () => null,
};
ctx.window = ctx; ctx.globalThis = ctx;
vm.createContext(ctx);

let bad = 0, rafs = 0;
const uncaught = [];
process.on("unhandledRejection", e => uncaught.push(e));

try { vm.runInContext(src, ctx, { timeout: 8000 }); }
catch (e) { console.log("FAIL  boot threw: " + e.message); process.exit(1); }

function attempt(label, fire) {
  const q = pick("#q");
  q.value = "Hi";
  try {
    const r = fire(q);
    if (r && typeof r.catch === "function") r.catch(e => uncaught.push(e));
  } catch (e) {
    console.log(`  FAIL  ${label} threw synchronously: ${e.constructor.name}: ${e.message}`);
    bad++; return;
  }
  console.log(`  ok    ${label} ran`);
}

attempt("Enter in the composer", q => q._h.keydown({ key: "Enter", shiftKey: false,
                                                    preventDefault(){}, stopPropagation(){} }));
attempt("clicking Send", () => pick("#sendBtn")._h.click({ preventDefault(){} }));

// Anything that threw inside the async body lands here rather than at the call site.
setTimeout(() => {
  for (const e of uncaught) {
    console.log(`  FAIL  rejected later: ${e && e.constructor ? e.constructor.name : "?"}: `
                + `${e && e.message}`);
    bad++;
  }
  // The tokens must actually reach the bubble, not merely stream without throwing.
  const painted = made.map(n => String(n.innerHTML || ""))
                    .filter(h => h.includes("Hello there."));
  if (painted.length) console.log("  ok    the answer was rendered into the bubble");
  else { console.log("  FAIL  the stream finished but nothing was painted"); bad++; }
  console.log("");
  console.log(bad ? `  ${bad} PROBLEM(S)`
                  : "  the composer sends and the answer renders");
  process.exit(bad ? 1 : 0);
}, 300);
