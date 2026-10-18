// Shared DOM and formatting helpers for the dashboard and the presentation stage.
// Loaded text is always inserted with textContent / createTextNode, never as HTML.

export const $ = (id) => document.getElementById(id);

export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "style") {
      // Custom properties (--l, --w, --ink-ms) need setProperty; Object.assign drops them silently.
      for (const [prop, val] of Object.entries(v)) {
        if (prop.startsWith("--")) node.style.setProperty(prop, val); else node.style[prop] = val;
      }
    }
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}

const SVG_NS = "http://www.w3.org/2000/svg";
export function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    if (k === "text") node.textContent = v; else node.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c) node.append(c);
  return node;
}

export const fmtInt = (v) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : Math.round(v).toLocaleString("en-US"));
export const fmtPct = (r, d = 1) => (r === null || r === undefined ? "—" : `${+(r * 100).toFixed(d)}%`);
export const ctxWords = (k) => (k === null || k === undefined ? "an unknown amount" : `about ${k}k`);
export const plural = (n, one, many = `${one}s`) => `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;
export const kfmt = (v) => (v >= 1000 ? `${+(v / 1000).toFixed(1)}k` : `${Math.round(v)}`);

/** Plain names first; the method codes from CONTRACT.md stay visible as secondary labels. */
export const METHOD_NAMES = {
  "A0": "Base model",
  "A1": "One shared LoRA",
  "A2": "Per-role LoRAs",
  "A3": "Naive cache reuse",
  "A5-H1": "RoleKV",
  "A5-H2": "RoleKV, keep handoff KV",
};
export const methodName = (m) => METHOD_NAMES[m] ?? m;

export function reducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** Say something to screen readers through the page's polite live region. */
export function announce(text) {
  const a = $("announcer");
  if (a) a.textContent = text;
}

/** Round-step axis from 0 to at least maxVal, about `count` intervals. */
export function niceTicks(maxVal, count = 4) {
  const raw = (maxVal > 0 ? maxVal : 1) / count;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw);
  const max = Math.ceil((maxVal > 0 ? maxVal : 1) / step) * step;
  const ticks = [];
  for (let v = 0; v <= max + step / 2; v += step) ticks.push(+v.toFixed(6));
  return { max, ticks };
}

/**
 * Hover and keyboard tooltip. `place` is "above" or "below" the pointer, so a tooltip on the top row of a
 * comparison never covers the row it should be compared with.
 */
export function attachTooltip(node, linesFn, place = "above") {
  const tip = $("tooltip");
  const show = (x, y, anchorH = 0) => {
    tip.replaceChildren(...linesFn().filter(Boolean).map((l) => el("div", { text: l })));
    tip.hidden = false;
    node.setAttribute("aria-describedby", "tooltip");
    const r = tip.getBoundingClientRect();
    tip.style.left = `${Math.max(8, Math.min(window.innerWidth - r.width - 8, x + 14))}px`;
    const top = place === "below" ? y + anchorH + 14 : y - r.height - 12;
    tip.style.top = `${Math.max(8, Math.min(window.innerHeight - r.height - 8, top))}px`;
  };
  const hide = () => { tip.hidden = true; node.removeAttribute("aria-describedby"); };
  node.addEventListener("mousemove", (e) => show(e.clientX, e.clientY));
  node.addEventListener("mouseleave", hide);
  node.addEventListener("focus", () => { const b = node.getBoundingClientRect(); show(b.left + b.width / 2, b.top, b.height); });
  node.addEventListener("blur", hide);
  node.addEventListener("keydown", (e) => { if (e.key === "Escape") hide(); });
}
