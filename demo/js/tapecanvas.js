// The session tape on a <canvas>, for dark pages (the homepage HUD and the live demo). It paints one turn the
// way the dashboard's DOM tape does, with light: paper is the session so far, the cyan rule is what the cache
// holds, coral is read this turn, violet is the handoff re-read, and the outline is what the role writes.
// Colors come from the page's CSS custom properties (--paper, --cache, --recompute, ...).
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.max(0, Math.min(r, h / 2, w / 2));
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

/**
 * A painter for one canvas. draw({ spans, scale, read, write }) paints one turn: `spans` from
 * metrics.tapeSpans, `scale` the token count that fills the width, and `read` and `write` the progress
 * (0..1) of the turn's two phases. It redraws the last frame by itself when the canvas is resized.
 */
export function tapePainter(canvas) {
  const ctx = canvas.getContext("2d");
  let w = 1, h = 1, c = {}, last = null;
  const resize = () => {
    const r = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    w = Math.max(1, r.width); h = Math.max(1, r.height);
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    c = {
      paper: cssVar("--paper"), rule: cssVar("--paper-rule"),
      cache: cssVar("--cache"), cacheSoft: cssVar("--cache-soft"), cacheGlow: cssVar("--cache-glow"),
      ink: cssVar("--recompute"), inkGlow: cssVar("--recompute-glow"),
      canon: cssVar("--canon"), canonGlow: cssVar("--canon-glow"), written: cssVar("--written"),
    };
    if (last) draw(last);
  };
  const draw = (f) => {
    last = f;
    const { spans: s, scale, read, write } = f;
    const X = (tok) => (tok / scale) * w;
    // A segment never vanishes: 3px minimum, like the dashboard's tapes.
    const span = (a, b) => { const x0 = X(a); return [x0, Math.max(3, X(b) - x0)]; };
    const readTo = s.reused[1] + (s.recomputed[1] - s.reused[1]) * read;
    const paperEnd = s.recomputed[1] + (s.end - s.recomputed[1]) * write;
    const pad = 5, y = pad, hh = h - 2 * pad;
    ctx.clearRect(0, 0, w, h);
    ctx.shadowBlur = 0;

    let [x, ww] = span(0, paperEnd);
    roundRect(ctx, x, y, ww, hh, 3);
    ctx.fillStyle = c.paper; ctx.fill();
    ctx.lineWidth = 1; ctx.strokeStyle = c.rule; ctx.stroke();

    if (s.reused[1] > 0) {
      [x, ww] = span(0, s.reused[1]);
      ctx.fillStyle = c.cacheSoft; ctx.fillRect(x, y, ww, hh);
      ctx.shadowColor = c.cacheGlow; ctx.shadowBlur = 12;
      ctx.fillStyle = c.cache; ctx.fillRect(x, y + hh - 3, ww, 3);
      ctx.shadowBlur = 0;
    }
    if (s.canon && readTo > s.canon[0]) {
      [x, ww] = span(s.canon[0], Math.min(readTo, s.canon[1]));
      ctx.shadowColor = c.canonGlow; ctx.shadowBlur = 12;
      ctx.fillStyle = c.canon; ctx.fillRect(x, y + hh * 0.52, ww, hh * 0.48);
      ctx.shadowBlur = 0;
    }
    if (readTo > s.recomputed[0]) {
      [x, ww] = span(s.recomputed[0], Math.min(readTo, s.recomputed[1]));
      ctx.shadowColor = c.inkGlow; ctx.shadowBlur = 14;
      ctx.fillStyle = c.ink; ctx.fillRect(x, y, ww, hh);
      ctx.shadowBlur = 0;
    }
    if (read > 0 && read < 1) {
      const hx = X(readTo);
      const g = ctx.createRadialGradient(hx, h / 2, 0, hx, h / 2, 22);
      g.addColorStop(0, "rgba(255,255,255,0.55)"); g.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = g; ctx.fillRect(hx - 22, 0, 44, h);
      ctx.fillStyle = "#fff"; ctx.fillRect(hx - 1, y - 2, 2, hh + 4);
    }
    if (write > 0) {
      [x, ww] = span(s.recomputed[1], paperEnd);
      ctx.lineWidth = 1.5; ctx.strokeStyle = c.written;
      ctx.strokeRect(x + 0.75, y + 0.75, Math.max(1, ww - 1.5), hh - 1.5);
      if (write < 1 && Math.floor(performance.now() / 240) % 2 === 0) {
        ctx.fillStyle = c.written; ctx.fillRect(x + ww - 1, y + 3, 2, hh - 6);
      }
    }
  };
  new ResizeObserver(resize).observe(canvas);
  resize();
  return { draw };
}
