// The KV cache as a thing you can look at. One session's cache is a block of cells pointing at the viewer:
// the face in front is a square map of the token positions (a few tokens per cell, laid out row by row),
// and behind it the model's layers line up into the depth. Two blocks, one per method, play the same
// session: the per-role LoRA block is emptied and re-read at every turn, a cross-section sweeping out of the
// depth towards the viewer; the RoleKV block keeps everything it has seen. Colors follow the dashboard's
// tape: coral = read this turn, cyan = held in the cache, violet = the handoff re-read, white = written this
// turn. The state of every cell is a pure function of (turn, read progress, write progress), the same
// inputs the 2D tapes use, so the two never disagree.
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { tapeSpans } from "./metrics.js";

const HEX = { cached: "#10A0AE", read: "#F0593A", canon: "#8470F5", written: "#E6EBF0", dim: "#2A3442", ghost: "#1A2230", bg: "#0B0E12" };
const GAP = 8;          // aisle between the two blocks, in cell pitches
const TAIL = 16;        // cells over which the wave's heat fades, about one row
const SIZE = { ghost: 0.14, dim: 0.4, lit: 0.84 };   // cell size by state: not yet in the session, emptied, held
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const lerp = (a, b, k) => a + (b - a) * k;
const lin = (hex) => { const c = new THREE.Color(hex); return [c.r, c.g, c.b]; };

/**
 * @param canvas the <canvas> to draw into (sized by CSS; the renderer follows it)
 * @param methods [{ key, turns }] in display order, left to right
 * @param scale   the token count that fills a block (the session's final length)
 * @param labels  { [key]: HTMLElement } floating labels, kept above each block's front face
 */
export function createCacheScene(canvas, { methods, scale, bin = 8, layers = 28, labels = {}, reducedMotion = false, onHover }) {
  const nCols = Math.ceil(scale / bin);          // cells per layer
  const side = Math.ceil(Math.sqrt(nCols));      // the layer map is side × side, filled row by row
  // The read wave is a cross-section moving out of the depth: each layer starts a whole map after the one
  // behind it, so by read = 1 the front layer has just finished.
  const lag = nCols / layers;
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(HEX.bg);
  scene.fog = new THREE.Fog(HEX.bg, 1, 1);
  const camera = new THREE.PerspectiveCamera(30, 1, 0.5, 4000);
  scene.add(new THREE.HemisphereLight(0x9fc9d6, 0x0b0e12, 1.15));
  const sun = new THREE.DirectionalLight(0xffffff, 1.9);
  sun.position.set(-60, 120, 90);
  scene.add(sun);
  const fill = new THREE.DirectionalLight(0x4fdce8, 0.5);
  fill.position.set(80, 30, -60);
  scene.add(fill);

  // A faint floor and a drift of loose tokens give the blocks somewhere to lie.
  const floorY = -(side - 1) / 2 - 1.2;
  const grid = new THREE.GridHelper(600, 60, 0x1e2733, 0x141b25);
  grid.position.y = floorY;
  grid.material.transparent = true;
  grid.material.opacity = 0.5;
  scene.add(grid);
  const motes = makeMotes(scene, side * 4);

  // The front face is the token map: cell c sits at column c % side, row floor(c / side) from the top (so
  // the map reads like a page, newest tokens at the bottom). Layer l sits l cells into the depth, layer 1 at
  // the back and the last layer nearest the viewer. The blocks stand side by side on the floor.
  const C = { cached: lin(HEX.cached), read: lin(HEX.read), canon: lin(HEX.canon), written: lin(HEX.written), dim: lin(HEX.dim), ghost: lin(HEX.ghost) };
  const geo = new THREE.BoxGeometry(1, 1, 1);
  const extent = { x: methods.length * side + (methods.length - 1) * GAP, y: side, z: layers };
  const _p = new THREE.Vector3(), _q = new THREE.Quaternion(), _s = new THREE.Vector3(), _m = new THREE.Matrix4();
  const slabs = methods.map((m, idx) => {
    const mesh = new THREE.InstancedMesh(geo, new THREE.MeshLambertMaterial({ color: 0xffffff }), nCols * layers);
    mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    const x = (idx - (methods.length - 1) / 2) * (side + GAP);
    const slab = { key: m.key, turns: m.turns, mesh, x, size: new Float32Array(nCols * layers).fill(-1), frame: { turn: 0, read: 0, write: 0 }, label: labels[m.key] };
    mesh.setColorAt(0, new THREE.Color(HEX.ghost));
    mesh.instanceColor.setUsage(THREE.DynamicDrawUsage);
    mesh.userData.slab = slab;
    scene.add(mesh);
    return slab;
  });
  const place = (slab, c, l, size) => {
    _p.set(slab.x + (c % side) - (side - 1) / 2, (side - 1) / 2 - Math.floor(c / side), l - (layers - 1) / 2);
    _s.setScalar(size);
    slab.mesh.setMatrixAt(l * nCols + c, _m.compose(_p, _q, _s));
  };

  // Post: bloom makes the hot cells glow; the output pass tone-maps and converts to sRGB.
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.65, 0.5, 0.7);
  composer.addPass(bloom);
  composer.addPass(new OutputPass());

  /** What a cell is on this turn: the dashboard's categories, decided by the cell's centre token. */
  const category = (sp, c) => {
    const mid = c * bin + bin / 2;
    if (mid >= sp.recomputed[1]) return "written";
    if (sp.canon && mid >= sp.canon[0] && mid < sp.canon[1]) return "canon";
    if (mid >= sp.recomputed[0]) return "recomputed";
    return "reused";
  };

  const paint = (slab, now) => {
    const { turn, read, write } = slab.frame;
    const sp = tapeSpans(slab.turns[turn]);
    // The wave starts at the first cell to read in the back layer and, by read = 1, has passed the last cell
    // to read in the front layer.
    const start = sp.reused[1] / bin, stop = sp.recomputed[1] / bin;
    const front = start + (stop - start + (layers - 1) * lag) * read;
    const paperEnd = sp.recomputed[1] + (sp.end - sp.recomputed[1]) * write;
    const colors = slab.mesh.instanceColor.array;
    const shimmer = now * 0.0016;
    let moved = false;
    for (let c = 0; c < nCols; c++) {
      const on = c * bin < paperEnd;
      const cat = on ? category(sp, c) : null;
      for (let l = 0; l < layers; l++) {
        let base, heat = 0, white = 0, size;
        if (!on) {
          base = C.ghost; size = SIZE.ghost;
        } else if (cat === "reused") {
          base = C.cached; size = SIZE.lit;
          heat = 0.08 * Math.sin(shimmer + c * 0.21 + l * 0.35);
        } else if (cat === "written") {
          base = C.written;
          const grown = clamp((paperEnd - c * bin) / bin, 0, 1);      // the cell being written grows in
          size = lerp(SIZE.ghost, SIZE.lit, grown);
          heat = write < 1 && grown < 1 ? 0.9 : 0.15;
        } else {
          const behind = front - l * lag - (c + 0.5);                 // cells behind the wave front are read
          if (behind < 0) { base = C.dim; size = SIZE.dim; }
          else {
            const k = clamp(1 - behind / TAIL, 0, 1);
            base = cat === "canon" ? C.canon : C.read;
            heat = 2.2 * k; white = 0.5 * k;
            size = lerp(SIZE.dim, SIZE.lit, clamp(behind / 1.5, 0, 1));
          }
        }
        const i = l * nCols + c;
        if (Math.abs(size - slab.size[i]) > 0.002) { slab.size[i] = size; place(slab, c, l, size); moved = true; }
        const shade = 1 - 0.26 * (Math.floor(c / side) / (side - 1));   // higher rows catch more light
        const gain = shade * (1 + heat);
        colors[i * 3] = (base[0] + (1 - base[0]) * white) * gain;
        colors[i * 3 + 1] = (base[1] + (1 - base[1]) * white) * gain;
        colors[i * 3 + 2] = (base[2] + (1 - base[2]) * white) * gain;
      }
    }
    slab.mesh.instanceColor.needsUpdate = true;
    if (moved) slab.mesh.instanceMatrix.needsUpdate = true;
  };

  // ---------- Camera ----------
  // `fraction` is how much of the view width the towers may take; `shift` and `lift` slide the picture right
  // and up with an off-centre projection, leaving the left for the words and the bottom for the HUD.
  const orbit = { az: 0.5, polar: 1.12, userAz: 0, userPolar: 0, targetAz: 0, targetPolar: 0, fraction: 0.46, shift: 0.2, lift: 0.1 };
  let w = 1, h = 1, dirty = true;
  const fit = () => {
    const r = canvas.getBoundingClientRect();
    w = Math.max(1, Math.round(r.width)); h = Math.max(1, Math.round(r.height));
    renderer.setSize(w, h, false);
    composer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    const narrow = w < 900;
    orbit.fraction = narrow ? 0.9 : 0.46;
    orbit.shift = narrow ? 0 : 0.2;
    orbit.lift = narrow ? 0 : 0.1;
    if (narrow) camera.clearViewOffset(); else camera.setViewOffset(w, h, -orbit.shift * w, orbit.lift * h, w, h);
    dirty = true;
  };
  const placeCamera = (now) => {
    const sway = reducedMotion ? 0 : 0.16 * Math.sin(now / 13000 * Math.PI * 2);
    orbit.userAz += (orbit.targetAz - orbit.userAz) * 0.08;
    orbit.userPolar += (orbit.targetPolar - orbit.userPolar) * 0.08;
    const az = orbit.az + sway + orbit.userAz;
    const polar = clamp(orbit.polar + orbit.userPolar, 0.8, 1.4);
    // Far enough that the blocks' footprint takes `fraction` of the view width and their height fits too.
    const halfTanY = Math.tan(camera.fov / 2 * Math.PI / 180), halfTanX = halfTanY * camera.aspect;
    const footprint = extent.x * Math.abs(Math.cos(az)) + extent.z * Math.abs(Math.sin(az)) + 4;
    // The blocks take about half the view height: their projected height over the visible half-height.
    const depth = extent.z * Math.abs(Math.cos(az)) + extent.x * Math.abs(Math.sin(az));
    const tall = (extent.y * Math.sin(polar) + depth * Math.cos(polar)) * 1.05;
    const R = Math.max(footprint / 2 / (orbit.fraction * halfTanX), tall / halfTanY);
    camera.position.set(R * Math.sin(polar) * Math.sin(az), R * Math.cos(polar), R * Math.sin(polar) * Math.cos(az));
    camera.lookAt(0, 0, 0);
    scene.fog.near = R * 0.9; scene.fog.far = R * 3;
  };

  // Drag to look around (mouse, pen or finger).
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); canvas.classList.add("dragging"); });
  canvas.addEventListener("pointermove", (e) => {
    if (drag) {
      orbit.targetAz = clamp(orbit.targetAz - (e.clientX - drag.x) * 0.005, -1.2, 1.2);
      orbit.targetPolar = clamp(orbit.targetPolar - (e.clientY - drag.y) * 0.004, -0.35, 0.3);
      drag = { x: e.clientX, y: e.clientY };
      dirty = true;
    } else { hover.x = e.clientX; hover.y = e.clientY; hover.pending = true; }
  });
  const endDrag = () => { drag = null; canvas.classList.remove("dragging"); };
  canvas.addEventListener("pointerup", endDrag);
  canvas.addEventListener("pointercancel", endDrag);
  canvas.addEventListener("pointerleave", () => { hover.pending = false; hover.last = null; onHover?.(null); });

  // ---------- Hover ----------
  const hover = { x: 0, y: 0, pending: false, last: null };
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  /** The cell under a viewport point, or null: which method, layer, token range and category. */
  const pickAt = (x, y) => {
    const r = canvas.getBoundingClientRect();
    ndc.set(((x - r.left) / r.width) * 2 - 1, -((y - r.top) / r.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera);
    const hit = raycaster.intersectObjects(slabs.map((s) => s.mesh), false)[0];
    if (!hit || hit.instanceId === undefined) return null;
    const slab = hit.object.userData.slab;
    const c = hit.instanceId % nCols, l = Math.floor(hit.instanceId / nCols);
    const sp = tapeSpans(slab.turns[slab.frame.turn]);
    if (c * bin >= sp.end) return null;                                 // a ghost cell: not in the session yet
    return { key: slab.key, layer: l + 1, layers, from: c * bin + 1, to: Math.min((c + 1) * bin, sp.end), category: category(sp, c), x, y };
  };
  const pick = () => {
    hover.pending = false;
    const info = pickAt(hover.x, hover.y);
    const sig = info && `${info.key}:${info.layer}:${info.from}:${info.category}`;
    if (sig !== hover.last || info) { hover.last = sig; onHover?.(info); }
  };

  // ---------- Labels ----------
  const _v = new THREE.Vector3();
  const placeLabels = () => {
    for (const s of slabs) {
      if (!s.label) continue;
      _v.set(s.x, (side - 1) / 2 + 2.2, (layers - 1) / 2).project(camera);
      const visible = _v.z < 1;
      s.label.style.transform = `translate(${((_v.x + 1) / 2 * w).toFixed(1)}px, ${((1 - _v.y) / 2 * h).toFixed(1)}px) translate(-50%, -100%)`;
      s.label.style.opacity = visible ? "1" : "0";
    }
  };

  // ---------- Loop ----------
  let running = true, raf = 0;
  const renderFrame = (now) => {
    placeCamera(now);
    for (const s of slabs) paint(s, now);
    if (hover.pending && !drag) pick();
    composer.render();
    placeLabels();
    dirty = false;
  };
  const loop = (now) => {
    raf = requestAnimationFrame(loop);
    if (!running) return;
    if (!reducedMotion) { motes.drift(now); dirty = true; }
    if (dirty || hover.pending) renderFrame(now);
  };
  const ro = new ResizeObserver(fit);
  ro.observe(canvas);
  fit();
  document.addEventListener("visibilitychange", () => { running = !document.hidden; });
  raf = requestAnimationFrame(loop);

  return {
    /** Set one method's frame: which turn, how far its read has swept and how far its write has grown (0..1). */
    setFrame(key, frame) {
      const s = slabs.find((x) => x.key === key);
      if (!s) return;
      Object.assign(s.frame, frame);
      dirty = true;
    },
    /** Draw one frame now, outside the animation loop (a still for tests or a background page). */
    render() { renderFrame(performance.now()); },
    pickAt, layers, bin, nCols, side,
    dispose() { cancelAnimationFrame(raf); ro.disconnect(); renderer.dispose(); },
  };
}

/** Loose tokens drifting past, far behind the towers. */
function makeMotes(scene, span) {
  const n = 600;
  const pos = new Float32Array(n * 3);
  const vel = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    pos[i * 3] = (Math.random() - 0.5) * span * 2;
    pos[i * 3 + 1] = (Math.random() - 0.5) * span * 1.2;
    pos[i * 3 + 2] = -span * 0.8 - Math.random() * span * 2;
    vel[i] = 0.004 + Math.random() * 0.012;
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const points = new THREE.Points(g, new THREE.PointsMaterial({ color: 0x3fb6c2, size: 0.5, sizeAttenuation: true, transparent: true, opacity: 0.5, depthWrite: false }));
  scene.add(points);
  let last = 0;
  return {
    drift(now) {
      const dt = last ? Math.min(50, now - last) : 16;
      last = now;
      for (let i = 0; i < n; i++) {
        pos[i * 3] += vel[i] * dt;
        if (pos[i * 3] > span) pos[i * 3] = -span;
      }
      g.attributes.position.needsUpdate = true;
    },
  };
}
