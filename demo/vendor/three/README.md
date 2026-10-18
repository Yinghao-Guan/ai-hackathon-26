# Three.js (vendored)

Used only by the homepage (`demo/js/cache3d.js`). Vendored so the demo keeps working without network access.

- Version: see `VERSION` (`three@0.186.1` at the time of writing). License: MIT, in `LICENSE`.
- Files are the official npm package contents, unmodified:
  - `three.module.js` and `three.core.js` from `build/`
  - `addons/postprocessing/*.js` and `addons/shaders/*.js` from `examples/jsm/`, the set needed for
    `EffectComposer` + `RenderPass` + `UnrealBloomPass` + `OutputPass`
- They were fetched from unpkg and jsDelivr and the two copies compared by SHA-256 before being committed.
  Note that jsDelivr serves its own Terser output for `*.min.js` paths that the package does not ship; use the
  unminified `build/` files, as here.
- `index.html` maps the bare specifiers with an import map: `three` → `three.module.js`,
  `three/addons/` → `addons/`.

To update: pick a version, replace the files from the same paths of that version, update `VERSION`, and check
the addons' relative imports still resolve (`grep -rhoE "from ['\"][^'\"]+['\"]" addons`).
