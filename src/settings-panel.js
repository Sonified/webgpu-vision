// Shared settings panel: hamburger button + slide-in drawer.
// Renders controls from a declarative schema, wired to a SettingsStore.
// Any demo can import this and get a fully functional settings UI in two lines.

import { SettingsStore } from './settings-store.js';

const DRAWER_WIDTH = 260;

const CSS = `
.wgv-hamburger {
  position: fixed; top: 12px; left: 12px; z-index: 1001;
  width: 36px; height: 36px;
  display: none; align-items: center; justify-content: center;
  background: rgba(20,20,40,0.85); border: 1px solid #444;
  border-radius: 4px; color: #ccc; font-size: 1.3rem;
  cursor: pointer; user-select: none;
  transition: transform 0.2s ease, color 0.15s;
}
.wgv-hamburger:hover { color: #4fc3f7; border-color: #4fc3f7; }
body.wgv-drawer-open .wgv-hamburger { transform: translateX(${DRAWER_WIDTH + 8}px); }

.wgv-drawer {
  position: fixed; top: 0; left: 0; z-index: 1100;
  width: ${DRAWER_WIDTH}px; height: 100%;
  background: #1e1e1e; border-right: 1px solid #444;
  padding: 14px 14px 14px; overflow-y: auto;
  transform: translateX(-100%); transition: transform 0.25s ease;
  font-family: system-ui, sans-serif; font-size: 0.85rem; color: #ddd;
  scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.15) transparent;
}
.wgv-drawer::-webkit-scrollbar { width: 5px; }
.wgv-drawer::-webkit-scrollbar-track { background: transparent; }
.wgv-drawer::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
.wgv-drawer.open { transform: translateX(0); }

.wgv-section { margin-bottom: 14px; }
.wgv-section:last-child { margin-bottom: 0; }
.wgv-section-label {
  color: #4fc3f7; font-weight: 600; font-size: 0.95rem;
  margin-top: 10px; margin-bottom: 5px;
  padding-top: 8px; border-top: 1px solid #333;
}
.wgv-section:first-child .wgv-section-label { margin-top: 0; padding-top: 0; border-top: none; }

.wgv-row {
  display: flex; align-items: center; margin-bottom: 4px;
}
.wgv-row-label {
  width: 72px; min-width: 72px; color: #888; font-size: 0.85rem;
  cursor: default;
}
.wgv-row-label[data-tip] {
  cursor: help; text-decoration: underline dotted #555;
}
.wgv-row select {
  flex: 1; background: #222; color: #ccc;
  border: 1px solid #444; border-radius: 3px;
  padding: 1px 4px; font-size: 0.75rem;
}
.wgv-row input[type="range"] {
  flex: 1; height: 4px; accent-color: #4fc3f7; cursor: pointer;
}
.wgv-row .wgv-val {
  width: 42px; text-align: right; color: #4fc3f7;
  font-size: 0.85rem; font-weight: 600;
  font-variant-numeric: tabular-nums;
}
.wgv-row input[type="checkbox"] { accent-color: #4fc3f7; cursor: pointer; }

.wgv-tip {
  position: fixed; z-index: 10000; pointer-events: none;
  background: #222; color: #ccc; font-size: 0.7rem;
  padding: 3px 6px; border-radius: 3px; border: 1px solid #444;
  white-space: nowrap; display: none;
}
`;

// Helper: merge overrides without clobbering controls. Supports extraControls to append.
function buildSection(base, overrides = {}) {
  const { extraControls, ...rest } = overrides;
  const section = { ...base, ...rest };
  if (extraControls) {
    section.controls = [...(base.controls || []), ...extraControls];
  }
  return section;
}

// Pre-built section schemas. Each is a function so demos can override or extend.
// Pass extraControls: [...] to inject demo-specific controls into a shared section.
export const SECTIONS = {
  backend: (overrides) => buildSection({
    label: 'Backend', id: 'wgv-sec-backend',
    controls: [
      { type: 'select', key: 'backend', label: 'Engine',
        options: [{value:'webgpu', label:'WebGPU'}, {value:'mediapipe', label:'MediaPipe'}] },
    ],
  }, overrides),

  mediapipe: (overrides) => buildSection({
    label: 'MediaPipe', id: 'wgv-sec-mediapipe', hidden: true,
    controls: [
      { type: 'select', key: 'mediapipe-mode', label: 'Mode',
        options: [{value:'workers', label:'Workers'}, {value:'standard', label:'Standard (main thread)'}] },
    ],
  }, overrides),

  head: (overrides) => buildSection({
    label: 'Head', id: 'wgv-sec-head',
    controls: [
      { type: 'select', key: 'face-model', label: 'Model',
        tip: 'Neural network for persistent face tracking between detections',
        options: [{value:'landmark', label:'Landmark (478pt, precise)'}, {value:'detector', label:'Detector (6pt, fast)'}] },
      { type: 'select', key: 'face-precision', label: 'Precision',
        tip: 'f32 = standard. f16 = optimized, smaller and faster',
        options: [{value:'f32', label:'f32'}, {value:'f16', label:'f16'}] },
      { type: 'checkbox', key: 'show-self', label: 'Show Self', text: 'Camera preview',
        tip: 'Show live camera feed in the corner' },
      { type: 'checkbox', key: 'show-crop', label: 'Show Crop', text: 'Detector input',
        tip: 'Show the cropped frame the detector actually sees' },
      { type: 'select', key: 'face-roi', label: 'ROI',
        tip: 'Region of interest cropping. Auto = crops when face is centered, Off = full frame',
        options: [{value:'auto', label:'Auto (center only)'}, {value:'off', label:'Off (full frame)'}, {value:'on', label:'On (always crop)'}] },
      { type: 'select', key: 'head-filter', label: 'Filter',
        tip: '1-Euro smooths jitter while preserving fast movement',
        options: [{value:'none', label:'None (raw)'}, {value:'one-euro', label:'1-Euro'}] },
      { type: 'range', key: 'head-floor', label: 'Floor',
        min: 50, max: 10000, step: 50,
        toSlider: v => v * 1000, fromSlider: v => v / 1000, format: v => v.toFixed(2),
        tip: 'Min cutoff. Lower = smoother, higher = more responsive' },
      { type: 'range', key: 'head-beta', label: 'Beta',
        min: 0, max: 1000, step: 5,
        toSlider: v => v * 10, fromSlider: v => v / 10, format: v => v.toFixed(1),
        tip: 'Speed sensitivity. Lower = steady, higher = reacts to fast movement' },
    ],
  }, overrides),

  hands: (overrides) => buildSection({
    label: 'Hands', id: 'wgv-sec-hands',
    controls: [
      { type: 'select', key: 'palm-engine', label: 'Palm',
        tip: 'Inference engine for palm detection. WGSL = custom shaders, ORT = ONNX Runtime',
        options: [{value:'ort', label:'ORT'}, {value:'wgsl', label:'WGSL'}] },
      { type: 'select', key: 'hand-model', label: 'Model',
        tip: 'Neural network for persistent hand landmark tracking between detections',
        options: [{value:'standard', label:'Standard'}, {value:'large', label:'Large'}] },
      { type: 'select', key: 'hand-precision', label: 'Precision',
        tip: 'f32 = standard. f16 = optimized, smaller and faster',
        options: [{value:'f32', label:'f32'}, {value:'f16', label:'f16'}] },
      { type: 'select', key: 'num-hands', label: 'Hands',
        tip: 'Max simultaneous hands to track',
        options: [{value:'1', label:'1'}, {value:'2', label:'2'}] },
      { type: 'select', key: 'z-tracking', label: 'Z Tracking',
        tip: 'Enable depth estimation from hand landmarks',
        options: [{value:'off', label:'Off'}, {value:'on', label:'On'}] },
      { type: 'select', key: 'hand-filter', label: 'Filter',
        tip: '1-Euro smooths jitter while preserving fast movement',
        options: [{value:'none', label:'None (raw)'}, {value:'one-euro', label:'1-Euro'}] },
      { type: 'range', key: 'hand-floor', label: 'Floor',
        min: 50, max: 10000, step: 50,
        toSlider: v => v * 1000, fromSlider: v => v / 1000, format: v => v.toFixed(2),
        tip: 'Min cutoff. Lower = smoother, higher = more responsive' },
      { type: 'range', key: 'hand-beta', label: 'Beta',
        min: 0, max: 1000, step: 5,
        toSlider: v => v * 10, fromSlider: v => v / 10, format: v => v.toFixed(1),
        tip: 'Speed sensitivity. Lower = steady, higher = reacts to fast movement' },
    ],
  }, overrides),

  interp: (overrides) => buildSection({
    label: 'Frame Interp', id: 'wgv-sec-interp',
    controls: [
      { type: 'select', key: 'interp', label: 'Mode',
        tip: 'Interpolate between inference frames. Adaptive adjusts by hand speed',
        options: [{value:'1', label:'1x (off)'}, {value:'2', label:'2x'}, {value:'4', label:'4x'}, {value:'adaptive', label:'Adaptive'}] },
      { type: 'custom', key: 'interp-zones', render: (row) => {
        row.className = '';
        row.style.cssText = 'display:flex; gap:14px; margin-top:8px; justify-content:center; align-items:flex-end;';
        row.id = 'interp-zones';
        const zones = ['HEAD', 'L', 'R'];
        const ids = ['wgv-boxes-head', 'wgv-boxes-handL', 'wgv-boxes-handR'];
        for (let i = 0; i < 3; i++) {
          const col = document.createElement('div');
          col.style.cssText = 'display:flex; flex-direction:column; align-items:center; gap:4px;';
          const boxes = document.createElement('div');
          boxes.id = ids[i];
          boxes.style.cssText = 'display:flex; flex-direction:column-reverse; gap:2px; height:42px;';
          col.appendChild(boxes);
          const lbl = document.createElement('span');
          lbl.style.cssText = 'color:#666; font-size:0.62rem;';
          lbl.textContent = zones[i];
          col.appendChild(lbl);
          row.appendChild(col);
        }
      }},
    ],
  }, overrides),

  detection: (overrides) => buildSection({
    label: 'Palm Detection', id: 'wgv-sec-detection',
    controls: [
      { type: 'range', key: 'hand-conf', label: 'HandConf',
        min: 0, max: 100, step: 5,
        toSlider: v => v * 100, fromSlider: v => v / 100, format: v => v.toFixed(2) },
      { type: 'range', key: 'dup-dist', label: 'Dup dist',
        min: 5, max: 80, step: 1, format: v => String(v),
        tip: 'px at which two slots count as duplicate' },
    ],
  }, overrides),

  logGates: (overrides) => buildSection({
    label: 'Log Output', id: 'wgv-sec-log-gates',
    controls: [
      { type: 'custom', key: 'log-gates-panel', render: (row) => {
        row.className = '';
        row.style.cssText = 'display:flex; flex-direction:column; gap:2px;';
        const gates = window.__wgvLogGates;
        if (!gates) { row.textContent = 'Log gates not initialized'; return; }
        for (const cat of gates.categories) {
          const label = document.createElement('label');
          label.style.cssText = 'display:flex; align-items:center; gap:8px; padding:2px 0; cursor:pointer; font-size:0.85rem; color:#ccc;';
          const cb = document.createElement('input');
          cb.type = 'checkbox';
          cb.style.accentColor = '#4fc3f7';
          cb.checked = !!gates.state[cat.id];
          cb.addEventListener('change', () => gates.set(cat.id, cb.checked));
          label.appendChild(cb);
          const span = document.createElement('span');
          span.textContent = cat.label;
          label.appendChild(span);
          row.appendChild(label);
        }
      }},
    ],
  }, overrides),
};

// Sensible defaults for the full pipeline.
// Use with new SettingsStore({ defaults: DEFAULTS, profiles: PROFILES, ... })
export const DEFAULTS = {
  'backend': 'webgpu',
  'mediapipe-mode': 'standard',
  'face-model': 'landmark',
  'face-precision': 'f16',
  'face-roi': 'auto',
  'show-self': false,
  'show-crop': false,
  'head-filter': 'one-euro',
  'head-floor': 3.0,
  'head-beta': 50.0,
  'palm-engine': 'ort',
  'hand-model': 'standard',
  'hand-precision': 'f32',
  'num-hands': '2',
  'z-tracking': 'off',
  'hand-filter': 'one-euro',
  'hand-floor': 2.0,
  'hand-beta': 55.5,
  'interp': 'adaptive',
  'hand-conf': 0.45,
  'dup-dist': 25,
};

// Per-engine profile overrides. Keys not listed here fall back to DEFAULTS.
export const PROFILES = {
  'webgpu': {
    'head-floor': 3.0,
    'head-beta': 50.0,
    'hand-floor': 2.0,
    'hand-beta': 55.5,
    'interp': 'adaptive',
  },
  'mediapipe': {
    'head-floor': 0.1,
    'head-beta': 3.5,
    'hand-floor': 0.1,
    'hand-beta': 26.5,
    'interp': '1',
  },
};

// Keys that stay the same regardless of engine
export const GLOBAL_KEYS = ['mediapipe-mode', 'face-model', 'palm-engine', 'hand-model', 'hand-precision', 'face-precision', 'face-roi', 'show-self', 'show-crop', 'num-hands', 'z-tracking', 'hand-conf', 'dup-dist'];

export class SettingsPanel {
  constructor(store, options = {}) {
    if (!(store instanceof SettingsStore)) {
      throw new Error('SettingsPanel requires a SettingsStore instance');
    }
    this.store = store;
    this.sections = options.sections || [];
    this.container = options.container || document.body;
    this.hotkey = options.hotkey ?? '`';
    this._controlEls = {};  // key -> { input, valSpan }
    this._open = false;

    this._injectCSS();
    this._buildDOM();
    this._bindHotkey();
    this._subscribeStore();
  }

  toggle() {
    this._open = !this._open;
    this._drawerEl.classList.toggle('open', this._open);
    document.body.classList.toggle('wgv-drawer-open', this._open);
    this._hamburgerEl.textContent = this._open ? '✕' : '☰';
  }

  show() { if (!this._open) this.toggle(); }
  hide() { if (this._open) this.toggle(); }

  // Add a section dynamically after construction
  addSection(section) {
    this.sections.push(section);
    this._renderSection(section, this._drawerEl);
  }

  // Show the hamburger (call after your app is ready)
  reveal() {
    this._hamburgerEl.style.display = 'flex';
  }

  _injectCSS() {
    if (document.getElementById('wgv-settings-css')) return;
    const style = document.createElement('style');
    style.id = 'wgv-settings-css';
    style.textContent = CSS;
    document.head.appendChild(style);
  }

  _buildDOM() {
    // Hamburger button
    this._hamburgerEl = document.createElement('button');
    this._hamburgerEl.className = 'wgv-hamburger';
    this._hamburgerEl.title = `Settings (press ${this.hotkey})`;
    this._hamburgerEl.textContent = '☰';
    this._hamburgerEl.addEventListener('click', () => this.toggle());
    this.container.appendChild(this._hamburgerEl);

    // Drawer
    this._drawerEl = document.createElement('div');
    this._drawerEl.className = 'wgv-drawer';
    this.container.appendChild(this._drawerEl);

    // Tooltip element
    this._tipEl = document.createElement('div');
    this._tipEl.className = 'wgv-tip';
    this._tipEl.style.display = 'none';
    this.container.appendChild(this._tipEl);

    // Render sections
    for (const section of this.sections) {
      this._renderSection(section, this._drawerEl);
    }

    // Tooltip hover listeners
    this._drawerEl.addEventListener('pointerenter', (e) => this._showTip(e), true);
    this._drawerEl.addEventListener('pointerleave', (e) => this._hideTip(e), true);
  }

  _renderSection(section, parent) {
    const div = document.createElement('div');
    div.className = 'wgv-section';
    if (section.id) div.id = section.id;
    if (section.hidden) div.style.display = 'none';

    const label = document.createElement('div');
    label.className = 'wgv-section-label';
    if (section.labelHTML) {
      label.innerHTML = section.labelHTML;
    } else {
      label.textContent = section.label;
    }
    div.appendChild(label);

    for (const ctrl of (section.controls || [])) {
      const row = this._renderControl(ctrl);
      if (row) div.appendChild(row);
    }

    parent.appendChild(div);
    return div;
  }

  _renderControl(ctrl) {
    const row = document.createElement('div');
    row.className = 'wgv-row';
    if (ctrl.id) row.id = ctrl.id;
    if (ctrl.hidden) row.style.display = 'none';

    const labelEl = document.createElement('span');
    labelEl.className = 'wgv-row-label';
    labelEl.textContent = ctrl.label || '';
    if (ctrl.tip) labelEl.dataset.tip = ctrl.tip;
    row.appendChild(labelEl);

    switch (ctrl.type) {
      case 'select': this._buildSelect(ctrl, row); break;
      case 'range': this._buildRange(ctrl, row); break;
      case 'checkbox': this._buildCheckbox(ctrl, row); break;
      case 'button': this._buildButton(ctrl, row); break;
      case 'custom': if (ctrl.render) ctrl.render(row, this.store); break;
    }

    return row;
  }

  _buildSelect(ctrl, row) {
    const sel = document.createElement('select');
    for (const opt of ctrl.options) {
      const o = document.createElement('option');
      if (typeof opt === 'string') {
        o.value = opt; o.textContent = opt;
      } else {
        o.value = opt.value; o.textContent = opt.label;
      }
      sel.appendChild(o);
    }
    sel.value = this.store.get(ctrl.key) ?? '';
    sel.addEventListener('change', (e) => {
      this.store.set(ctrl.key, e.target.value);
      if (ctrl.onChange) ctrl.onChange(e.target.value, this.store);
    });
    row.appendChild(sel);
    this._controlEls[ctrl.key] = { input: sel };
  }

  _buildRange(ctrl, row) {
    const input = document.createElement('input');
    input.type = 'range';
    input.min = ctrl.min ?? 0;
    input.max = ctrl.max ?? 100;
    input.step = ctrl.step ?? 1;

    const rawVal = this.store.get(ctrl.key) ?? ctrl.min ?? 0;
    input.value = ctrl.toSlider ? ctrl.toSlider(rawVal) : rawVal;

    const valSpan = document.createElement('span');
    valSpan.className = 'wgv-val';
    valSpan.textContent = ctrl.format ? ctrl.format(rawVal) : rawVal;

    input.addEventListener('input', (e) => {
      const sliderVal = parseFloat(e.target.value);
      const storeVal = ctrl.fromSlider ? ctrl.fromSlider(sliderVal) : sliderVal;
      this.store.set(ctrl.key, storeVal);
      valSpan.textContent = ctrl.format ? ctrl.format(storeVal) : storeVal;
      if (ctrl.onChange) ctrl.onChange(storeVal, this.store);
    });

    row.appendChild(input);
    row.appendChild(valSpan);
    this._controlEls[ctrl.key] = { input, valSpan, ctrl };
  }

  _buildCheckbox(ctrl, row) {
    const label = document.createElement('label');
    label.style.cssText = 'flex:1; display:flex; align-items:center; gap:4px; cursor:pointer;';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = !!this.store.get(ctrl.key);
    cb.addEventListener('change', () => {
      this.store.set(ctrl.key, cb.checked);
      if (ctrl.onChange) ctrl.onChange(cb.checked, this.store);
    });
    label.appendChild(cb);
    if (ctrl.text) {
      const span = document.createElement('span');
      span.style.cssText = 'color:#888; font-size:0.75rem;';
      span.textContent = ctrl.text;
      label.appendChild(span);
    }
    row.appendChild(label);
    this._controlEls[ctrl.key] = { input: cb };
  }

  _buildButton(ctrl, row) {
    const btn = document.createElement('button');
    btn.style.cssText = 'background:#222; color:#4fc3f7; border:1px solid #444; border-radius:4px; padding:2px 8px; font-size:0.75rem; cursor:pointer;';
    btn.textContent = ctrl.text || ctrl.label;
    btn.addEventListener('click', () => {
      if (ctrl.onClick) ctrl.onClick(btn, this.store);
    });
    row.appendChild(btn);
  }

  _bindHotkey() {
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === this.hotkey || e.key === '~') {
        e.preventDefault();
        this.toggle();
      }
    });
  }

  _subscribeStore() {
    this.store.onAny((key, value) => {
      const el = this._controlEls[key];
      if (!el) return;
      const { input, valSpan, ctrl } = el;
      if (input.type === 'checkbox') {
        input.checked = !!value;
      } else if (input.tagName === 'SELECT') {
        input.value = value;
      } else if (input.type === 'range' && ctrl) {
        input.value = ctrl.toSlider ? ctrl.toSlider(value) : value;
        if (valSpan) valSpan.textContent = ctrl.format ? ctrl.format(value) : value;
      }
    });
  }

  _showTip(e) {
    if (!e.target || !e.target.closest) return;
    const el = e.target.closest('[data-tip]');
    if (!el) return;
    const r = el.getBoundingClientRect();
    this._tipEl.textContent = el.dataset.tip;
    this._tipEl.style.display = 'block';
    this._tipEl.style.left = r.left + 'px';
    this._tipEl.style.top = (r.top - 28) + 'px';
  }

  _hideTip(e) {
    if (!e.target || !e.target.closest) return;
    if (e.target.closest('[data-tip]')) {
      this._tipEl.style.display = 'none';
    }
  }

  // Convenience: get a section element by id to show/hide
  getSection(id) {
    return this._drawerEl.querySelector(`#${id}`);
  }
}

// Utility: update interp zone boxes from a render loop.
// mode: 0 (inactive), 1, 2, or 4. Call per-zone each frame.
const INTERP_BOX_COLORS = { 0: '#333', 1: '#ef5350', 2: '#ffca28', 4: '#66bb6a' };
export function renderInterpBoxes(elId, mode) {
  const el = document.getElementById(elId);
  if (!el) return;
  const color = INTERP_BOX_COLORS[mode] || '#666';
  let html = '';
  if (mode === 0) {
    html = '<div style="width:16px; height:6px; background:#222; border-radius:1px;"></div>';
  } else {
    for (let i = 0; i < mode; i++) {
      html += `<div style="width:16px; height:6px; background:${color}; border-radius:1px;"></div>`;
    }
  }
  el.innerHTML = html;
}

export { SettingsStore };
