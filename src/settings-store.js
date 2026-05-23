// Reactive key-value store for pipeline settings.
// Supports engine-dependent profiles: when the profile key changes (e.g. 'backend'),
// all profiled settings swap to that engine's saved/default values automatically.

export class SettingsStore {
  constructor({ defaults = {}, profiles = {}, profileKey = 'backend', globalKeys = [] } = {}) {
    this._defaults = { ...defaults };
    this._profiles = profiles;         // { 'webgpu': {...}, 'mediapipe': {...} }
    this._profileKey = profileKey;     // which key determines active profile
    this._globalKeys = new Set(globalKeys); // keys that are NOT profiled
    this._savedProfiles = {};          // runtime-modified profile values
    this._values = {};
    this._listeners = {};
    this._anyListeners = new Set();

    // Initialize: merge defaults with active profile defaults
    const activeProfile = defaults[profileKey] || Object.keys(profiles)[0] || '';
    Object.assign(this._values, defaults, profiles[activeProfile] || {});
  }

  get(key) {
    return this._values[key];
  }

  set(key, value) {
    const old = this._values[key];
    if (old === value) return;
    this._values[key] = value;

    // If this is the profile key changing, swap all profiled values
    if (key === this._profileKey) {
      this._switchProfile(old, value);
    } else if (!this._globalKeys.has(key)) {
      // Save to the active profile
      const profile = this._values[this._profileKey];
      if (!this._savedProfiles[profile]) this._savedProfiles[profile] = {};
      this._savedProfiles[profile][key] = value;
    }

    this._fire(key, value, old);
  }

  _switchProfile(oldProfile, newProfile) {
    // Save current profiled values to old profile
    if (oldProfile && !this._savedProfiles[oldProfile]) {
      this._savedProfiles[oldProfile] = {};
    }
    if (oldProfile) {
      for (const [k, v] of Object.entries(this._values)) {
        if (k === this._profileKey || this._globalKeys.has(k)) continue;
        this._savedProfiles[oldProfile][k] = v;
      }
    }

    // Load new profile: saved values > profile defaults > store defaults
    const saved = this._savedProfiles[newProfile] || {};
    const profileDefaults = this._profiles[newProfile] || {};
    const changed = [];

    for (const key of Object.keys(this._defaults)) {
      if (key === this._profileKey || this._globalKeys.has(key)) continue;
      const newVal = saved[key] ?? profileDefaults[key] ?? this._defaults[key];
      const oldVal = this._values[key];
      if (newVal !== oldVal) {
        this._values[key] = newVal;
        changed.push([key, newVal, oldVal]);
      }
    }

    // Fire events for all changed profiled values
    for (const [key, val, old] of changed) {
      this._fire(key, val, old);
    }
  }

  _fire(key, value, old) {
    if (this._listeners[key]) {
      for (const fn of this._listeners[key]) fn(value, old);
    }
    for (const fn of this._anyListeners) fn(key, value, old);
  }

  on(key, fn) {
    if (!this._listeners[key]) this._listeners[key] = new Set();
    this._listeners[key].add(fn);
    return () => this._listeners[key].delete(fn);
  }

  onAny(fn) {
    this._anyListeners.add(fn);
    return () => this._anyListeners.delete(fn);
  }

  off(key, fn) {
    if (this._listeners[key]) this._listeners[key].delete(fn);
  }

  snapshot() {
    return { ...this._values };
  }

  batch(updates) {
    const changed = [];
    for (const [key, value] of Object.entries(updates)) {
      const old = this._values[key];
      if (old !== value) {
        this._values[key] = value;
        changed.push([key, value, old]);
      }
    }
    for (const [key, value, old] of changed) {
      this._fire(key, value, old);
    }
  }

  save(namespace = 'wgv') {
    const data = {
      global: {},
      profiles: this._savedProfiles,
    };
    // Save globals + current profile key
    for (const key of this._globalKeys) {
      data.global[key] = this._values[key];
    }
    data.global[this._profileKey] = this._values[this._profileKey];
    // Save current active profile state too
    const active = this._values[this._profileKey];
    if (!data.profiles[active]) data.profiles[active] = {};
    for (const [k, v] of Object.entries(this._values)) {
      if (k === this._profileKey || this._globalKeys.has(k)) continue;
      data.profiles[active][k] = v;
    }
    localStorage.setItem(namespace, JSON.stringify(data));
  }

  load(namespace = 'wgv') {
    const raw = localStorage.getItem(namespace);
    if (!raw) return;
    try {
      const data = JSON.parse(raw);
      if (data.profiles) this._savedProfiles = data.profiles;
      if (data.global) {
        for (const [k, v] of Object.entries(data.global)) {
          this._values[k] = v;
        }
      }
      // Load active profile
      const active = this._values[this._profileKey];
      const saved = this._savedProfiles[active] || {};
      const profileDefaults = this._profiles[active] || {};
      for (const key of Object.keys(this._defaults)) {
        if (key === this._profileKey || this._globalKeys.has(key)) continue;
        this._values[key] = saved[key] ?? profileDefaults[key] ?? this._defaults[key];
      }
    } catch { /* ignore corrupt storage */ }
  }
}
