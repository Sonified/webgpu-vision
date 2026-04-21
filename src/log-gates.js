// Shared log-gate helper. Exports a gated `log(category, ...args)` and a
// category-aware `makeLogger(category, intervalMs)` for rate-limited sites.
// Each worker calls applyLogGatesFromUrl() at the top of its file so gate
// state is populated BEFORE any gated log fires.
// Main thread passes current gate state to workers via the spawn URL
// (`?gates=<urlencoded JSON>`); live updates go through postMessage.

// On the main thread, bind to the shared state object set up by the early
// inline script in the demo page so drawer toggles + localStorage-restored
// values take effect immediately. In workers (no window) start with a local
// all-on default; applyLogGatesFromUrl() overwrites it from ?gates=.
let currentGates = (typeof window !== 'undefined' && window.__wgvLogGates && window.__wgvLogGates.state)
  ? window.__wgvLogGates.state
  : { lifecycle: true, performance: true, tracking: true };

/** Gated log. If the category is disabled, nothing runs -- callers should
 *  inline their template literals into the args so message construction is
 *  also skipped. */
export function log(category, ...args) {
  if (!currentGates[category]) return;
  console.log(...args);
}

/** Rate-limited, category-gated logger. Returns a function that logs at most
 *  once per intervalMs. When the category is off, no message is built and no
 *  rate-limit state is touched. */
export function makeLogger(category, intervalMs = 2000) {
  let lastLog = 0;
  return function (...args) {
    if (!currentGates[category]) return;
    const now = performance.now();
    if (now - lastLog > intervalMs) {
      console.log(...args);
      lastLog = now;
    }
  };
}

/** Update gate state. Safe to call multiple times -- gates merge. */
export function applyLogGates(gates) {
  if (gates) currentGates = { ...currentGates, ...gates };
}

/** Read gates from the current global's location.search (workers use
 *  self.location; main thread uses window.location). Also installs a
 *  message handler for live gate updates from the main thread via
 *  `{ type: 'log_gates', gates }`. */
export function applyLogGatesFromUrl() {
  try {
    const search = (typeof self !== 'undefined' && self.location ? self.location.search : '') || '';
    const params = new URLSearchParams(search);
    const g = params.get('gates');
    if (g) applyLogGates(JSON.parse(decodeURIComponent(g)));
  } catch {}

  // Live gate updates from the spawning thread (dedicated worker only).
  if (typeof self !== 'undefined' && typeof window === 'undefined' && self.addEventListener) {
    self.addEventListener('message', (e) => {
      if (e && e.data && e.data.type === 'log_gates' && e.data.gates) {
        applyLogGates(e.data.gates);
      }
    });
  }
}

/** Main-thread helper: builds a Worker URL with current gate state encoded as
 *  a query string. Workers read this via applyLogGatesFromUrl(). */
export function workerUrlWithGates(urlBase) {
  try {
    const gates = (typeof window !== 'undefined' && window.__wgvLogGates && window.__wgvLogGates.state) || null;
    if (gates) {
      const url = new URL(urlBase);
      url.searchParams.set('gates', JSON.stringify(gates));
      return url;
    }
  } catch {}
  return urlBase;
}

/** Main-thread helper: register a worker so it receives live gate updates
 *  when the user toggles checkboxes in the UI. Call right after `new Worker`. */
export function registerWorkerForGateUpdates(worker) {
  try {
    if (typeof window !== 'undefined' && window.__wgvLogGates && window.__wgvLogGates.registerWorker) {
      window.__wgvLogGates.registerWorker(worker);
    }
  } catch {}
}
