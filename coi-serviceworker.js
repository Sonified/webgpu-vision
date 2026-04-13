/*! coi-serviceworker - enables crossOriginIsolated + asset caching */
if (typeof window === 'undefined') {
  const CACHE_NAME = 'webgpu-vision-v1';

  // File extensions worth caching (models, WASM, JS modules).
  // HTML is NOT cached so deploys take effect immediately.
  const CACHEABLE = /\.(wasm|onnx|mjs|js|json|png|jpg|css)(\?|$)/;

  self.addEventListener("install", () => self.skipWaiting());
  self.addEventListener("activate", (e) => {
    // Clean up old cache versions
    e.waitUntil(
      caches.keys().then(keys =>
        Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
      ).then(() => self.clients.claim())
    );
  });

  self.addEventListener("fetch", (e) => {
    if (e.request.cache === "only-if-cached" && e.request.mode !== "same-origin") return;
    if (e.request.method !== "GET") return;

    const url = new URL(e.request.url);
    const shouldCache = CACHEABLE.test(url.pathname);

    e.respondWith((async () => {
      // Check cache first for cacheable assets
      if (shouldCache) {
        const cached = await caches.match(e.request);
        if (cached) return cached;
      }

      const res = await fetch(e.request);
      if (res.status === 0) return res;

      // Add COEP/COOP headers for crossOriginIsolated
      const headers = new Headers(res.headers);
      headers.set("Cross-Origin-Embedder-Policy", "credentialless");
      headers.set("Cross-Origin-Opener-Policy", "same-origin");

      const response = new Response(res.body, {
        status: res.status,
        statusText: res.statusText,
        headers,
      });

      // Cache the response for next time (clone because body can only be read once)
      if (shouldCache && res.status === 200) {
        const cache = await caches.open(CACHE_NAME);
        cache.put(e.request, response.clone());
      }

      return response;
    })().catch((err) => console.error(err)));
  });
} else {
  // Main thread: register SW, reload once it takes control
  (async () => {
    if (!("serviceWorker" in navigator)) return;

    const registration = await navigator.serviceWorker.register(
      new URL("coi-serviceworker.js", document.currentScript.src)
    );

    // If already controlling, nothing to do
    if (navigator.serviceWorker.controller) return;

    // Wait for the SW to become active
    const sw = registration.installing || registration.waiting || registration.active;
    if (!sw) return;

    if (sw.state === "activated") {
      window.location.reload();
    } else {
      sw.addEventListener("statechange", () => {
        if (sw.state === "activated") window.location.reload();
      });
    }
  })();
}
