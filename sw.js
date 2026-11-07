// Seeyra service worker: network-first, so you always get the newest page.
// The cache is only used when the network is unavailable (offline fallback).
const V = "seeyra-v1";
const SHELL = ["./", "index.html", "manifest.webmanifest", "icon-192.png", "icon-512.png"];

self.addEventListener("install", (e) => {
  self.skipWaiting();
  e.waitUntil(caches.open(V).then((c) => c.addAll(SHELL)).catch(() => {}));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== V).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const r = e.request;
  if (r.method !== "GET") return;                       // POST (Gemini, save) goes straight to the network
  const u = new URL(r.url);
  if (u.origin !== location.origin) return;             // other sites (CDN, worker) are not touched
  if (u.pathname.indexOf("/cifar_model/") > -1) return; // big model files: browser handles them
  e.respondWith(
    fetch(r)
      .then((res) => {
        if (res && res.ok && res.type === "basic") {
          const copy = res.clone();
          caches.open(V).then((c) => c.put(r, copy)).catch(() => {});
        }
        return res;
      })
      .catch(() =>
        caches.match(r).then((m) => m || (r.mode === "navigate" ? caches.match("index.html") : undefined))
      )
  );
});
