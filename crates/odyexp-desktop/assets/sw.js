var cacheName = 'egui-template-pwa';
var filesToCache = [
  './',
  './index.html',
  // disable this for now, it needs to match our wasm and js file names
  // currently trunk adds a random hash to the end of the file name
  // so this won't work and gives 404 errors
  // setting is in trunk.toml 
  // [build]
  // filehash = false // default is true

  //
  // './odyexp-desktop.js',
  // './odyexp-desktop_bg.wasm',
];

/* Start the service worker and cache all of the app's content */
self.addEventListener('install', function (e) {
  e.waitUntil(
    caches.open(cacheName).then(function (cache) {
      return cache.addAll(filesToCache);
    })
  );
});

/* Serve cached content when offline */
self.addEventListener('fetch', function (e) {
  e.respondWith(
    caches.match(e.request).then(function (response) {
      return response || fetch(e.request);
    })
  );
});
