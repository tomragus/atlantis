// Puts the Next dev server (:3000) and the FastAPI backend (:8000) behind
// one origin for local dev, the same way production has them share one
// origin (see backend/main.py's conditional frontend mount).
//
// Without this, the browser sees frontend and backend as different sites
// (different hostnames or, per-browser, blocked as third-party even with
// SameSite=None), so the session_id cookie backend/main.py's
// session_cookie_middleware sets never survives past the first response --
// confirmed live: every request after the first looked like a brand new,
// empty session, breaking /generate (can't find the run /tag just created)
// and history (always empty on reload) even for a single developer working
// alone, not just for testing multi-visitor isolation.
//
// next.config.ts sets output: "export" (required for the Docker build),
// which makes Next's own dev-time proxy features (rewrites, middleware/
// proxy.ts) hard errors under `next dev` -- see this repo's pinned Next
// version's own docs at node_modules/next/dist/docs/01-app/02-guides/
// static-exports.md, "Unsupported Features". Hence a standalone process
// instead of a next.config.ts rewrite.
//
// Run alongside `npm run dev` and the backend's `uvicorn main:app --reload`,
// then open http://127.0.0.1:3001 (not :3000) in the browser.
const http = require("http");
const net = require("net");
const httpProxy = require("http-proxy");

const PROXY_PORT = 3001;
const FRONTEND_TARGET = "http://127.0.0.1:3000";
const BACKEND_TARGET = "http://127.0.0.1:8000";

// Keep in sync with backend/main.py's registered routes. Anything not
// listed here falls through to the frontend, matching how production's
// StaticFiles mount at "/" is the catch-all behind every real API route.
const BACKEND_PATH_PREFIXES = [
  "/api",
  "/tag",
  "/generate",
  "/history",
  "/genres",
  "/algorithms",
  "/test-images",
  "/test-image-files",
  "/test-image-thumbnails",
  "/files",
];

function isBackendPath(url) {
  return BACKEND_PATH_PREFIXES.some((prefix) => url === prefix || url.startsWith(`${prefix}/`) || url.startsWith(`${prefix}?`));
}

const proxy = httpProxy.createProxyServer({});

proxy.on("error", (err, req, res) => {
  console.error(`proxy error for ${req.url}:`, err.message);
  if (!res.writeHead) return;
  if (isBackendPath(req.url)) {
    // lib.ts's apiRequest always does response.json() before checking
    // response.ok -- a plain-text body here throws a SyntaxError there
    // instead of surfacing the fallback message the UI was designed to
    // show (e.g. "Could not analyze the image with siglip."), so this has
    // to look like a JSON error response, not a human-readable one.
    res.writeHead(502, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ detail: "Bad gateway (is the backend running?)" }));
  } else {
    res.writeHead(502);
    res.end("Bad gateway (is the frontend dev server running?)");
  }
});

const server = http.createServer((req, res) => {
  const target = isBackendPath(req.url) ? BACKEND_TARGET : FRONTEND_TARGET;
  // changeOrigin rewrites the Host header to match the real target
  // (127.0.0.1:3000/8000) instead of forwarding the browser's original
  // 127.0.0.1:3001 -- without it Next dev's own Host-based dev-origin
  // check rejects the request.
  proxy.web(req, res, { target, changeOrigin: true });
});

// Next dev's HMR websocket lives on the frontend target. Relayed with a raw
// TCP pipe rather than http-proxy's own ws() support: that path hit a
// "Parse Error: Expected HTTP/, RTSP/ or ICE/" against this Node version
// regardless of Origin/Host rewriting, and was severe enough in testing to
// stop the page's mount-effect fetches from ever running at all, not just
// break hot-reload. Manually replaying the upgrade request over a plain
// socket sidesteps http-proxy's WS code entirely.
server.on("upgrade", (req, socket, head) => {
  const target = new URL(FRONTEND_TARGET);
  const targetSocket = net.connect(Number(target.port), target.hostname, () => {
    // "localhost", not target.hostname (127.0.0.1): Next dev's WS-specific
    // dev-origin allowlist rejects the handshake outright with a Host or
    // Origin naming this proxy's own 127.0.0.1:3001 -- confirmed both
    // headers are checked by replaying the raw upgrade request by hand
    // with each header varied independently. Both get rewritten to match
    // what Next's dev server itself expects; the TCP connection still goes
    // to target.hostname regardless of what these headers say.
    const headers = {
      ...req.headers,
      host: `localhost:${target.port}`,
      origin: `http://localhost:${target.port}`,
    };
    const headerLines = Object.entries(headers)
      .map(([key, value]) => `${key}: ${value}`)
      .join("\r\n");
    targetSocket.write(`${req.method} ${req.url} HTTP/1.1\r\n${headerLines}\r\n\r\n`);
    if (head && head.length) targetSocket.write(head);
    targetSocket.pipe(socket);
    socket.pipe(targetSocket);
  });
  targetSocket.on("error", () => socket.destroy());
  socket.on("error", () => targetSocket.destroy());
});

server.listen(PROXY_PORT, () => {
  console.log(`Dev proxy on http://127.0.0.1:${PROXY_PORT} -- frontend ${FRONTEND_TARGET}, backend ${BACKEND_TARGET}`);
});
