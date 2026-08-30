import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Built once as static HTML/JS/CSS and served by FastAPI's StaticFiles
  // mount alongside the API in production, rather than running a separate
  // Node server -- see backend/main.py's conditional frontend mount.
  output: "export",
};

export default nextConfig;
