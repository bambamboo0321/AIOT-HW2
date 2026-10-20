import type { NextConfig } from "next";

const config: NextConfig = {
  poweredByHeader: false,
  reactStrictMode: true,
  devIndicators: false,
  // Safari checks static module-Worker imports against worker-src as well.
  // Only same-origin dependencies are allowed here; CDN scripts stay governed by script-src.
  async headers() {
    return [{ source: "/browser-compute/worker.mjs", headers: [
      { key: "Content-Security-Policy", value: "default-src 'none'; script-src 'self' https://cdn.jsdelivr.net/pyodide/v314.0.7/full/ 'wasm-unsafe-eval'; connect-src 'self' https://cdn.jsdelivr.net/pyodide/v314.0.7/full/; worker-src 'self'; object-src 'none'" },
      { key: "X-Content-Type-Options", value: "nosniff" },
      { key: "Cache-Control", value: "no-store" },
    ] }];
  },
  turbopack: { root: process.cwd() },
};

export default config;
