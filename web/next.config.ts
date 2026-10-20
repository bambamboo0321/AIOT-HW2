import type { NextConfig } from "next";

const config: NextConfig = {
  poweredByHeader: false,
  reactStrictMode: true,
  devIndicators: false,
  async headers() {
    return [{ source: "/browser-compute/worker.mjs", headers: [
      { key: "Content-Security-Policy", value: "default-src 'none'; script-src 'self' https://cdn.jsdelivr.net/pyodide/v314.0.7/full/ 'wasm-unsafe-eval'; connect-src 'self' https://cdn.jsdelivr.net/pyodide/v314.0.7/full/; worker-src 'none'; object-src 'none'" },
      { key: "X-Content-Type-Options", value: "nosniff" },
      { key: "Cache-Control", value: "no-store" },
    ] }];
  },
  turbopack: { root: process.cwd() },
};

export default config;
