import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiTarget = process.env.TG_RADAR_UI_API_TARGET || "http://127.0.0.1:18081";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    allowedHosts: true,
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, "")
      }
    }
  }
});
