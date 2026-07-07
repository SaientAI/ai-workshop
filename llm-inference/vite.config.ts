import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte()],
  clearScreen: false,
  // overlay:false — a runtime error (e.g. a video OOM) must not throw the full-screen
  // dev error overlay that can only be cleared by reloading. The global handler in
  // App.svelte surfaces errors as dismissible toasts instead; compile errors still
  // print to the terminal running `tauri dev`.
  server: {
    port: 1421,
    strictPort: true,
    hmr: { overlay: false },
    watch: {
      ignored: [
        "**/.venvs/**",
        "**/tools/local-3d/vendor/**",
        "**/src-tauri/target/**",
        "**/assets/game-assets/**",
      ],
    },
  },
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    target: ["es2021", "chrome100", "safari13"],
    minify: !process.env.TAURI_DEBUG ? "esbuild" : false,
    sourcemap: !!process.env.TAURI_DEBUG,
  },
});
