// Curated starter models — shared by the setup wizard and the sidebar
// download button. All are 7B (untied embeddings, so tinyq4 loads them) and
// quantized Q4_K_M (~4.7 GB, fits comfortably in 8 GB+ VRAM).

export interface StarterModel {
  name: string;
  repo: string;   // HuggingFace repo
  file: string;   // GGUF filename
  size: string;
  desc: string;
}

export const STARTER_MODELS: StarterModel[] = [
  {
    name: "Qwen2.5-7B Instruct",
    repo: "bartowski/Qwen2.5-7B-Instruct-GGUF",
    file: "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
    size: "4.7 GB",
    desc: "Smart all-rounder — chat, writing, reasoning.",
  },
  {
    name: "Qwen2.5-Coder-7B",
    repo: "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
    file: "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
    size: "4.7 GB",
    desc: "Tuned for code — best for the Kairo agent.",
  },
];
