// Saient model catalog — curated, license-clean GGUFs hosted on Vercel Blob.
// The mobile app (Models tab) fetches this to list downloadable models. Curation lives
// HERE so we control exactly what phones can grab — only models we can legally redistribute
// (Apache-2.0 / MIT), and only sizes that actually run well on a phone.

const BLOB = "https://3qqwzeu2cgtrtrdd.public.blob.vercel-storage.com/models";

const MODELS = [
  {
    id: "qwen3-1.7b-q4km",
    name: "Qwen3 1.7B",
    file: "qwen3-1.7b-q4km.gguf",
    url: `${BLOB}/qwen3-1.7b-q4km-7gXZcq6CxKGKMxB2ZXI7LEaWuaiZdO.gguf`,
    sizeBytes: 1282439264,
    quant: "Q4_K_M",
    params: "1.7B",
    tier: "Medium",
    license: "Apache-2.0",
    licenseUrl: "https://huggingface.co/Qwen/Qwen3-1.7B/blob/main/LICENSE",
    desc: "A stronger on-device all-rounder for chat and agent tasks.",
  },
];

export default function handler(req, res) {
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "public, max-age=300, s-maxage=600");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.status(200).json(MODELS);
}
