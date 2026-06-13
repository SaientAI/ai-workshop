// Saient model catalog — curated, license-clean GGUFs hosted on Vercel Blob.
// The mobile app (Models tab) fetches this to list downloadable models. Curation lives
// HERE so we control exactly what phones can grab — only models we can legally redistribute
// (Apache-2.0 / MIT), and only sizes that actually run well on a phone.

const BLOB = "https://3qqwzeu2cgtrtrdd.public.blob.vercel-storage.com/models";

const MODELS = [
  {
    id: "qwen2.5-0.5b-q4km",
    name: "Qwen2.5 0.5B Instruct",
    file: "qwen2.5-0.5b-instruct-q4km.gguf",
    url: `${BLOB}/qwen2.5-0.5b-instruct-q4km-APCSlYEEZx3X3MkA36XulEzLWfP4Q6.gguf`,
    sizeBytes: 397808192,
    quant: "Q4_K_M",
    params: "0.5B",
    tier: "Tiny",
    license: "Apache-2.0",
    licenseUrl: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/LICENSE",
    desc: "Tiny & instant — runs on almost any phone.",
  },
  {
    id: "qwen2.5-1.5b-q4km",
    name: "Qwen2.5 1.5B Instruct",
    file: "qwen2.5-1.5b-instruct-q4km.gguf",
    url: `${BLOB}/qwen2.5-1.5b-instruct-q4km-8Xes1U1RzC6PKYG0eVdHbgP8juKhyY.gguf`,
    sizeBytes: 986048768,
    quant: "Q4_K_M",
    params: "1.5B",
    tier: "Medium",
    license: "Apache-2.0",
    licenseUrl: "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/main/LICENSE",
    desc: "The sweet spot — a great all-rounder for a phone.",
  },
];

export default function handler(req, res) {
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.setHeader("Cache-Control", "public, max-age=300, s-maxage=600");
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.status(200).json(MODELS);
}
