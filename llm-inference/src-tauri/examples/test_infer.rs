use llm_inference::engine::{Engine, SamplingParams, stream_generate};
use std::path::Path;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};

#[tokio::main]
async fn main() {
    let model_path = std::env::args().nth(1)
        .expect("usage: test_infer <model.gguf> [gpu_layers] [ctx_size]");
    let gpu_layers: i32 = std::env::args().nth(2)
        .and_then(|s| s.parse().ok()).unwrap_or(999);
    let ctx_size: u32 = std::env::args().nth(3)
        .and_then(|s| s.parse().ok()).unwrap_or(4096);

    println!("Loading: {}  ngl={}  ctx={}", model_path, gpu_layers, ctx_size);

    let engine = Engine::load(Path::new(&model_path), gpu_layers, ctx_size)
        .await
        .expect("Failed to load engine");

    println!("Ready on port {}  model: {}", engine.port, engine.summary.name);
    println!("\nGenerating…\n");

    let messages = vec![
        serde_json::json!({"role": "system", "content": "You are a helpful assistant."}),
        serde_json::json!({"role": "user",   "content": "What is 2+2? Answer in one word."}),
    ];

    let params = SamplingParams {
        max_tokens: 64,
        temperature: 0.1,
        top_p: 0.9,
        top_k: 20,
        repeat_penalty: 1.1,
        seed: 42,
    };

    let stop = Arc::new(AtomicBool::new(false));
    let result = stream_generate(
        &engine.client, engine.port, &messages, &params,
        |piece, is_reasoning| {
            use std::io::Write;
            if is_reasoning { print!("\x1b[2m{}\x1b[0m", piece); } else { print!("{}", piece); }
            std::io::stdout().flush().ok();
        },
        |done, total| { eprint!("\rprefill {}/{}...", done + 1, total); },
        stop,
    ).await.expect("Generate failed");

    println!("\n\n--- Stats ---");
    println!("Tokens: {}", result.tokens_generated);
    println!("Speed:  {:.2} tok/s", result.tokens_per_sec);
    println!("Time:   {}ms", result.elapsed_ms);
}
