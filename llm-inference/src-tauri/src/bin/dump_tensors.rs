fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_tensors <model.gguf>");
    let g = llm_inference::gguf::GgufFile::open(std::path::Path::new(&path))
        .expect("cannot open gguf");
    let s = g.summary();
    println!("name:         {}", s.name);
    println!("architecture: {}", s.architecture);
    println!("quant:        {}", s.quant);
    println!("ctx_length:   {}", s.context_length);
    println!("blocks:       {}", s.block_count);
    println!("embedding:    {}", s.embedding_length);
    println!("tensors:      {}", s.tensor_count);
    println!("version:      {}", s.version);
}
