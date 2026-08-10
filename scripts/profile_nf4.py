import time, torch

print(f"torch {torch.__version__}, device {torch.cuda.get_device_name(0)}, cap {torch.cuda.get_device_capability(0)}")

from diffusers.models.transformers.transformer_wan import WanTransformer3DModel

CACHE = "/home/tiny/projects/saient/desktop/data/config/saient/wan-transformer-4bit/Wan2.2-T2V-A14B-Diffusers"

t0 = time.time()
transformer = WanTransformer3DModel.from_pretrained(CACHE, torch_dtype=torch.bfloat16, device_map={"": 0})
transformer.eval()
print(f"transformer loaded in {time.time()-t0:.1f}s")

cfg = transformer.config
print("in_channels", cfg.in_channels, "text_dim", cfg.text_dim, "num_attention_heads", cfg.num_attention_heads)

# small dummy inputs: 1280x720 -> latent 90x160, 2 latent frames (keeps the profiled call short)
B, C, T, H, W = 1, cfg.in_channels, 2, 720 // 8, 1280 // 8
hidden_states = torch.randn(B, C, T, H, W, device="cuda", dtype=torch.bfloat16)
timestep = torch.tensor([500], device="cuda", dtype=torch.long)
seq_len = 226
encoder_hidden_states = torch.randn(B, seq_len, cfg.text_dim, device="cuda", dtype=torch.bfloat16)

print("warmup forward (not profiled)...")
t1 = time.time()
with torch.no_grad():
    for _ in range(2):
        _ = transformer(hidden_states=hidden_states, timestep=timestep, encoder_hidden_states=encoder_hidden_states, return_dict=False)
torch.cuda.synchronize()
print(f"warmup (2 fwd) done in {time.time()-t1:.1f}s")

print("profiling 3 forward passes...")
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    t2 = time.time()
    with torch.no_grad():
        for _ in range(3):
            _ = transformer(hidden_states=hidden_states, timestep=timestep, encoder_hidden_states=encoder_hidden_states, return_dict=False)
    torch.cuda.synchronize()
    print(f"profiled 3 fwd in {time.time()-t2:.1f}s")

print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40))
prof.export_chrome_trace("/tmp/claude-1000/-home-tiny/ccdac3e2-c1c7-4a38-80fb-f98d5da29583/scratchpad/nf4_trace.json")
print("trace saved")
