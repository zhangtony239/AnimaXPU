import gc

import torch
from diffusers import (
    AnimaTextConditioner,
    AutoencoderKLQwenImage,
    CosmosTransformer3DModel,
)
from diffusers.loaders.single_file_utils import (
    convert_cosmos_transformer_checkpoint_to_diffusers,
    convert_wan_vae_to_diffusers,
)
from safetensors.torch import load_file

# 1. Test VAE conversion (qwen_image_vae.safetensors -> AutoencoderKLQwenImage)
print("--- 1. Testing VAE ---")
vae_sd = convert_wan_vae_to_diffusers(load_file("models/qwen_image_vae.safetensors"))
with torch.device("meta"):
    vae = AutoencoderKLQwenImage()
missing, unexpected = vae.load_state_dict(vae_sd, strict=False, assign=True)
print(f"VAE missing: {len(missing)}, unexpected: {len(unexpected)}")
del vae, vae_sd
gc.collect()

# 2. Test DiT & AnimaTextConditioner
print("\n--- 2. Testing DiT & AnimaTextConditioner ---")
net_sd = load_file("models/MiaoMiao RealSkin _Anima_v1.1zs_net.safetensors")

adapter_sd = {}
dit_raw_sd = {}
for k in list(net_sd.keys()):
    v = net_sd.pop(k)
    if k.startswith("net.llm_adapter."):
        adapter_sd[k.removeprefix("net.llm_adapter.")] = v
    else:
        dit_raw_sd[k] = v
del net_sd

with torch.device("meta"):
    conditioner = AnimaTextConditioner()
c_missing, c_unexpected = conditioner.load_state_dict(
    adapter_sd, strict=False, assign=True
)
print(f"Conditioner missing: {len(c_missing)}, unexpected: {len(c_unexpected)}")
del conditioner, adapter_sd

converted_dit_sd = convert_cosmos_transformer_checkpoint_to_diffusers(dit_raw_sd)
del dit_raw_sd
block_ids = {
    int(k.split(".")[1])
    for k in converted_dit_sd
    if k.startswith("transformer_blocks.")
}
print(f"Detected DiT num_layers: {max(block_ids) + 1}")

with torch.device("meta"):
    dit = CosmosTransformer3DModel(
        in_channels=16,
        out_channels=16,
        num_attention_heads=16,
        attention_head_dim=128,
        num_layers=max(block_ids) + 1,
        mlp_ratio=4.0,
        text_embed_dim=1024,
        adaln_lora_dim=256,
        max_size=(128, 240, 240),
        patch_size=(1, 2, 2),
        rope_scale=(1.0, 1.0, 1.0),
        concat_padding_mask=True,
        extra_pos_embed_type=None,
    )
d_missing, d_unexpected = dit.load_state_dict(
    converted_dit_sd, strict=False, assign=True
)
print(f"DiT missing: {len(d_missing)}, unexpected: {len(d_unexpected)}")
if d_missing:
    print("  missing sample:", d_missing[:10])
if d_unexpected:
    print("  unexpected sample:", d_unexpected[:10])
del dit, converted_dit_sd
gc.collect()

# 3. Test Qwen3Model
print("\n--- 3. Testing Qwen3Model ---")
try:
    from transformers import Qwen3Config, Qwen3Model
except ImportError:
    print("transformers is not installed yet (uv add transformers)")
    raise SystemExit(0)
txt_sd = load_file("models/anima_baseV10_txt.safetensors")
txt_sd_stripped = {k.removeprefix("model."): txt_sd.pop(k) for k in list(txt_sd.keys())}
del txt_sd
layer_ids = {int(k.split(".")[1]) for k in txt_sd_stripped if k.startswith("layers.")}
qwen_cfg = Qwen3Config(
    vocab_size=151936,
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=max(layer_ids) + 1,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
)
with torch.device("meta"):
    qwen = Qwen3Model(qwen_cfg)
q_missing, q_unexpected = qwen.load_state_dict(
    txt_sd_stripped, strict=False, assign=True
)
print(f"Qwen3 missing: {len(q_missing)}, unexpected: {len(q_unexpected)}")
if q_missing:
    print("  missing:", q_missing[:5])
if q_unexpected:
    print("  unexpected:", q_unexpected[:5])
