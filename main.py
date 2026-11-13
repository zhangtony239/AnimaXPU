from tqdm import tqdm

# --- 配置区 ---
DIT_NAME = "MiaoMiao RealSkin _Anima_v1.1zs_net.safetensors"
VAE_NAME = "qwen_image_vae.safetensors"
TXT_NAME = "anima_baseV10_txt.safetensors"
OUTPUT_PATH = "C:\\Users\\Tony\\Downloads\\"
HEIGHT = 1560
WIDTH = 920
CFG_SCALE = 4
CFG_BATCH_SIZE = 2
STEPS = 30
NEGATIVE_PROMPT = "worst quality, low quality, score_1, score_2, score_3, artist name"
HOTWORDS = {
    "airki": "1girl,white hair,blue eyes,cat ears",
}

# fmt: off
with tqdm(total=23, desc="Importing dependencies") as pbar:
    from pathlib import Path
    pbar.update()
    import gc
    pbar.update()
    from random import randint
    pbar.update()
    from typing import cast
    pbar.update()
    import numpy as np
    pbar.update()
    from PIL import Image
    pbar.update()
    import torch
    pbar.update()
    import torch.nn.functional as F
    pbar.update()
    from safetensors.torch import load_file
    pbar.update()
    import transformers
    pbar.update()
    from transformers import Qwen2Tokenizer
    pbar.update()
    from transformers import Qwen3Config, Qwen3Model
    pbar.update()
    from transformers import T5Tokenizer
    pbar.update()
    from diffusers.utils import logging as diffusers_logging
    pbar.update()
    from diffusers.image_processor import VaeImageProcessor
    pbar.update()
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    pbar.update()
    from diffusers.models.transformers.transformer_cosmos import (
        CosmosTransformer3DModel,
    )
    pbar.update()
    from diffusers.models.condition_embedders.condition_embedder_anima import (
        AnimaTextConditioner,
    )
    pbar.update()
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
        AutoencoderKLQwenImage,
    )
    pbar.update()
    from diffusers.loaders.single_file_utils import (
        convert_cosmos_transformer_checkpoint_to_diffusers,
        convert_wan_vae_to_diffusers,
    )
    pbar.update()
    from tags.TagCompleter import init_tags
    pbar.update()
    from prompt_toolkit import prompt
    pbar.update()
    from prompt_toolkit.history import InMemoryHistory
    pbar.update()
# fmt: on

# 简化日志 & TF32 计算加速
diffusers_logging.disable_progress_bar()
transformers.utils.logging.set_verbosity_error()
# torch.set_float32_matmul_precision("high")

# --- 1. 载入 DiT & Text Conditioner (带 models/.cache/ 缓存) ---
dit_cache_dir = Path("models/.cache") / Path(DIT_NAME).stem
transformer_cache = dit_cache_dir / "transformer"
conditioner_cache = dit_cache_dir / "text_conditioner"

if (
    not (transformer_cache / "config.json").exists()
    or not (conditioner_cache / "config.json").exists()
):
    print(f"Converting DiT & Conditioner to cache: {dit_cache_dir}")
    net_sd = load_file(Path("models") / DIT_NAME)
    adapter_sd = {}
    dit_raw_sd = {}
    for k in list(net_sd.keys()):
        v = net_sd.pop(k)
        if k.startswith("net.llm_adapter."):
            adapter_sd[k.removeprefix("net.llm_adapter.")] = v
        else:
            dit_raw_sd[k] = v
    del net_sd

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        text_conditioner = AnimaTextConditioner()
        text_conditioner.load_state_dict(adapter_sd, strict=True, assign=True)
        text_conditioner.save_pretrained(conditioner_cache)
        del text_conditioner, adapter_sd
        gc.collect()

        dit_sd = convert_cosmos_transformer_checkpoint_to_diffusers(dit_raw_sd)
        del dit_raw_sd
        num_layers = (
            max(
                int(k.split(".")[1])
                for k in dit_sd
                if k.startswith("transformer_blocks.")
            )
            + 1
        )
        dit = CosmosTransformer3DModel(
            in_channels=16,
            out_channels=16,
            num_attention_heads=16,
            attention_head_dim=128,
            num_layers=num_layers,
            mlp_ratio=4.0,
            text_embed_dim=1024,
            adaln_lora_dim=256,
            max_size=(128, 240, 240),
            patch_size=(1, 2, 2),
            rope_scale=(1.0, 1.0, 1.0),
            concat_padding_mask=True,
            extra_pos_embed_type=None,
        )
        dit.load_state_dict(dit_sd, strict=True, assign=True)
        dit.save_pretrained(transformer_cache)
        del dit, dit_sd
        gc.collect()
    finally:
        torch.set_default_dtype(prev_dtype)

dit = CosmosTransformer3DModel.from_pretrained(
    transformer_cache, torch_dtype=torch.bfloat16
).to("xpu")  # type: ignore
text_conditioner = AnimaTextConditioner.from_pretrained(
    conditioner_cache, torch_dtype=torch.bfloat16
).to("xpu")  # type: ignore

# --- 2. 载入 VAE (带 models/.cache/ 缓存) ---
vae_cache = Path("models/.cache") / Path(VAE_NAME).stem
if not (vae_cache / "config.json").exists():
    print(f"Converting VAE to cache: {vae_cache}")
    vae_sd = convert_wan_vae_to_diffusers(load_file(Path("models") / VAE_NAME))
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        vae = AutoencoderKLQwenImage()
        vae.load_state_dict(vae_sd, strict=True, assign=True)
        vae.save_pretrained(vae_cache)
        del vae, vae_sd
        gc.collect()
    finally:
        torch.set_default_dtype(prev_dtype)

vae = AutoencoderKLQwenImage.from_pretrained(vae_cache, torch_dtype=torch.bfloat16).to(
    "xpu"  # type: ignore
)

# --- 3. 载入 Text Encoder & Tokenizers (带 models/.cache/ 缓存) ---
txt_cache = Path("models/.cache") / Path(TXT_NAME).stem
if not (txt_cache / "config.json").exists():
    print(f"Converting Text Encoder to cache: {txt_cache}")
    txt_sd = load_file(Path("models") / TXT_NAME)
    txt_sd = {k.removeprefix("model."): txt_sd.pop(k) for k in list(txt_sd.keys())}
    num_hidden_layers = (
        max(int(k.split(".")[1]) for k in txt_sd if k.startswith("layers.")) + 1
    )
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        qwen_cfg = Qwen3Config(
            vocab_size=151936,
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
        )
        text_encoder = Qwen3Model(qwen_cfg)
        text_encoder.load_state_dict(txt_sd, strict=True, assign=True)
        text_encoder.save_pretrained(txt_cache)
        del text_encoder, txt_sd
        gc.collect()
    finally:
        torch.set_default_dtype(prev_dtype)

text_encoder = Qwen3Model.from_pretrained(txt_cache, torch_dtype=torch.bfloat16).to(
    "xpu"  # type: ignore
)

qwen_tokenizer_cache = Path("models/.cache/tokenizers/qwen")
t5_tokenizer_cache = Path("models/.cache/tokenizers/t5")
if not qwen_tokenizer_cache.exists():
    qwen_tokenizer = Qwen2Tokenizer.from_pretrained(
        "circlestone-labs/Anima-Base-v1.0-Diffusers", subfolder="tokenizer"
    )
    qwen_tokenizer.save_pretrained(qwen_tokenizer_cache)
else:
    qwen_tokenizer = Qwen2Tokenizer.from_pretrained(qwen_tokenizer_cache)

if not t5_tokenizer_cache.exists():
    t5_tokenizer = T5Tokenizer.from_pretrained(
        "circlestone-labs/Anima-Base-v1.0-Diffusers", subfolder="t5_tokenizer"
    )
    t5_tokenizer.save_pretrained(t5_tokenizer_cache)
else:
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_tokenizer_cache)

scheduler = FlowMatchEulerDiscreteScheduler()
image_processor = VaeImageProcessor(vae_scale_factor=8)

history = InMemoryHistory()
completer = init_tags(hotwords=HOTWORDS)

gen = torch.Generator(device="xpu")
MAX_SEED = np.iinfo("int32").max


def encode_prompt(prompt_text: str, max_sequence_length: int = 512) -> torch.Tensor:
    qwen_inputs = qwen_tokenizer(
        [prompt_text],
        padding="longest",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    qwen_ids = qwen_inputs.input_ids.to("xpu")
    qwen_mask = qwen_inputs.attention_mask.to("xpu")
    qwen_embeds = text_encoder(
        input_ids=qwen_ids,
        attention_mask=qwen_mask,
        output_hidden_states=False,
    ).last_hidden_state.to(dtype=torch.bfloat16)
    qwen_embeds = qwen_embeds * qwen_mask.to(qwen_embeds).unsqueeze(-1)

    t5_inputs = t5_tokenizer(
        [prompt_text],
        padding="longest",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    t5_ids = t5_inputs.input_ids.to("xpu")
    t5_mask = t5_inputs.attention_mask.to("xpu")

    return text_conditioner(
        source_hidden_states=qwen_embeds,
        target_input_ids=t5_ids,
        target_attention_mask=t5_mask,
        source_attention_mask=qwen_mask,
    ).to(dtype=torch.bfloat16)


def draw(
    prompt_text: str,
    seed: int,
    num_inference_steps: int = STEPS,
    guidance_scale: float = CFG_SCALE,
):
    print(f"Current seed: {seed}")

    with torch.inference_mode():
        # 1. 文本编码阶段
        prompt_embeds = encode_prompt(prompt_text)
        negative_prompt_embeds = (
            encode_prompt(NEGATIVE_PROMPT) if guidance_scale > 1.0 else None
        )

        # 2. 潜变量与调度器初始化
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps).tolist()
        scheduler.set_timesteps(device="xpu", sigmas=sigmas)
        timesteps = scheduler.timesteps
        latent_h, latent_w = HEIGHT // 8, WIDTH // 8
        pad_h, pad_w = latent_h % 2, latent_w % 2
        latents = torch.randn(
            (1, 16, 1, latent_h, latent_w),
            generator=gen.manual_seed(seed),
            device="xpu",
            dtype=torch.float32,
        )
        padding_mask = latents.new_zeros(
            1, 1, (latent_h + pad_h) * 8, (latent_w + pad_w) * 8, dtype=torch.bfloat16
        )
        if pad_h:
            padding_mask[:, :, HEIGHT:, :] = 1.0
        if pad_w:
            padding_mask[:, :, :, WIDTH:] = 1.0

        cfg_embeds: torch.Tensor | None = None
        if negative_prompt_embeds is not None and CFG_BATCH_SIZE >= 2:
            seq_diff = prompt_embeds.shape[1] - negative_prompt_embeds.shape[1]
            if seq_diff > 0:
                negative_prompt_embeds = F.pad(
                    negative_prompt_embeds, (0, 0, 0, seq_diff)
                )
            elif seq_diff < 0:
                prompt_embeds = F.pad(prompt_embeds, (0, 0, 0, -seq_diff))
            cfg_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

        # 3. 去噪循环
        for t in tqdm(timesteps, desc="Denoising"):
            latent_model_input = latents.to(torch.bfloat16)
            if pad_h or pad_w:
                latent_model_input = F.pad(latent_model_input, (0, pad_w, 0, pad_h))
            timestep = (
                t.expand(latents.shape[0]).to(torch.bfloat16)
                / scheduler.config["num_train_timesteps"]
            )

            if cfg_embeds is not None:
                noise_pred_uncond, noise_pred_cond = dit(
                    hidden_states=torch.cat(
                        [latent_model_input, latent_model_input], dim=0
                    ),
                    timestep=timestep.expand(2),
                    encoder_hidden_states=cfg_embeds,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0].chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            elif negative_prompt_embeds is not None:
                noise_pred_cond = dit(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]
                noise_pred_uncond = dit(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
            else:
                noise_pred = dit(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    padding_mask=padding_mask,
                    return_dict=False,
                )[0]

            if pad_h or pad_w:
                noise_pred = noise_pred[:, :, :, :latent_h, :latent_w]

            latents = scheduler.step(
                cast(torch.FloatTensor, noise_pred),
                cast(torch.FloatTensor, t),
                cast(torch.FloatTensor, latents),
                return_dict=False,
            )[0]
            torch.xpu.synchronize()

        # 4. VAE 解码阶段
        latents = latents.to(vae.dtype)
        z_dim = vae.config["z_dim"]
        latents_mean = (
            torch.tensor(vae.config["latents_mean"])
            .view(1, z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config["latents_std"]).view(
            1, z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean

        image_tensor = vae.decode(latents, return_dict=False)[0][:, :, 0]
        images = cast(
            list[Image.Image],
            image_processor.postprocess(image_tensor, output_type="pil"),
        )
        image = images[0]

    image.save(f"{OUTPUT_PATH}{seed}.png")


if __name__ == "__main__":
    while True:
        try:
            prompts = prompt("Prompt: ", completer=completer, history=history).strip()
        except KeyboardInterrupt:
            continue
        if prompts in ["Q", "q", "exit"]:
            break
        elif len(prompts.split("seed")) > 1:
            seed = int(prompts.split("seed")[1].split(",")[0][1:])
            prompts = prompts.split("seed")[0] + ",".join(
                prompts.split("seed")[1].split(",")[1:]
            )
        else:
            seed = randint(0, MAX_SEED)
        prompt_tags = [t.strip() for t in prompts.split(",")]
        processed_tags = [HOTWORDS.get(t.lower(), t) for t in prompt_tags]
        prompts = ",".join(processed_tags)
        try:
            draw(prompts, seed)
        except KeyboardInterrupt:
            print("Drawing cancelled.")
            continue
