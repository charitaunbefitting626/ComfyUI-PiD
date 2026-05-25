"""
ComfyUI custom nodes for PiD (Pixel Diffusion Decoder).

PiD is a plug-and-play diffusion decoder that replaces VAE decoders,
turning latent representations directly into super-resolved pixels.

Nodes:
  - PiDDecode: Takes latent samples + prompt, decodes via PiD to produce
    high-resolution super-resolved images (e.g. 512px latent → 2048px output).
"""

import logging
import os
import sys

import torch

logger = logging.getLogger("PiD")

# ---------------------------------------------------------------------------
# Ensure the PiD repo root is on sys.path so `import pid` resolves.
# ComfyUI may or may not do this depending on version; be defensive.
# ---------------------------------------------------------------------------
_PID_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PID_ROOT not in sys.path:
    sys.path.insert(0, _PID_ROOT)

# ---------------------------------------------------------------------------
# Model cache — keyed by (backbone, ckpt_type) to avoid reloading.
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[tuple[str, str], tuple] = {}


def _get_available_backbones():
    """Return the list of backbone + ckpt_type combos that have checkpoints present."""
    from pid._src.inference.checkpoint_registry import PID_CHECKPOINT_REGISTRY

    available = []
    for (backbone, ckpt_type), ckpt_info in PID_CHECKPOINT_REGISTRY.items():
        ckpt_path = os.path.join(_PID_ROOT, ckpt_info.checkpoint_path)
        if os.path.exists(ckpt_path):
            available.append(f"{backbone}_{ckpt_type}")
    return available


def _load_pid_model(backbone: str, ckpt_type: str):
    """Load a PiD model, returning (model, config). Uses cache."""
    cache_key = (backbone, ckpt_type)
    if cache_key in _MODEL_CACHE:
        logger.info(f"PiD model cache hit: {backbone}/{ckpt_type}")
        return _MODEL_CACHE[cache_key]

    logger.info(f"Loading PiD model: backbone={backbone}, ckpt_type={ckpt_type} ...")

    from pid._src.inference.checkpoint_registry import get_pid_checkpoint
    from pid._src.utils.model_loader import load_model_from_checkpoint

    ckpt_info = get_pid_checkpoint(backbone, ckpt_type)
    experiment_name = ckpt_info.experiment
    checkpoint_path = ckpt_info.checkpoint_path

    # Resolve relative checkpoint path against the PiD repo root
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(_PID_ROOT, checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"PiD checkpoint not found at: {checkpoint_path}\n"
            f"Please download checkpoints first:\n"
            f"  cd {_PID_ROOT}\n"
            f"  huggingface-cli download nvidia/PiD --local-dir . --include \"checkpoints/*\""
        )

    config_file = os.path.join(_PID_ROOT, "pid", "_src", "configs", "pid", "config.py")

    model, config = load_model_from_checkpoint(
        experiment_name=experiment_name,
        checkpoint_path=checkpoint_path,
        config_file=config_file,
        enable_fsdp=False,
        experiment_opts=[],
        strict=False,
        load_ema_to_reg=False,
    )
    model.eval()

    _MODEL_CACHE[cache_key] = (model, config)
    logger.info(f"PiD model loaded successfully: {backbone}/{ckpt_type}")
    return model, config


class PiDDecode:
    """
    Decode latent samples using PiD (Pixel Diffusion Decoder).

    PiD replaces the standard VAE decoder with a conditional pixel-space
    diffusion model that produces super-resolved images in one pass.
    For example, a 512px Flux latent becomes a 2048px image (4× SR).

    Supported backbones:
      - flux: Flux 1 (16-ch VAE, 8× spatial compression)
      - flux2: Flux 2 (128-ch VAE, 16× spatial compression)
      - sd3: Stable Diffusion 3 (16-ch VAE, 8× spatial compression)
      - zimage: ZImage (reuses Flux 1's VAE)
      - rae: Representation Autoencoder / DINOv2 (768-ch RAE, 16× spatial compression)
      - scale_rae: Scale RAE / SigLIP-2 (768-ch RAE, 16× spatial compression)

    Supported checkpoint variants:
      - 2k: Original 2048px-trained decoder (512→2048 at 4× scale, or 256→2048 at 8× scale)
      - 2kto4k: Multi-resolution decoder (1024→4K at 4× scale)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "high quality, detailed",
                    "tooltip": "Text prompt describing the image content. Used by PiD's text encoder for conditioning.",
                }),
                "backbone": (["flux", "flux2", "sd3", "zimage", "rae", "scale_rae"], {
                    "default": "flux",
                    "tooltip": "VAE backbone. Must match the VAE/encoder used to encode the latent. zimage reuses flux's VAE.",
                }),
                "ckpt_type": (["2k", "2kto4k"], {
                    "default": "2k",
                    "tooltip": "2k: 512→2048px decoder. 2kto4k: 1024→4K decoder.",
                }),
                "scale": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Upscale factor. Output resolution = VAE_native × scale. "
                               "Default 4 means 512→2048 for flux/zimage.",
                }),
                "pid_inference_steps": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 20,
                    "step": 1,
                    "tooltip": "Number of denoising steps for PiD. 4 is optimal for the distilled checkpoints.",
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 15.0,
                    "step": 0.1,
                    "tooltip": "Classifier-free guidance scale for PiD.",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFF,
                    "tooltip": "Random seed for the PiD diffusion process.",
                }),
            },
            "optional": {
                "degrade_sigma": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Noise level indicator. 0.0 = clean latent (standard KSampler output). "
                               "Higher values indicate the latent has more noise (e.g. from early termination).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "PiD"
    DESCRIPTION = (
        "Decode latent samples using PiD (Pixel Diffusion Decoder). "
        "Replaces standard VAE Decode with a diffusion-based decoder that "
        "produces super-resolved images (e.g. 512px → 2048px)."
    )

    def decode(
        self,
        latent: dict,
        prompt: str,
        backbone: str,
        ckpt_type: str,
        scale: int,
        pid_inference_steps: int,
        cfg_scale: float,
        seed: int,
        degrade_sigma: float = 0.0,
    ):
        # ---- Load model (cached) ----
        model, config = _load_pid_model(backbone, ckpt_type)

        # Move model, text encoder, and any unregistered tensors to CUDA (GPU)
        model.to("cuda")
        if hasattr(model, "text_encoder") and model.text_encoder is not None:
            model.text_encoder.to("cuda")
        if hasattr(model, "_null_caption_embs") and isinstance(model._null_caption_embs, torch.Tensor):
            model._null_caption_embs = model._null_caption_embs.to("cuda")

        # ---- Extract latent tensor from ComfyUI format ----
        # ComfyUI latents: {"samples": tensor} where tensor is [B, C, H, W]
        latent_tensor = latent["samples"]
        B, C, zH, zW = latent_tensor.shape

        logger.info(
            f"PiD decode: latent shape={latent_tensor.shape}, backbone={backbone}, "
            f"ckpt_type={ckpt_type}, scale={scale}, steps={pid_inference_steps}"
        )

        # ---- Determine VAE spatial compression factor dynamically ----
        if hasattr(model, "vae_encoder") and model.vae_encoder is not None:
            vae_compression = getattr(model.vae_encoder, "spatial_compression_factor", 8)
            if callable(vae_compression):
                try:
                    vae_compression = vae_compression()
                except Exception:
                    pass
        else:
            # Fallback based on backbone name if vae_encoder isn't present
            if backbone in ("flux", "sd3", "zimage"):
                vae_compression = 8
            elif backbone in ("flux2", "rae", "scale_rae"):
                vae_compression = 16
            else:
                vae_compression = 8

        # Compute target image size
        target_h = zH * vae_compression * scale
        target_w = zW * vae_compression * scale
        image_size = (target_h, target_w)

        logger.info(
            f"PiD decode: latent {zH}×{zW} → vae_native {zH * vae_compression}×{zW * vae_compression} "
            f"→ PiD output {target_h}×{target_w}"
        )

        # ---- Build data batch for PiD ----
        # PiD expects:
        #   - input_caption_key (default "caption"): list of prompt strings
        #   - LQ_latent: the VAE latent [B, C, zH, zW] in normalized space
        #   - LQ_video_or_image: zeros placeholder (model uses latent-only conditioning)
        #   - degrade_sigma: noise level [B] tensor
        caption_key = model.config.input_caption_key
        captions = [prompt] * B

        data_batch = {
            caption_key: captions,
            "LQ_latent": latent_tensor.to(dtype=torch.bfloat16, device="cuda"),
            "LQ_video_or_image": torch.zeros(
                B, 3, zH * vae_compression, zW * vae_compression,
                dtype=torch.bfloat16, device="cuda",
            ),
            "degrade_sigma": torch.tensor(
                [float(degrade_sigma)] * B, device="cuda", dtype=torch.float32,
            ),
        }

        # ---- Run PiD decode ----
        import comfy.utils
        pbar = comfy.utils.ProgressBar(pid_inference_steps)

        def progress_callback(step, total_steps):
            pbar.update_absolute(step + 1, total_steps, None)

        with torch.no_grad():
            samples = model.generate_samples_from_batch(
                data_batch,
                cfg_scale=cfg_scale,
                num_steps=pid_inference_steps,
                seed=seed,
                shift=None,
                image_size=image_size,
                callback=progress_callback,
            )

        # ---- Convert PiD output to ComfyUI IMAGE format ----
        # PiD output: [B, 3, 1, H, W] in [-1, 1] (5D with T=1)
        # ComfyUI IMAGE: [B, H, W, 3] in [0, 1]
        output = samples.float().clamp(-1, 1)

        # Squeeze the temporal dimension if present
        if output.ndim == 5:
            output = output.squeeze(2)  # [B, 3, H, W]

        # Convert [-1, 1] → [0, 1] and rearrange to [B, H, W, C]
        output = (output + 1.0) / 2.0
        output = output.permute(0, 2, 3, 1).cpu()  # [B, H, W, 3]

        # Move model and its text encoder back to CPU (RAM) to free up VRAM
        model.to("cpu")
        if hasattr(model, "text_encoder") and model.text_encoder is not None:
            model.text_encoder.to("cpu")
        if hasattr(model, "_null_caption_embs") and isinstance(model._null_caption_embs, torch.Tensor):
            model._null_caption_embs = model._null_caption_embs.to("cpu")

        # Clear CUDA memory cache
        torch.cuda.empty_cache()

        return (output,)


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "PiDDecode": PiDDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PiDDecode": "PiD Decode (Pixel Diffusion)",
}
