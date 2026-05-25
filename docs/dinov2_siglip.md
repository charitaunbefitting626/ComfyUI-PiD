# DINOv2-RAE / SigLIP-2 Scale-RAE LDM backbones

The `dinov2` and `siglip` entry points (`from_clean_dinov2`,
`from_clean_siglip`, `from_ldm_dinov2`, `from_ldm_siglip`) wrap two
latent-diffusion models that are **not** distributed through `diffusers` —
the upstream class-conditional ImageNet-512 [RAE](https://github.com/bytetriper/RAE)
and the text-conditional 256px [Scale-RAE](https://github.com/ZitengWangNYU/Scale-RAE).
`from_clean_*` needs the RAE / Scale-RAE tokenizer decoder weights only;
`from_ldm_*` additionally needs the upstream LDM repos on `sys.path`.

> [!NOTE]
> LDM in vision encoder space is hard to train. When the latent itself is
> highly unstructured and unreasonable, RAE decoder produces unsatisfactory
> results, and PiD cannot correct it as well.

## Installation

```bash
# 1) Clone the upstream repos NEXT TO the pid repo (as siblings, not inside).
#    Run these from the directory that *contains* your pid checkout so the
#    repos land at ../RAE and ../Scale-RAE relative to the pid working tree —
#    this is the default the inference scripts look for, and keeps the pid
#    working tree clean. Any other location works as long as you point the
#    RAE_REPO_PATH / SCALE_RAE_REPO_PATH env vars (or CLI flags) at it.
cd ..
git clone https://github.com/bytetriper/RAE.git
git clone https://github.com/ZitengWangNYU/Scale-RAE.git
cd pid

# 2) Install Scale-RAE (--no-deps because its pyproject pins torch/torchvision/
#    transformers/tokenizers that would clobber the rest of the env). Then add
#    the runtime deps the upstream code actually needs, plus pin transformers
#    to 4.57.x — Scale-RAE's custom Qwen LM is tightly coupled to the 4.x API
#    and silently produces garbage embeddings on transformers 5.x.
#
#    Expected: the second `pip install` prints ~10 lines of
#      "scale-rae 1.0.0 requires <pkg>==<old-version>, but you have …"
#    These warnings are by design — Scale-RAE's pyproject pins ancient
#    versions of peft / torchtext / accelerate / transformers / tokenizers
#    that we deliberately ignore via `--no-deps` above. The PiD inference
#    code only exercises the parts of Scale-RAE that work with the newer
#    versions; from_ldm_siglip / from_clean_siglip have been verified
#    end-to-end against the versions installed below.
pip install --no-deps -e ../Scale-RAE
pip install torchdiffeq timm omegaconf ezcolorlog shortuuid open_clip_torch accelerate

# 3) Point the demos at the repos. The CLI flags fall back to these env vars,
#    which default to ../RAE and ../Scale-RAE (sibling of the pid working tree).
export RAE_REPO_PATH=$(realpath ../RAE)
export SCALE_RAE_REPO_PATH=$(realpath ../Scale-RAE)

# 4) Download the RAE weights. Scale-RAE weights are downloaded automatically.
cd $RAE_REPO_PATH
hf download nyu-visionx/RAE-collections \
    --local-dir models
```

## `from_ldm_*`: class / text → upstream LDM → PiD decode

Class-conditional example (DINOv2-RAE, ImageNet-512):

```bash
export RAE_REPO_PATH=$(realpath ../RAE)
PYTHONPATH=. python -m pid._src.inference.from_ldm_dinov2 \
    --load_ema_to_reg \
    --rae_class_ids 207 281 387 \
    --num_inference_steps 50 --save_xt_steps 44 46 48 \
    --output_dir ./results/official_demo/dinov2 \
    --cfg_scale 1 --pid_inference_steps 4 --scale 4
```

Text-conditional example (Scale-RAE, 256 → 2048 at 8×):

```bash
export SCALE_RAE_REPO_PATH=$(realpath ../Scale-RAE)
PYTHONPATH=. python -m pid._src.inference.from_ldm_siglip \
    --load_ema_to_reg \
    --prompt "A cat sitting on a windowsill at sunset" \
    --save_xt_steps 44 46 48 \
    --output_dir ./results/official_demo/siglip \
    --cfg_scale 1 --pid_inference_steps 4 --scale 8
```

Suggested step counts (see each script's docstring for the exact recipe):

| Backbone | LDM steps flag          | Default steps | `--save_xt_steps` (example) |
|----------|-------------------------|---------------|-----------------------------|
| dinov2   | `--num_inference_steps` | 50            | `44 46 48`                  |
| siglip   | (no flag; LM-driven)    | —             | `44 46 48`                  |

## `from_clean_*`: image → encode → PiD decode

The `dinov2` and `siglip` `from_clean_*` flows accept the same flags as the
diffusers backbones (see the main README) but with different
`--input_resolution` and `--scale` defaults to match each tokenizer's native
interface:

- `dinov2` → `--input_resolution 512 --scale 4` (512 → 2048)
- `siglip` → `--input_resolution 256 --scale 8` (256 → 2048)

Keep `--degrade_sigmas` in `[0, 0.5]` — RAE's normalized DINOv2 / SigLIP-2
features tolerate less added noise than standard VAE latents.
