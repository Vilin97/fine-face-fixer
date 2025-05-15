"""
faithdiff_sr.py  –  single-image super-resolution with FaithDiff (SDXL base)

Usage (bash):
    python faithdiff_sr.py \
        --input  ./low_res.jpg \
        --output ./out_sr.png \
        --scale  4                 # 2, 3 or 4
        --steps  20                # ↓ to save VRAM / ↑ for quality
        --cpu_offload             # add on 12 GB cards if OOM
        --fp8                      # 8-bit UNet quantisation (saves ~35 %)
"""

import os, argparse, torch
from PIL import Image
from huggingface_hub import hf_hub_download

# --------------------------------------------------------------------- args
p = argparse.ArgumentParser()
p.add_argument("--input",  required=True)
p.add_argument("--output", required=True)
p.add_argument("--scale",  type=int, default=4, choices=[2,3,4])
p.add_argument("--steps",  type=int, default=20)
p.add_argument("--guidance", type=float, default=5.0)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--cpu_offload", action="store_true")
p.add_argument("--fp8", action="store_true")
args = p.parse_args()

# ---------------------------------------------------------------- checkpoints
chk_dir = "./checkpoints"
os.makedirs(chk_dir, exist_ok=True)

SDXL_PATH = os.path.join(chk_dir, "RealVisXL_V4.0")
VAE_PATH  = os.path.join(chk_dir, "sdxl-vae-fp16")
FD_BIN    = hf_hub_download("jychen9811/FaithDiff", "FaithDiff.bin",
                            local_dir=os.path.join(chk_dir, "FaithDiff"),
                            local_dir_use_symlinks=False)

# pull SDXL base + VAE once (then cached)
from diffusers import DiffusionPipeline, AutoencoderKL
_ = DiffusionPipeline.from_pretrained(
        "SG161222/RealVisXL_V4.0",
        cache_dir=SDXL_PATH,
        torch_dtype=torch.float16,
        variant="fp16")
_ = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        cache_dir=VAE_PATH,
        torch_dtype=torch.float16)

# ---------------------------------------------------------------- FaithDiff
from FaithDiff.create_FaithDiff_model import FaithDiff_pipeline   # :contentReference[oaicite:0]{index=0}
pipe = FaithDiff_pipeline(
           sdxl_path=SDXL_PATH,
           VAE_FP16_path=VAE_PATH,
           FaithDiff_path=FD_BIN,
           use_fp8=args.fp8)
pipe = pipe.to("cuda")

if args.cpu_offload:
    pipe.enable_model_cpu_offload()

# ---------------------------------------------------------------- run SR
torch.manual_seed(args.seed)
lr = Image.open(args.input).convert("RGB")
w, h = lr.size
target_w, target_h = w * args.scale, h * args.scale

out = pipe(
        lr_img   = lr,
        prompt   = "",              # no T2I prompt needed for pure SR
        negative_prompt = "",
        start_point     = "lr",      # start from low-res image
        guidance_scale  = args.guidance,
        num_inference_steps = args.steps,
        height = target_h,
        width  = target_w).images[0]

out.save(args.output)
print("⸨✓⸩ saved to", args.output)
