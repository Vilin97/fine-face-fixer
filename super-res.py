#%%
from huggingface_hub import hf_hub_download

# Download the FaithDiff model weights
model_file = hf_hub_download(
    repo_id="jychen9811/FaithDiff",
    filename="FaithDiff.bin",
    local_dir="./proc_data/faithdiff",
    local_dir_use_symlinks=False
)
#%%
"The base diffusion model"
from diffusers import DiffusionPipeline

# Save to checkpoints directory
pipe = DiffusionPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0",
    cache_dir="./checkpoints/RealVisXL_V4.0",  # This forces download to local folder
    local_files_only=False  # Ensure it fetches from Hugging Face
)
pipe = pipe.to("cuda")

#%%
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 4k"
image = pipe(prompt).images[0]
image

#%%
"The VAE"
from diffusers import AutoencoderKL
import torch


vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    cache_dir="./checkpoints/sdxl-vae-fp16-fix",  # Save locally
    torch_dtype=torch.float16
)
vae = vae.to("cuda")

#%%
import random
import numpy as np
import torch
from diffusers import DiffusionPipeline, AutoencoderKL
from huggingface_hub import hf_hub_download
from diffusers.utils import load_image
from PIL import Image

device = "cuda"
dtype = torch.float16
MAX_SEED = np.iinfo(np.int32).max

# Download weights for FaithDiff
model_file = hf_hub_download(
    "jychen9811/FaithDiff",
    filename="FaithDiff.bin",
    local_dir="./proc_data/faithdiff", 
    local_dir_use_symlinks=False
)

# Load VAE (must match the model version)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=dtype)

# Load custom pipeline (put mixture_tiling_sdxl.py in the directory)
model_id = "SG161222/RealVisXL_V4.0"
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype,
    vae=vae,
    unet=None,
    custom_pipeline="pipeline_faithdiff_stable_diffusion_xl",  # This is important!
    use_safetensors=True,
    variant="fp16",
).to(device)

# Load UNet and FaithDiff weights
from pipeline_faithdiff_stable_diffusion_xl import UNet2DConditionModel
#%%
pipe.unet = UNet2DConditionModel.from_pretrained(
    model_id, subfolder="unet", variant="fp16", use_safetensors=True
)
pipe.unet.load_additional_layers(weight_path="proc_data/faithdiff/FaithDiff.bin", dtype=dtype)

# pipe.unet = pipe.unet_model.from_pretrained(model_id, subfolder="unet", variant="fp16", use_safetensors=True)
pipe.unet.load_additional_layers(weight_path="proc_data/faithdiff/FaithDiff.bin", dtype=dtype)

# Enable tiling (for large images)
pipe.set_encoder_tile_settings()
pipe.enable_vae_tiling()

# Set a scheduler (optional, but recommended)
from diffusers import UniPCMultistepScheduler
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

#%%
# Load your low-res image (local file or URL)
lq_image = load_image("your_low_res_image.png")  # or use a URL

# Choose upscaling factor (e.g., 2x)
upscale = 2
original_height = lq_image.height
original_width = lq_image.width
width = original_width * upscale
height = original_height * upscale

# Resize image to target size (Lanczos recommended)
image = lq_image.resize((width, height), Image.LANCZOS)
input_image, width_init, height_init, width_now, height_now = pipe.check_image_size(image)
