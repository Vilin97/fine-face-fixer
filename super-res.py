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

#%%
prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 4k"
image = pipe(prompt).images[0]

#%%
"The VAE"
from diffusers import AutoencoderKL
import torch


vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    cache_dir="./checkpoints/sdxl-vae-fp16-fix",  # Save locally
    torch_dtype=torch.float16
)

#%%
"CLIP encoder, optional"
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

model = AutoModelForZeroShotImageClassification.from_pretrained(
    "openai/clip-vit-large-patch14-336",
    cache_dir="./checkpoints/clip-vit-large-patch14-336"
)
processor = AutoProcessor.from_pretrained(
    "openai/clip-vit-large-patch14-336",
    cache_dir="./checkpoints/clip-vit-large-patch14-336"
)

#%%
"MMLM, Optional"
from transformers import AutoProcessor, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "liuhaotian/llava-v1.5-13b",
    cache_dir="./checkpoints/llava-v1.5-13b"
)
processor = AutoProcessor.from_pretrained(
    "liuhaotian/llava-v1.5-13b",
    cache_dir="./checkpoints/llava-v1.5-13b"
)



#%%
import torch
import random
import numpy as np
from PIL import Image
from diffusers import (
    DiffusionPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from huggingface_hub import hf_hub_download
from diffusers.utils import load_image

# Device and data type settings
device = "cuda"
dtype = torch.float16
MAX_SEED = np.iinfo(np.int32).max

# Model identifiers
base_model_id = "SG161222/RealVisXL_V4.0"
vae_model_id = "madebyollin/sdxl-vae-fp16-fix"
faithdiff_repo = "jychen9811/FaithDiff"
faithdiff_filename = "FaithDiff.bin"

# Download FaithDiff weights
faithdiff_path = hf_hub_download(
    repo_id=faithdiff_repo,
    filename=faithdiff_filename,
    local_dir="./checkpoints/faithdiff",
    local_dir_use_symlinks=False,
)

# Load VAE
vae = AutoencoderKL.from_pretrained(vae_model_id, torch_dtype=dtype)

# Load base pipeline without UNet
pipe = DiffusionPipeline.from_pretrained(
    base_model_id,
    vae=vae,
    unet=None,
    torch_dtype=dtype,
    variant="fp16",
    use_safetensors=True,
    custom_pipeline="mixture_tiling_sdxl",
).to(device)

# Load UNet separately
pipe.unet = UNet2DConditionModel.from_pretrained(
    base_model_id,
    subfolder="unet",
    torch_dtype=dtype,
    variant="fp16",
    use_safetensors=True,
)

# Define and apply the function to load additional FaithDiff layers
def load_additional_layers(unet, weight_path, dtype):
    state_dict = torch.load(weight_path, map_location="cpu")
    state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
    missing, unexpected = unet.load_state_dict(state_dict, strict=False)
    print(f"[FaithDiff] Loaded weights with {len(missing)} missing and {len(unexpected)} unexpected keys.")
    return missing, unexpected

load_additional_layers(pipe.unet, faithdiff_path, dtype)

# Enable VAE tiling for large images
pipe.set_encoder_tile_settings()
pipe.enable_vae_tiling()

# Optimize memory usage
pipe.enable_model_cpu_offload()

# Set scheduler
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# Define input parameters
prompt = "A smiling woman in her 50s, blonde hair, white shirt, natural lighting"
upscale_factor = 2  # e.g., 2x upscale
start_point = "lr"  # 'lr' for low-resolution input
latent_tiled_overlap = 0.5
latent_tiled_size = 1024

# Load and prepare the input image
image_url = "https://huggingface.co/datasets/DEVAIEXP/assets/resolve/main/woman.png"
lq_image = load_image(image_url)
original_width, original_height = lq_image.width, lq_image.height
new_width, new_height = original_width * upscale_factor, original_height * upscale_factor

# Resize the image using LANCZOS filter
resized_image = lq_image.resize((new_width, new_height), Image.LANCZOS)

# Check and adjust image size for the pipeline
input_image, width_init, height_init, width_now, height_now = pipe.check_image_size(resized_image)

# Generate a random seed for reproducibility
generator = torch.Generator(device=device).manual_seed(random.randint(0, MAX_SEED))

# Perform super-resolution
output = pipe(
    lr_img=input_image,
    prompt=prompt,
    num_inference_steps=20,
    guidance_scale=5,
    generator=generator,
    start_point=start_point,
    height=height_now,
    width=width_now,
    overlap=latent_tiled_overlap,
    target_size=(latent_tiled_size, latent_tiled_size),
)

# Crop the output image to the original dimensions and save
output_image = output.images[0].crop((0, 0, width_init, height_init))
output_image.save("faithdiff_result.png")

# %%


