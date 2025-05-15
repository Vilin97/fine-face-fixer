#%%
from PIL import Image, ImageOps
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
# from types import MethodType
import numpy as np
import os
# from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import label

# %%
#for making the diffusion edit

def load_image(filename, image_dir, max_size=512):
    image_path = os.path.join(image_dir, filename)
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    aspect_ratio = image.width / image.height
    if image.width > image.height:
        new_width = max_size
        new_height = int(max_size / aspect_ratio)
    else:
        new_height = max_size
        new_width = int(max_size * aspect_ratio)

    image = image.resize((new_width, new_height), Image.LANCZOS)
    return image

def generate_edited_image(filename, prompt, image_dir, output_dir, resolution, seed=3, num_inference_steps=100, image_guidance_scale=1.5, guidance_scale=7):
    image = load_image(filename, image_dir, resolution)
    generator = torch.Generator("cuda").manual_seed(seed)

    output = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=num_inference_steps,
        generator=generator,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale
    )

    generated_image = output.images[0]

    generated_image.save(EDITED_IMAGE)

    return generated_image

#%%
model_id = "peter-sushko/RealEdit"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None, cache_dir = 'cache')
pipe.to('cuda')
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# %%
image = Image.open("vas_with_hat.png")

width, height = image.size

prompt = "fix his teeth"
display(image)
images = pipe(prompt, image=image, num_inference_steps=10, image_guidance_scale=1.8).images
images[0].save("vas_with_hat_fixed.png")
images[0]
# %%
width, height = image.size
left = width // 3
right = 2 * width // 3
bottom = height // 3
top = 2 * height // 3
cropped_image = image.crop((left, bottom, right, top))
display(cropped_image)
cropped_image.save("vas_cropped.png")

#%%
cropped_image = cropped_image.resize((512, 512), Image.LANCZOS)
cropped_image.save("vas_cropped_512.png")
display(cropped_image)

#%%
image = Image.open("vas_cropped_512.png")

prompt = "make his teeth look normal"
display(image)
images = pipe(prompt, image=image, num_inference_steps=50, image_guidance_scale=1.8).images
images[0]

#%%

def encode(pipe, image):
    # Load and preprocess the image
    image = Image.open("vas.jpg").convert("RGB")
    display(image)
    peter_pixels = np.asarray(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    peter_tensor = torch.tensor(peter_pixels).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, H, W)
    # Move to the same dtype and device as the VAE expects
    peter_tensor = peter_tensor.to(dtype=pipe.vae.dtype, device=pipe.device)
    # Encode
    encoded = pipe.vae.encode(peter_tensor)
    latent = encoded.latent_dist.sample()
    return latent

def decode(pipe, latent):
    decoded = pipe.vae.decode(latent)
    # 1. Get the image tensor (usually in decoded.sample)
    image_tensor = decoded.sample  # This is usually a tensor like (1, 3, H, W)
    # 2. Squeeze batch dimension and move to CPU
    image_tensor = image_tensor.squeeze(0).detach().cpu()
    # 3. Clamp values to [0, 1] to make it valid for display
    image_tensor = image_tensor.clamp(0, 1)
    # 4. Convert to numpy and then PIL image
    image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_np)
    return image_pil

encoded = encode(pipe, image)
decoded = decode(pipe, encoded)
display(decoded)

#%%
# Convert both images to numpy arrays
original_np = np.asarray(image).astype(np.float32)
decoded_np = np.asarray(decoded).astype(np.float32)

# Ensure both arrays have the same shape
if original_np.shape != decoded_np.shape:
    decoded_np = np.array(decoded.resize(image.size)).astype(np.float32)

# Subtract decoded from original
diff_np = original_np - decoded_np

# Normalize to [0, 255] for display
diff_np = np.clip((diff_np - diff_np.min()) / (np.ptp(diff_np) + 1e-8) * 255, 0, 255).astype(np.uint8)

# Convert back to PIL Image and display
diff_image = Image.fromarray(diff_np)
display(diff_image)

#%%
from diffusers import DiffusionPipeline
import torch

model_id = "jychen9811/FaithDiff"
pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe.to("cuda")  # or "cpu" if you don't have a GPU

# %%
