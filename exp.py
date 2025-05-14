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
image = Image.open("peter_bad_eyes.png")

width, height = image.size

prompt = "fix the eyes"
display(image)
images = pipe(prompt, image=image, num_inference_steps=50, image_guidance_scale=1.8).images
images[0].save("peter_better_eyes.png")
images[0]
# %%
