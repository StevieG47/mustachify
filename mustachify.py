# -*- coding: utf-8 -*-
"""mustachify.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xEBRuGEmUEtiDsHLr2Jsle4n83OMg7Rp
"""

!pip install -qq -U diffusers==0.11.1 transformers ftfy gradio accelerate

import inspect
from typing import List, Optional, Union

import numpy as np
import torch

import PIL
import gradio as gr
from diffusers import StableDiffusionInpaintPipeline

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from diffusers import StableDiffusionInpaintPipeline

image = Image.open('image.png')
mask_image = Image.open('mask.png')

# Define the inpainting pipeline
device = "cuda"
model_path = "stabilityai/stable-diffusion-2-inpainting"
#"runwayml/stable-diffusion-inpainting"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
).to(device)

import matplotlib.pyplot as plt
import numpy as np

# Inpaint the masked region
prompt = "a high quality hyper-realistic photograph of a man with a mustache, thick mustache, dark mustache, facial hair"
n_prompt = "cartoon, blurry, ugly, bad anatomy, disfigured"

cfg = 8
denoise=.7
steps=30
num_samples = 1
generator = torch.Generator(device="cuda").manual_seed(np.random.randint(1e5)) # change the seed to get different results

images = pipe(
    height=512,
    width=512,
    prompt=prompt,
    negative_prompt=n_prompt,
    num_inference_steps=20,
    image=image,
    mask_image=mask_image,
    guidance_scale=cfg,
    generator=generator,
    num_images_per_prompt=num_samples,
).images

# insert initial image in the list so we can compare side by side
images.insert(0, image)
fig1=plt.figure(figsize=(10,5))
ax1,ax2=fig1.subplots(1,2)
ax1.imshow(images[0])
ax2.imshow(images[1])