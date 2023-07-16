# !pip install diffusers
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline
import numpy as np
model_id = "google/ddpm-cifar10-32"

# load model and scheduler
ddpm = DDIMPipeline.from_pretrained(model_id)  # you can replace DDPMPipeline with DDIMPipeline or PNDMPipeline for faster inference
ddpm.to("cuda")
# run pipeline in inference (sample random noise and denoise)
for u in range(50000):
    image = ddpm(batch_size=1, num_inference_steps=100, output_type='np', return_dict=False)
    np.save(f'./cifar_diffusion/image{u}', image)