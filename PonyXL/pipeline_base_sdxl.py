from diffusers import StableDiffusionXLPipeline, AutoencoderKL, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler
from transformers import CLIPTextModel,CLIPTextModelWithProjection
import torch 


pipe = StableDiffusionXLPipeline.from_pretrained("Bakanayatsu/Pony-Diffusion-V6-XL-for-Anime", torch_dtype=torch.float16).to("cuda")

image = pipe(prompt="a mystic rat on a mountain", height=1024, width=1024, guidance_scale=7, num_inference_steps=5, ).images[0]

image.save("./test_inference_pony.png")

