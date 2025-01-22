import os 
import torch 
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.modeling_outputs import Transformer2dModelOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from tqdm import tqdm 

class FlowEuler:
    def __init__(self, model_fn, condition, uncondition, cfg_scale, model_kwargs):
        self.model = model_fn
        self.condition = condition 
        self.uncondition = uncondition 
        self.cfg_scale = cfg_scale
        self.model_kwargs = model_kwargs
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.0)
    def sample(self, latents, steps=28):
        device = self.condition.device
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, steps, device, None)
        do_classifier_free_guidance = True
        prompt_embeds = self.condition
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([self.uncondition, self.condition], dim=0)

        for i, t in tqdm(list(enumerate(timesteps)), disable-os.getenv("DPM_TQDM", "False") == "True"):
            latent_model_input = torch.cat([latent]*2) if do_classifier_free_guidance else latents
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.model(
                latent_model_input
                timestep,
                prompt_embeds,
                **self.model_kwargs
            )
            if isinstance(noise_pred, Transformer2DModelOutput):
                noise_pred = noise_pred[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text-noise_pred_uncond)

            latents_dtype = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            latents = latents.to(latents_dtype)

        return latents
