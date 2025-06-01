import os
import torch
from torch import Tensor
from PIL import Image
from torchvision import transforms
import glob
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import load_file
from diffusers import AutoencoderKL

def load_image(path: str, resolution: int = 1024) -> Tensor:
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])
    return transform(img).unsqueeze(0)  # [1, 3, 1024, 1024]

def encode_and_save_latents(vae_pth: str, data_dir: str = "data", latent_dir: str = "latents", device: str = "cuda"):
    # AutoencoderKL configuration
    config = {
        "_class_name": "AutoencoderKL",
        "act_fn": "silu",
        "block_out_channels": [128, 256, 512, 512],
        "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
        "force_upcast": True,
        "in_channels": 3,
        "latent_channels": 16,  # Match checkpoint
        "latents_mean": None,
        "latents_std": None,
        "layers_per_block": 2,
        "mid_block_add_attention": True,
        "norm_num_groups": 32,
        "out_channels": 3,
        "sample_size": 1024,
        "scaling_factor": 0.3611,
        "shift_factor": 0.1159,
        "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
        "use_post_quant_conv": False,
        "use_quant_conv": False
    }

    # Initialize AutoencoderKL with config
    vae = AutoencoderKL.from_config(config).to(device)
    
    # Load state dict from .safetensors
    try:
        state_dict = load_file(vae_pth, device=device)
        vae.load_state_dict(state_dict, strict=True)
    except Exception as e:
        raise ValueError(f"Failed to load .safetensors file from {vae_pth}: {e}")
    vae.eval()

    # Create latents directory
    Path(latent_dir).mkdir(exist_ok=True)

    # Get image paths
    image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
    if not image_paths:
        raise ValueError(f"No images found in {data_dir}")
    print(f"Encoding {len(image_paths)} images")

    # Encode and save latents
    for path in tqdm(image_paths, desc="Encoding"):
        img_key = Path(path).stem
        latent_path = os.path.join(latent_dir, f"{img_key}.pt")
        
        if os.path.exists(latent_path):
            continue

        try:
            img = load_image(path, resolution=1024).to(device)
            with torch.no_grad():
                latent_dist = vae.encode(img).latent_dist
                z = latent_dist.mean * 0.3611 + 0.1159  # Apply scaling_factor and shift_factor
            torch.save(z.cpu(), latent_path)
        except:
            print(img_key,"skipped")

    print(f"Latents saved to {latent_dir}")

if __name__ == "__main__":
    encode_and_save_latents(vae_pth="./vae.safetensors")

