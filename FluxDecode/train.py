import os
import torch
from torch import nn, Tensor
from PIL import Image
import glob
from tqdm import tqdm
from torchvision import transforms
from pathlib import Path
import wandb
import torchvision.utils as vutils
import random
from torch.utils.data import Dataset, DataLoader

class SimpleDecoder(nn.Module):
    def __init__(self, z_channels: int = 16, out_channels: int = 3, resolution: int = 1024, latent_size: int = 128):
        super().__init__()
        self.z_channels = z_channels
        self.out_channels = out_channels
        self.target_resolution = resolution
        self.latent_size = latent_size

        self.conv1 = nn.Conv2d(
            in_channels=z_channels,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            padding=1
        )
        self.conv_out = nn.Conv2d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )
        self.upsample = nn.Upsample(size=(resolution, resolution), mode='bilinear', align_corners=False)

    def forward(self, z: Tensor) -> Tensor:
        x = self.conv1(z)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.upsample(x)
        x = self.conv_out(x)
        return x

def load_image(path: str, resolution: int = 1024) -> Tensor:
    img = Image.open(path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(img)

def denormalize(tensor: Tensor) -> Tensor:
    return (tensor * 0.5 + 0.5).clamp(0, 1)

class LatentDataset(Dataset):
    def __init__(self, pairs, resolution=1024):
        self.pairs = pairs
        self.resolution = resolution

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, latent_path = self.pairs[idx]
        img = load_image(img_path, self.resolution)  # [3, 1024, 1024]
        z = torch.load(latent_path, weights_only=True)  # [1, 16, 128, 128]
        z = z.squeeze(0)  # [16, 128, 128]
        if z.shape != (16, 128, 128):
            raise ValueError(f"Unexpected latent shape: {z.shape}, expected [16, 128, 128]")
        return img, z, img_path

def train(
    data_dir: str = "data",
    latent_dir: str = "latents",
    steps: int = 50000,
    device: str = "cuda",
    log_samples_every_n_epochs: int = 5,
    batch_size: int = 8,
    num_workers: int = 4
):
    # Config parameters
    resolution = 1024
    z_channels = 16
    out_channels = 3
    scaling_factor = 0.3611
    shift_factor = 0.1159
    latent_size = 128

    # Initialize SimpleDecoder
    decoder = SimpleDecoder(
        z_channels=z_channels,
        out_channels=out_channels,
        resolution=resolution,
        latent_size=latent_size
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(decoder.parameters(), lr=5e-6)

    # Dataset: start with latents
    latent_paths = glob.glob(os.path.join(latent_dir, "*.pt"))
    if not latent_paths:
        raise ValueError(f"No latents found in {latent_dir}")

    pairs = []
    for latent_path in latent_paths:
        img_key = Path(latent_path).stem
        img_path = os.path.join(data_dir, f"{img_key}.jpg")
        if os.path.exists(img_path):
            pairs.append((img_path, latent_path))
    if not pairs:
        raise ValueError(f"No matching image-latent pairs found")
    print(f"Total image-latent pairs: {len(pairs)}")

    # Split into 90% train, 10% eval
    random.seed(42)
    random.shuffle(pairs)
    train_size = int(0.9 * len(pairs))
    train_pairs = pairs[:train_size]
    eval_pairs = pairs[train_size:]
    print(f"Training on {len(train_pairs)} pairs, evaluating on {len(eval_pairs)} pairs")

    # DataLoaders
    train_dataset = LatentDataset(train_pairs, resolution=resolution)
    eval_dataset = LatentDataset(eval_pairs, resolution=resolution)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Calculate epochs
    steps_per_epoch = len(train_loader)
    total_epochs = (steps + steps_per_epoch - 1) // steps_per_epoch
    print(f"Steps per epoch: {steps_per_epoch}, Total epochs: {total_epochs}")

    # Initialize Weights & Biases
    wandb.init(project="flux_decoder_training_adam_3000", config={
        "steps": steps,
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "total_epochs": total_epochs,
        "learning_rate": 5e-6,
        "resolution": resolution,
        "z_channels": z_channels,
        "train_size": len(train_pairs),
        "eval_size": len(eval_pairs),
        "latent_size": latent_size
    })

    def train_step(imgs: Tensor, latents: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        imgs = imgs.to(device)
        latents = latents.to(device)
        latents = (latents - shift_factor) / scaling_factor
        decoder.train()
        recon = decoder(latents)
        loss = nn.functional.l1_loss(recon, imgs)
        return loss, recon, imgs

    def eval_step(imgs: Tensor, latents: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        imgs = imgs.to(device)
        latents = latents.to(device)
        latents = (latents - shift_factor) / scaling_factor
        decoder.eval()
        with torch.no_grad():
            recon = decoder(latents)
            loss = nn.functional.l1_loss(recon, imgs)
        return loss, recon, imgs

    # Select 5 samples for visualization
    train_sample_pairs = train_pairs[:5] if len(train_pairs) >= 5 else train_pairs
    eval_sample_pairs = eval_pairs[:5] if len(eval_pairs) >= 5 else eval_pairs

    step_count = 0
    for epoch in range(total_epochs):
        print(f"Epoch {epoch + 1}/{total_epochs}")
        epoch_train_loss = 0.0
        num_batches = 0
        for imgs, latents, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            if step_count >= steps:
                break
            optimizer.zero_grad()
            loss, recon, ground_truth = train_step(imgs, latents)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * imgs.size(0)
            num_batches += imgs.size(0)
            wandb.log({"step": step_count, "train_loss": loss.item()})
            step_count += 1

            if step_count % 100 == 0:
                print(f"Step {step_count}, Train Loss: {loss.item():.4f}")

        avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else 0.0

        # Evaluation
        eval_loss = 0.0
        num_eval = 0
        for imgs, latents, _ in eval_loader:
            loss, _, _ = eval_step(imgs, latents)
            eval_loss += loss.item() * imgs.size(0)
            num_eval += imgs.size(0)
        avg_eval_loss = eval_loss / num_eval if num_eval > 0 else 0.0
        print(f"Epoch {epoch + 1}, Eval Loss: {avg_eval_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "train_loss_avg": avg_train_loss, "eval_loss": avg_eval_loss})

        # Log sample images
        if (epoch + 1) % log_samples_every_n_epochs == 0 or step_count >= steps:
            decoder.eval()
            train_images = []
            for img_path, latent_path in train_sample_pairs:
                img = load_image(img_path, resolution=resolution).to(device)
                z = torch.load(latent_path, weights_only=True, map_location=device)
                z = z.squeeze(0)  # [16, 128, 128]
                z = (z - shift_factor) / scaling_factor
                with torch.no_grad():
                    recon = decoder(z.unsqueeze(0)).squeeze(0)
                img = denormalize(img)
                recon = denormalize(recon)
                train_images.extend([img, recon])
            train_grid = vutils.make_grid(train_images, nrow=2, padding=10)
            eval_images = []
            for img_path, latent_path in eval_sample_pairs:
                img = load_image(img_path, resolution=resolution).to(device)
                z = torch.load(latent_path, weights_only=True, map_location=device)
                z = z.squeeze(0)  # [16, 128, 128]
                z = (z - shift_factor) / scaling_factor
                with torch.no_grad():
                    recon = decoder(z.unsqueeze(0)).squeeze(0)
                img = denormalize(img)
                recon = denormalize(recon)
                eval_images.extend([img, recon])
            eval_grid = vutils.make_grid(eval_images, nrow=2, padding=10)
            wandb.log({
                "epoch": epoch + 1,
                "train_samples": wandb.Image(train_grid, caption=f"Epoch {epoch + 1}: Training Original vs Reconstructed"),
                "eval_samples": wandb.Image(eval_grid, caption=f"Epoch {epoch + 1}: Eval Original vs Reconstructed")
            })
            decoder.train()

        if step_count % 5000 == 0 and step_count > 0:
            torch.save(decoder.state_dict(), f"simple_decoder_step_{step_count}.pth")

        if step_count >= steps:
            break

    torch.save(decoder.state_dict(), "simple_decoder_final.pth")
    wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    train(data_dir="data", latent_dir="latents", batch_size=16)
