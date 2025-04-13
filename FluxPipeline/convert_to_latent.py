import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Model parameters
params = {
    'resolution': 256,
    'in_channels': 3,
    'ch': 128,
    'ch_mult': [1, 2, 4, 8],
    'num_res_blocks': 2,
    'z_channels': 16,  # Matches your latent shape [batch, 16, 128, 128]
    'out_channels': 3,
    'scale_factor': 1.0,  # Adjust based on your model
    'shift_factor': 0.0
}

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = AutoEncoder(params).to(device)

# Initialize dataset
image_folder = "path/to/your/image/folder"  # Replace with your folder path
dataset = VAELatentDataset(
    image_folder=image_folder,
    autoencoder=autoencoder,
    crop_sizes=[256, 512],  # Crop sizes divisible by 16
    small_batch_size=4,
    device=device
)

# Initialize DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=1,  # We control batching in collate_fn
    shuffle=True,
    collate_fn=lambda batch: vae_collate_fn(batch, autoencoder, small_batch_size_range=(1, 4), device=device)
)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.approx_decoder.parameters(), lr=1e-3)

# Training loop
num_epochs = 10
batch_size = 128  # Desired batch size for latents
latent_buffer = []
target_buffer = []

for epoch in range(num_epochs):
    for latents, targets in dataloader:
        # latents: [small_batch_size, 16, 128, 128]
        # targets: [small_batch_size, 3, 256, 256]
        
        # Accumulate latents and targets
        latent_buffer.append(latents)
        target_buffer.append(targets)
        
        # When we have enough for a full batch (128)
        while len(latent_buffer) * latents.size(0) >= batch_size:
            # Concatenate latents and targets
            batch_latents = torch.cat(latent_buffer, dim=0)[:batch_size]
            batch_targets = torch.cat(target_buffer, dim=0)[:batch_size]
            
            # Remove used items from buffers
            total_samples = batch_size // latents.size(0)
            latent_buffer = latent_buffer[total_samples:]
            target_buffer = target_buffer[total_samples:]
            if batch_latents.size(0) > batch_size:
                # If we took too many, put leftovers back
                leftover_latents = batch_latents[batch_size:]
                leftover_targets = batch_targets[batch_size:]
                latent_buffer.insert(0, leftover_latents)
                target_buffer.insert(0, leftover_targets)
                batch_latents = batch_latents[:batch_size]
                batch_targets = batch_targets[:batch_size]
            
            # Train on the batch
            batch_latents = batch_latents.to(device)
            batch_targets = batch_targets.to(device)
            
            # Forward pass
            pred = autoencoder.approximate_decode(batch_latents)
            loss = criterion(pred, batch_targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    # Handle any remaining samples in the buffer at epoch end
    if latent_buffer:
        batch_latents = torch.cat(latent_buffer, dim=0)
        batch_targets = torch.cat(target_buffer, dim=0)
        if batch_latents.size(0) > 0:
            batch_latents = batch_latents.to(device)
            batch_targets = batch_targets.to(device)
            
            pred = autoencoder.approximate_decode(batch_latents)
            loss = criterion(pred, batch_targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Final Batch Loss: {loss.item():.4f}")
        
        latent_buffer = []
        target_buffer = []

# Save the trained approximate decoder
torch.save(autoencoder.approx_decoder.state_dict(), "approx_decoder.pth")
