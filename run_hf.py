# Imports
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from diffusers.optimization import get_scheduler
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, CenterCrop

# Parameters
BATCH_SIZE = 16
EPOCHS = 1000
CHECKPOINT = 100
INF_BATCH_SIZE = 64
NOISE_TIMESTEPS = 1000
ADAM_LR = 2E-4
ADAM_B1 = 0.95
ADAM_B2 = 0.99
ADAM_DECAY = 1E-6
ADAM_EPS = 1E-8
LR_WARMUP = 500
LR_STEPS_SCALE = 100
MOD_BLOCK_LAYERS = 2
MOD_OUTCH1 = 128
MOD_OUTCH2 = 256
MOD_OUTCH3 = 512

# Load Data
transform = Compose([
    ToTensor(),
    CenterCrop([128, 128])
])
dataset = ImageFolder("./data/pokemini", transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Init Model
model = UNet2DModel(
    sample_size=128,
    in_channels=3,
    out_channels=3,
    layers_per_block=MOD_BLOCK_LAYERS,
    block_out_channels=(
    MOD_OUTCH1, MOD_OUTCH1, 
    MOD_OUTCH2, MOD_OUTCH2, 
    MOD_OUTCH3, MOD_OUTCH3
    ),
    down_block_types=(
    "DownBlock2D", "DownBlock2D", "DownBlock2D", "DownBlock2D",
    "AttnDownBlock2D",  "DownBlock2D",
    ),
    up_block_types=(
    "UpBlock2D", "AttnUpBlock2D",
    "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D",
    ),
)


# Create Accelerator
accelerator = Accelerator(
    gradient_accumulation_steps=1,
    mixed_precision="no"
)

# Create DDPM Scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=NOISE_TIMESTEPS
)

# Create Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(), lr=ADAM_LR,
    betas=(ADAM_B1, ADAM_B2),
    weight_decay=ADAM_DECAY,
    eps=ADAM_EPS
)

# Create Learning Reate Scheduler
lr_scheduler = get_scheduler(
    "cosine", 
    optimizer=optimizer,
    num_warmup_steps=LR_WARMUP,
    num_training_steps=(
        len(dataloader) * LR_STEPS_SCALE
    ),
)

# Prepare objects with accelerator
model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, dataloader, lr_scheduler
)

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    for step, batch in enumerate(dataloader):
        clean_images = batch[0].to("cuda")
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (clean_images.shape[0],), device=clean_images.device
        ).long()
        
        # Add noise to the clean images according to the noise magnitude
        # at each timestep (this is the forward diffusion process)
        noisy_images = noise_scheduler.add_noise(
            clean_images, noise, timesteps
        )

        with accelerator.accumulate(model):
            # Predict the noise residual
            noise_pred = model(noisy_images, timestep=timesteps).sample
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
    print(f"Epoch {epoch+1} / {EPOCHS} - Loss: {loss.detach().item()}")

    # Using  Will stop the execution of the current process
    # until every other process has reached that point.
    # This does nothing when running on a single process.
    accelerator.wait_for_everyone()

    
    # Generate sample images for visual inspection
    if accelerator.is_main_process:
        if epoch % CHECKPOINT == 0 or epoch == EPOCHS - 1:
            pipeline = DDPMPipeline(
                unet=accelerator.unwrap_model(model),
                scheduler=noise_scheduler,
            )

            generator = torch.manual_seed(0)
            # run pipeline in inference (sample random noise and denoise)
            images = pipeline(
                generator=generator,
                batch_size=INF_BATCH_SIZE,
                output_type="numpy"
            ).images

            # denormalize the images and save to wandb
            processed_images = torch.tensor(images).transpose(1, 3).transpose(2, 3)
            save_image(processed_images, f"./hf_epoch-{epoch+1}.png")
            pipeline.save_pretrained("hf_poke.pth")

    accelerator.wait_for_everyone()

accelerator.end_training()