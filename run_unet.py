# Libs
import math, functools
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.multiprocessing
from torch.optim import Adam
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import MultiplicativeLR, LambdaLR
from torchvision.transforms import ToTensor, Compose, CenterCrop, Normalize
torch.multiprocessing.set_sharing_strategy('file_system')

# Extra
SIGMA = 25.0
DEVICE = "cuda"
from unet.diffusionutil import *
from unet.diffusionmodel import UNet
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=SIGMA, device=DEVICE)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=SIGMA, device=DEVICE)


def train_score_model(score_model, dataset, lr, n_epochs, batch_size,
                      marginal_prob_std_fn=marginal_prob_std_fn,
                      lr_scheduler_fn=lambda epoch: 0.995 ** epoch,
                      device="cuda"):
  
  # Setup Training
  optimizer = Adam(score_model.parameters(), lr=lr)
  scheduler = LambdaLR(optimizer, lr_lambda=lr_scheduler_fn)
  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
  for epoch in range(n_epochs):
    score_model.train()
    avg_loss = 0.
    num_items = 0
    for x, y in data_loader:
      x = x.to(device)
      y = y.to(device)
      loss = loss_fn_cond(score_model, x, y, marginal_prob_std_fn)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      avg_loss += loss.item() * x.shape[0]
      num_items += x.shape[0]
    scheduler.step()
    lr_current = scheduler.get_last_lr()[0]
    print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
    # Update the checkpoint after each epoch of training.
    torch.save(score_model.state_dict(), f'unet_pokemod.pth')


# Get Dataset
transform = Compose([
    ToTensor(),
    CenterCrop([128, 128])#,
    # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset = ImageFolder("./data/pokemicro", transform=transform)

# Build Model
# model = torch.nn.DataParallel(
#   UNet(marginal_prob_std=marginal_prob_std_fn))
model = UNet(marginal_prob_std=marginal_prob_std_fn)
model.load_state_dict(torch.load(f'unet_pokemod.pth', weights_only=True))
model = model.to(DEVICE)

# Training Step 1
LR = 1e-3
EPOCHS = 300
BATCHSIZE = 32
DEVICE = 'cuda'
train_score_model(model, dataset, LR, EPOCHS, BATCHSIZE, device=DEVICE)
train_score_model(model, dataset, LR, EPOCHS, BATCHSIZE, device=DEVICE) # Do it twice

# Training Step 2
LR = 5e-4
EPOCHS = 100
BATCHSIZE = 1
train_score_model(model, dataset, LR, EPOCHS, BATCHSIZE, device=DEVICE)

# Training Step 3
LR = 1e-4
EPOCHS = 100
BATCHSIZE = 1
train_score_model(model, dataset, LR, EPOCHS, BATCHSIZE, device=DEVICE)

# Training Step 4
LR = 1e-6
EPOCHS = 200
BATCHSIZE = 1
train_score_model(model, dataset, LR, EPOCHS, BATCHSIZE, device=DEVICE,
                  lr_scheduler_fn=lambda epoch: 1)

# Training Step 5
LR = 1e-7
EPOCHS = 200
BATCHSIZE = 1
train_score_model(model, dataset, LR, EPOCHS, BATCHSIZE, device=DEVICE,
                  lr_scheduler_fn=lambda epoch: 1)

# Run Sample
model = UNet(marginal_prob_std=marginal_prob_std_fn)
model.load_state_dict(torch.load(f'unet_pokemod.pth', weights_only=True))
model.to(DEVICE)
sample_batch_size = 64
num_steps = 250
sampler = Euler_Maruyama_sampler
samples = sampler(model,
        marginal_prob_std_fn,
        diffusion_coeff_fn,
        sample_batch_size,
        x_shape=(3, 128, 128),
        num_steps=num_steps,
        device=DEVICE,
        y=torch.zeros([64], dtype=torch.int32).to(DEVICE))
# denormalize = Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225],
#                         [1/0.229, 1/0.224, 1/0.225])
# samples = denormalize(samples).clamp(0.0, 1.0)
save_image(samples, "./unet_output.png")