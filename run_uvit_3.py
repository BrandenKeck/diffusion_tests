# Import
import torch
import functools
import torch.multiprocessing
from torch.optim import Adam
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import ToTensor, Compose, CenterCrop, Normalize
torch.multiprocessing.set_sharing_strategy('file_system')

# Extra
SIGMA = 25.0
DEVICE = "cuda"
from uvit.uvit import UViT
from uvit.diffusionutil import *
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=SIGMA, device=DEVICE)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=SIGMA, device=DEVICE)


def train_score_model(score_model, dataset, lr, n_epochs, batch_size,
                      marginal_prob_std_fn=marginal_prob_std_fn,
                      lr_scheduler_fn=lambda epoch: 0.99 ** epoch,
                      device="cuda"):
  
  # Setup Training
  optimizer = Adam(score_model.parameters(), lr=lr)
  scheduler = LambdaLR(optimizer, lr_lambda=lr_scheduler_fn)
  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
  for epoch in range(n_epochs):
    score_model.train()
    avg_loss = 0.
    num_items = 0
    for x, _ in data_loader:
      x = x.to(device)
      loss = loss_fn_cond(score_model, x, marginal_prob_std_fn)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      avg_loss += loss.item() * x.shape[0]
      num_items += x.shape[0]
    scheduler.step()
    lr_current = scheduler.get_last_lr()[0]
    print('{} Average Loss: {:5f} lr {:.1e}'.format(epoch, avg_loss / num_items, lr_current))
    # Update the checkpoint after each epoch of training.


# Get Dataset
transform = Compose([
    ToTensor(),
    # CenterCrop([30, 30])#,
    # Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
dataset = ImageFolder("./data/shapes", transform=transform)

# Build Model
# model = torch.nn.DataParallel(
#   UNet(marginal_prob_std=marginal_prob_std_fn))
model = UViT(img_size=30, patch_size=6, embed_dim=1024, depth=8, num_heads=8)
# model.load_state_dict(torch.load(f'uvit_shapesmod.pth', weights_only=True))
model = model.to(DEVICE)

# Training Step 0
LR = 4.12e-4
EPOCHS = 20
BATCHSIZE = 32
DEVICE = 'cuda'
train_score_model(model, dataset, LR, EPOCHS, BATCHSIZE, device=DEVICE)
torch.save(model.state_dict(), f'uvit_shapesmod.pth')

# Training Step 1
LR = 1e-4
EPOCHS = 20
BATCHSIZE = 16
DEVICE = 'cuda'
train_score_model(model, dataset, LR, EPOCHS, BATCHSIZE, device=DEVICE)
torch.save(model.state_dict(), f'uvit_shapesmod.pth')

# Training Step 2
LR = 1e-5
EPOCHS = 100
BATCHSIZE = 8
DEVICE = 'cuda'
train_score_model(model, dataset, LR, EPOCHS, BATCHSIZE, device=DEVICE)
torch.save(model.state_dict(), f'uvit_shapesmod.pth')

# Training Step 2
LR = 5e-6
EPOCHS = 100
BATCHSIZE = 4
DEVICE = 'cuda'
train_score_model(model, dataset, LR, EPOCHS, BATCHSIZE, device=DEVICE)
torch.save(model.state_dict(), f'uvit_shapesmod.pth')

# Training Step 3
LR = 1e-6
EPOCHS = 100
BATCHSIZE = 2
DEVICE = 'cuda'
train_score_model(model, dataset, LR, EPOCHS, BATCHSIZE, device=DEVICE)
torch.save(model.state_dict(), f'uvit_shapesmod.pth')

# Run Sample
model = UViT(img_size=30, patch_size=6, embed_dim=1024, depth=8, num_heads=8)
model.load_state_dict(torch.load(f'uvit_shapesmod.pth', weights_only=True))
model.to(DEVICE)
sample_batch_size = 64
num_steps = 300
sampler = Euler_Maruyama_sampler
samples = sampler(model,
        marginal_prob_std_fn,
        diffusion_coeff_fn,
        sample_batch_size,
        x_shape=(3, 30, 30),
        num_steps=num_steps,
        eps=1e-4,
        device=DEVICE)
# denormalize = Normalize([-0.485/0.229, -0.456/0.224, -0.40