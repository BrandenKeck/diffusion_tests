# Imported Libraries
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.optim.lr_scheduler import LambdaLR
from torchvision.transforms import ToTensor, Compose, CenterCrop, Normalize


# Imported Model(s)
import uvit.sde as sde
from uvit.uvit import UViT
from uvit.dpm_solver_pp import NoiseScheduleVP, DPM_Solver


# Constants
LR = 5e-6
DECAY = 0.995
EPOCHS = 5000
BATCHSIZE = 1
EVALSTEPS = 25
EVALOUTPUTS = 64
NORMMEAN = 0.5
NORMSTDEV = 0.2
DEVICE = "cuda"


# A function to prepare an inference diffusion schedule
def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


# Model runner to be passed to DPM Solver
def model_fn(x, t_continuous):
    t = t_continuous * len(_betas)
    _cond = nnet(x, t)
    return _cond


# Prepare data and models
transform = Compose([
    ToTensor(),
    CenterCrop([128, 128])#,
    # Normalize(3*[NORMMEAN], 3*[NORMSTDEV])
])
dataset = ImageFolder("./data/pokesolo", transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
model = UViT(img_size=128)
model.to(DEVICE)
# model.load_state_dict(torch.load(f'./uvit_pokemod.pth', weights_only=True))
model.train()
score_model = sde.ScoreModel(model, pred='noise_pred', sde=sde.VPSDE())
optimizer = Adam(model.parameters(), lr=LR)
lr_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: DECAY ** epoch)


# Train Model
best_loss = 9999
for epoch in range(EPOCHS):
    avg_loss = 0.
    num_items = 0
    for x, _ in dataloader:
        x = x.to(DEVICE)
        optimizer.zero_grad()
        loss = sde.LSimple(score_model, x, pred='noise_pred').mean()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    lr_scheduler.step()
    lr = lr_scheduler.get_last_lr()[0]
    if avg_loss / num_items < best_loss:
        torch.save(model.state_dict(), f'./uvit_pokemod.pth')
        best_loss = avg_loss / num_items
    print(f"Epoch: {epoch+1} [LR: {lr} | Loss: {avg_loss / num_items}]")


# Load Saved Model
nnet = UViT(img_size=128)
nnet.to(DEVICE)
model.load_state_dict(torch.load(f'./uvit_.pth', weights_only=True))
nnet.eval()


# Perform Evaluation
_betas = stable_diffusion_beta_schedule()  # set the noise schedule
noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=DEVICE).float())
z_init = torch.randn(EVALOUTPUTS, 3, 128, 128, device=DEVICE)
dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
with torch.no_grad():
  with torch.cuda.amp.autocast():  # inference with mixed precision
    samples = dpm_solver.sample(z_init, steps=EVALSTEPS, eps=1. / len(_betas), T=1.)
# denormalize = Compose([
#     Normalize(mean = 3*[0.], std = 3*[1/NORMSTDEV]),
#     Normalize(mean = 3*[-NORMMEAN], std = 3*[1.])
# ])
# samples = denormalize(samples)
save_image(samples, "uvit_output.png")