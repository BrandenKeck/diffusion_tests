# Import
import torch
import functools
import torch.multiprocessing
from torchvision.utils import save_image
torch.multiprocessing.set_sharing_strategy('file_system')

# Extra
SIGMA = 16.0
DEVICE = "cuda"
from uvit.uvit import UViT
from uvit.diffusionutil import *
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=SIGMA, device=DEVICE)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=SIGMA, device=DEVICE)

# Run Sample
model = UViT()
model.load_state_dict(torch.load(f'uvit_pokemod.pth', weights_only=True))
model.to(DEVICE)
sample_batch_size = 64
num_steps = 100
sampler = Euler_Maruyama_sampler
samples = sampler(model,
        marginal_prob_std_fn,
        diffusion_coeff_fn,
        sample_batch_size,
        x_shape=(3, 128, 128),
        num_steps=num_steps,
        eps=1e-2,
        device=DEVICE)
save_image(samples, "./uvit_output.png")