# ----------------------------------------------------------
# some snippets for the evaluation durring training
#
# author: Sebastian Eisner
# last change: Jan 18 2026, SE
# ----------------------------------------------------------

import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# 1) Loss Curves with Moving Variance
#       Evaluate numerical stability of training
#       Show trends of generator and discriminator losses
#       Moving variance highlights oscillations over time
#
#       Use: immediately after training with provided loss logs
def plot_loss_curves(g_losses, d_losses, window=10):
    def moving_variance(loss_list):
        return [np.var(loss_list[i:i+window]) for i in range(len(loss_list)-window)]
    
    g_variance = moving_variance(g_losses)
    d_variance = moving_variance(d_losses)
    
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.title("GAN Loss Curves")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(g_variance, label="Generator Variance")
    plt.plot(d_variance, label="Discriminator Variance")
    plt.title(f"Moving Variance (window={window})")
    plt.xlabel("Iteration")
    plt.ylabel("Variance")
    plt.legend()
    plt.show()
    
# Example usage:
# plot_loss_curves(g_losses, d_losses)

# 2) Fixed-Noise Image Grid
#       Generate images from a fixed set of noise vectors
#       Evaluate visual stability and diversity
#       Controlled & compareable --> see what the generator learned for specific inputs
# 
#       use: after training, important: use same noise vector for every stage of model!
def plot_fixed_noise_grid(generator, noise_v_dim, device, num_images=16, fixed_noise=None):
    if fixed_noise is None:
        fixed_noise = torch.randn(num_images, noise_v_dim, 1, 1, device=device)
        # standard normal distribution 
        # vector shape: (num_images, noise_v_dim, 1, 1) --> 4D tensor
        #   num_images: how many images to generate at once
        #   noise_v_dim: size of each noise vector (e.g. 100)
        #   1,1: spatial dimensions expected by the generator
        #   device=device: ensures the tensor is on the same device (CPU/GPU) as the model
        # reuse makes it "fixed" noise ...
        
        # torch.save(fixed_noise, "fixed_noise.pt")  # optional saving for reuse after restart
    
    with torch.no_grad():       # don't compute gradients --> save memory
        # replace by our actual model usage
        fake_images = None
        
    grid = vutils.make_grid(fake_images, nrow=4, normalize=True)
    plt.figure(figsize=(6,6))
    plt.imshow(grid.permute(1, 2, 0))       # permute reorders dimensions for correct display
    plt.title("Generated Images from Fixed Noise")
    plt.axis('off')
    plt.show()
    
    return fixed_noise  # return for reuse later

# Example usage:
# noise_vector = torch.load("fixed_noise.pt")   # optional, load locally saved noise vector
# plot_fixed_noise_grid(generator, noise_v_dim=100, device=device, fixed_noise=noise_vector)

# 3) Real vs Generated Side-by-Side Grid
#       Qualitative evaluation of how real the images look
#
#       use: after training, visual inspection
def plot_real_vs_generated(generator, real_dataset, noise_v_dim, device, batch_size=16):
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True)
    real_images, _ = next(iter(real_loader))
    
    noise = torch.randn(batch_size, noise_v_dim, 1, 1, device=device)
    with torch.no_grad():       # don't compute gradients --> save memory
        # replace by our actual model usage
        fake_images = None
    
    # Make grids
    real_grid = vutils.make_grid(real_images, nrow=4, normalize=True)
    fake_grid = vutils.make_grid(fake_images, nrow=4, normalize=True)
    
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(real_grid.permute(1, 2, 0))  # permute reorders dimensions for correct display
    plt.title("Real Images")
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(fake_grid.permute(1, 2, 0))  # permute reorders dimensions for correct display
    plt.title("Generated Images")
    plt.axis('off')
    
    plt.show()
    
# Example usage:
# plot_real_vs_generated(generator, real_dataset, noise_v_dim=100, device=device)
