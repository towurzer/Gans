import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import os
from datetime import datetime
import time

        
# =========================================================
# DCGAN Generator
# =========================================================
class Generator(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

# =========================================================
# DCGAN Discriminator
# =========================================================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)

# =========================================================
# Weight initialization
# =========================================================
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
def smooth(x, window=50):
    return np.convolve(x, np.ones(window)/window, mode='valid')

def moving_variance(loss_list, window=10):
    return [np.var(loss_list[i:i+window]) for i in range(len(loss_list)-window)]

if __name__ == "__main__":
    # Reproducibility
    manualSeed = 999
    print("Random Seed:", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Hyperparameters
    batch_size = 64
    image_size = 32
    nc = 3
    noise_dim = 100
    num_epochs = 1     # 200
    #lr = 0.0002
    lrD = 0.00015
    lrG = 0.0002

    beta1 = 0.5
    workers = 2         

    # Dataset transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5))
    ])

    # CIFAR-10 labels: 0=airplane, 1=car, 2=bird, 3=cat, 4=deer, 5=dog, 6=frog, 7=horse, 8=ship, 9=truck
    target_class = 1  # get cars

    dataset = CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    indices = [i for i, t in enumerate(dataset.targets) if t == target_class] #indices for target class
    dataset = Subset(dataset, indices) #Create subset of data set

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        drop_last=True
    )

    # Build models
    netG = Generator(noise_dim).to(device)
    netD = Discriminator().to(device)

    netG.apply(weights_init)
    netD.apply(weights_init)

    print(netG)
    print(netD)

    # Loss & Optimizers
    criterion = nn.BCELoss()

    optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, noise_dim, 1, 1, device=device)

    real_label = 0.9
    fake_label = 0.0

    # Training Loop
    img_list = []
    G_losses = []
    D_losses = []

    print("Starting Training Loop...")
    startTime = time.monotonic()

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):


            # Discriminator
            netD.zero_grad()

            real = data[0].to(device)
            b_size = real.size(0)
            label = torch.full((b_size,), real_label, device=device)

            output = netD(real)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(b_size, noise_dim, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)

            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()

            optimizerD.step()

            errD = errD_real + errD_fake


            # Generator
            netG.zero_grad()
            label.fill_(real_label)

            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            if i % 50 == 0:
                print(f"[{epoch}/{num_epochs}] "
                    f"[{i}/{len(dataloader)}] "
                    f"Loss_D: {errD.item():.4f} "
                    f"Loss_G: {errG.item():.4f}")

            G_losses.append(errG.item())
            D_losses.append(errD.item())

        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    # ------------------------------------------------
    # Plots
    # ------------------------------------------------
    
    # for automated saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")        # Format: YYYYMMDD_HHMMSS (e.g. 20260119_134500)
    script_dir = os.path.dirname(os.path.abspath(__file__))     # directory of this script
    parent_dir = os.path.dirname(script_dir)                    # one level up

    logs_dir = os.path.join(parent_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)                       # ensure logs/ exists

    output_folder = os.path.join(logs_dir, f"{timestamp}_DCGAN_output")
    os.makedirs(output_folder, exist_ok=True)
    
    plt.figure()
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.title("GAN Loss Curves")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"loss_curves_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.show()
        
    G_variance = moving_variance(G_losses)
    D_variance = moving_variance(D_losses)
    plt.plot(G_variance, label="Generator Variance")
    plt.plot(D_variance, label="Discriminator Variance")
    plt.title(f"Moving Variance over Losses")
    plt.xlabel("Iteration")
    plt.ylabel("Variance")
    plt.legend()   
    plt.savefig(os.path.join(output_folder, f"loss_variance_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.show()     

    plt.plot(smooth(G_losses), label="Generator Loss (smoothed)")
    plt.plot(smooth(D_losses), label="Discriminator Loss (smoothed)")
    plt.title("GAN Smoothed Loss Curves")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"smoothed_loss_curves_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # ------------------------------------------------
    # Images
    # ------------------------------------------------
    plt.figure(figsize=(15,6))

    plt.subplot(1,2,1)
    plt.title("Real Images")
    plt.axis("off")
    plt.imshow(np.transpose(
        vutils.make_grid(data[0][:64], padding=5, normalize=True),
        (1,2,0)
    ))

    plt.subplot(1,2,2)
    plt.title("Fake Images")
    plt.axis("off")
    plt.imshow(np.transpose(img_list[-1], (1,2,0)))     # last generated batch

    plt.savefig(os.path.join(output_folder, f"real_vs_fake_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    # ------------------------------------------------
    # Save training parameters in a text file
    # ------------------------------------------------
    param_file = os.path.join(output_folder, f"training_parameters_{timestamp}.txt")
    with open(param_file, "w") as f:
        f.write("DCGAN Training Parameters\n")
        f.write("=========================\n\n")
        f.write(f"Date & Time: {timestamp}\n")
        f.write(f"Target Class (CIFAR-10): {target_class}\n")
        f.write(f"Number of Epochs: {num_epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate (Generator): {lrG}\n")
        f.write(f"Learning Rate (Discriminator): {lrD}\n")
        f.write(f"Beta1: {beta1}\n")
        f.write(f"Noise Dimension: {noise_dim}\n")
        f.write(f"Image Size: {image_size}x{image_size}\n")
        f.write(f"Number of Channels: {nc}\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Loss Function: BCELoss\n")
        f.write(f"Total Training Time: {time.monotonic() - startTime:.2f} seconds\n\n")
        
        f.write(netG.__str__() + "\n\n")
        f.write(netD.__str__() + "\n")
