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

# =========================================================
# Vanilla GAN Generator (MLP)
# =========================================================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(hidden_dim * 4, image_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)

# =========================================================
# Vanilla GAN Discriminator (MLP)
# =========================================================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(image_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1)
def smooth(x, window=50):
    return np.convolve(x, np.ones(window)/window, mode='valid')

# Reproducibility
manualSeed = 999
print("Random Seed:", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Hyperparameters
batch_size = 128
image_size = 32
nc = 3
image_dim = nc * image_size * image_size  # 3072
noise_dim = 100
hidden_dim = 512
num_epochs = 200
lr = 0.0001
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
netG = Generator().to(device)
netD = Discriminator().to(device)
print(netG)
print(netD)


# Loss & Optimizers
criterion = nn.BCELoss()

optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(64, noise_dim, device=device)

real_label = 1.0
fake_label = 0.0

# Training
img_list = []

print("Starting Training Loop...")
G_losses = []
D_losses = []

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader):


        # Discriminator
        netD.zero_grad()

        real = data[0].to(device)
        real = real.view(real.size(0), -1)
        b_size = real.size(0)

        label = torch.full((b_size,), real_label, device=device)
        output = netD(real)
        errD_real = criterion(output, label)
        errD_real.backward()
        #Fake images
        noise = torch.randn(b_size, noise_dim, device=device)
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
        # Save loss for plotting
        G_losses.append(errG.item())
        D_losses.append((errD_real + errD_fake).item())
    # Save samples per epoch
    with torch.no_grad():
        fake = netG(fixed_noise).cpu()
        fake = fake.view(-1, nc, image_size, image_size)
    img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

    print(f"[{epoch}/{num_epochs}] Loss_D: {(errD_real+errD_fake).item():.4f} Loss_G: {errG.item():.4f}")

# Plots
plt.figure(figsize=(10, 5))
plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.title("Vanilla GAN Training Losses")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Smoothed Plots
plt.figure(figsize=(10,5))
plt.plot(smooth(G_losses), label="Generator Loss (smoothed)")
plt.plot(smooth(D_losses), label="Discriminator Loss (smoothed)")
plt.legend()
plt.show()

# Images
real_batch = next(iter(dataloader))[0][:64].to(device)
with torch.no_grad():
    noise = torch.randn(64, noise_dim, device=device)
    fake_batch = netG(noise)

if fake_batch.dim() == 2:
    fake_batch = fake_batch.view(-1, nc, image_size, image_size)


plt.figure(figsize=(18, 9))

# Real Images
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images", fontsize=16)
plt.imshow(np.transpose(
    vutils.make_grid(real_batch, padding=2, normalize=True).cpu(),
    (1, 2, 0)
))

# Fake Images
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images (Generated)", fontsize=16)
plt.imshow(np.transpose(
    vutils.make_grid(fake_batch, padding=2, normalize=True).cpu(),
    (1, 2, 0)
))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

