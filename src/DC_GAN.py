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
num_epochs = 200
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


# Plot
plt.figure(figsize=(10,5))
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.legend()
plt.title("Generator and Discriminator Loss")
plt.show()

# Smoothed
plt.figure(figsize=(10,5))
plt.plot(smooth(G_losses), label="Generator Loss (smoothed)")
plt.plot(smooth(D_losses), label="Discriminator Loss (smoothed)")
plt.legend()
plt.show()

# Images
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
plt.imshow(np.transpose(img_list[-1], (1,2,0)))

plt.show()
