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
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import csv

        
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
            #nn.Sigmoid()
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

    # Enable performance optimizations
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    # Hyperparameters
    batch_size = 128
    image_size = 32
    nc = 3
    noise_dim = 100
    num_epochs = 1     # 200
    #lr = 0.0002
    lrD = 0.00015
    lrG = 0.0002

    beta1 = 0.5
    workers = 0 if device.type == "cpu" else 4         

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
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=True if workers > 0 else False,
        prefetch_factor=4 if workers > 0 else None
    )

    # Build models
    netG = Generator(noise_dim).to(device)
    netD = Discriminator().to(device)

    # Convert to channels_last for better conv performance
    if device.type == "cuda":
        netG = netG.to(memory_format=torch.channels_last)
        netD = netD.to(memory_format=torch.channels_last)

    # Load saved weights if they exist
    try:
        netG.load_state_dict(torch.load('config/generator.pth', map_location=device))
        netD.load_state_dict(torch.load('config/discriminator.pth', map_location=device))
        print("Loaded pre-trained models")
    except FileNotFoundError:
        print("No saved models found, starting from scratch")
        netG.apply(weights_init)
        netD.apply(weights_init)

    # Compile models for faster execution 
    if device.type == "cuda":
        try:
            netG = torch.compile(netG)
            netD = torch.compile(netD)
            print("Models compiled successfully")
        except Exception as e:
            print(f"Model compilation not available: {e}")

    print(netG)
    print(netD)

    # Loss & Optimizers
    #criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    optimizerD = optim.Adam(netD.parameters(), lr=lrD, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lrG, betas=(beta1, 0.999))

    fixed_noise = torch.randn(64, noise_dim, 1, 1, device=device)

    real_label = 0.9
    fake_label = 0.0

    # Pre-allocate tensors for training loop
    noise = torch.randn(batch_size, noise_dim, 1, 1, device=device)
    label_real = torch.full((batch_size,), real_label, device=device, dtype=torch.float32)
    label_fake = torch.full((batch_size,), fake_label, device=device, dtype=torch.float32)

    # Initialize AMP scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    # Initialize FID and IS metrics
    fid_metric = FrechetInceptionDistance(normalize=True).to(device)
    is_metric = InceptionScore(normalize=True).to(device)

    metric_eval_freq = 25  # Evaluate FID/IS every N epochs

    # Training Loop
    img_list = []
    G_losses = []
    D_losses = []
    fid_scores = []
    is_scores = []
    epochs_list = []

    print("Starting Training Loop...")
    startTime = time.monotonic()

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            b_size = data[0].size(0)
            real = data[0].to(device, non_blocking=True)
            if device.type == "cuda":
                real = real.to(memory_format=torch.channels_last)

            # Update pre-allocated label tensors if batch size differs
            if b_size != batch_size:
                label_real_batch = label_real[:b_size]
                label_fake_batch = label_fake[:b_size]
            else:
                label_real_batch = label_real
                label_fake_batch = label_fake

            # ============================
            # Train Discriminator
            # ============================
            netD.zero_grad(set_to_none=True)

            if device.type == "cuda" and scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = netD(real)
                    errD_real = criterion(output, label_real_batch)
                scaler.scale(errD_real).backward()

                # Generate fake images
                noise.normal_()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    fake = netG(noise[:b_size])
                    output = netD(fake.detach())
                    errD_fake = criterion(output, label_fake_batch)
                scaler.scale(errD_fake).backward()

                scaler.step(optimizerD)
                scaler.update()
            else:
                output = netD(real)
                errD_real = criterion(output, label_real_batch)
                errD_real.backward()

                noise.normal_()
                fake = netG(noise[:b_size])
                output = netD(fake.detach())
                errD_fake = criterion(output, label_fake_batch)
                errD_fake.backward()

                optimizerD.step()

            errD = errD_real + errD_fake

            # ============================
            # Train Generator
            # ============================
            netG.zero_grad(set_to_none=True)

            if device.type == "cuda" and scaler is not None:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = netD(fake)
                    errG = criterion(output, label_real_batch)
                scaler.scale(errG).backward()
                scaler.step(optimizerG)
                scaler.update()
            else:
                output = netD(fake)
                errG = criterion(output, label_real_batch)
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

            if epoch % metric_eval_freq == 0:

                # Generate samples for FID/IS computation
                num_samples = 500
                batch_size_eval = 64
                fake_images = []
                real_images = []
                
                for batch_idx, batch_data in enumerate(dataloader):
                    if batch_idx * batch_size_eval >= num_samples:
                        break
                    real_batch = batch_data[0].to(device)
                    real_images.append(real_batch)
                
                real_images = torch.cat(real_images, dim=0)[:num_samples]
                
                for _ in range((num_samples + batch_size_eval - 1) // batch_size_eval):
                    z = torch.randn(batch_size_eval, noise_dim, 1, 1, device=device)
                    fake_batch = netG(z)
                    fake_images.append(fake_batch.detach())
                
                fake_images = torch.cat(fake_images, dim=0)[:num_samples]
                
                # Denormalize images for metric computation (scale from [-1,1] to [0,255])
                real_images_denorm = ((real_images + 1) / 2 * 255).byte()
                fake_images_denorm = ((fake_images + 1) / 2 * 255).byte()
                
                # Compute FID and IS with error handling
                try:
                    fid_metric.update(real_images_denorm, real=True)
                    fid_metric.update(fake_images_denorm, real=False)
                    fid_score = fid_metric.compute()
                    fid_scores.append(fid_score.item())
                    fid_metric.reset()
                    
                    is_metric.update(fake_images_denorm)
                    is_score = is_metric.compute()
                    is_scores.append(is_score[0].item())
                    is_metric.reset()
                    
                    epochs_list.append(epoch)
                    print(f"Epoch [{epoch}/{num_epochs}] - FID: {fid_score:.4f}, IS: {is_score[0]:.4f}")
                except Exception as e:
                    print(f"Warning: Could not compute metrics at epoch {epoch}: {e}")


    endTime = time.monotonic()
    print(f"Training completed in {endTime - startTime:.2f} seconds")

    # Save models
    torch.save(netG.state_dict(), 'config/generator.pth')
    torch.save(netD.state_dict(), 'config/discriminator.pth')

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
    plt.close()
        
    G_variance = moving_variance(G_losses)
    D_variance = moving_variance(D_losses)
    plt.plot(G_variance, label="Generator Variance")
    plt.plot(D_variance, label="Discriminator Variance")
    plt.title(f"Moving Variance over Losses")
    plt.xlabel("Iteration")
    plt.ylabel("Variance")
    plt.legend()   
    plt.savefig(os.path.join(output_folder, f"loss_variance_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.plot(smooth(G_losses), label="Generator Loss (smoothed)")
    plt.plot(smooth(D_losses), label="Discriminator Loss (smoothed)")
    plt.title("GAN Smoothed Loss Curves")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_folder, f"smoothed_loss_curves_{timestamp}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # FID and IS Scores
    if fid_scores and is_scores:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(epochs_list, fid_scores, marker='o', label="FID Score", color='blue')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("FID Score (lower is better)")
        ax1.set_title("Frechet Inception Distance")
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs_list, is_scores, marker='o', label="IS Score", color='green')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("IS Score (higher is better)")
        ax2.set_title("Inception Score")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"fid_is_{timestamp}.png"), dpi=300, bbox_inches='tight')
        plt.close()

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
    plt.close()
    
    print(f"All plots saved to: {output_folder}")
    
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
        f.write(f"Learning Rate Generator: {lrG}\n")
        f.write(f"Learning Rate Discriminator: {lrD}\n")
        f.write(f"Beta1: {beta1}\n")
        f.write(f"Noise Dimension: {noise_dim}\n")
        f.write(f"Image Size: {image_size}x{image_size}\n")
        f.write(f"Number of Channels: {nc}\n")
        f.write(f"Optimizer Discriminator: {optimizerD.__str__()}\n")
        f.write(f"Optimizer Generator: {optimizerG.__str__()}\n")
        f.write(f"Loss Function: {criterion.__str__()}\n")
        f.write(f"Total Training Time: {endTime - startTime:.2f} seconds\n\n")
        
        f.write(netG.__str__() + "\n\n")
        f.write(netD.__str__() + "\n")
        
    print(f"Training parameters saved to: {param_file}")
        
    # ------------------------------------------------
    # Export csv file for loss values and metrics
    # ------------------------------------------------
    with open(os.path.join(output_folder, "losses_and_metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Generator Loss", "Discriminator Loss", "Epoch", "FID Score", "IS Score"])

        for i in range(max(len(G_losses), len(epochs_list))):
            writer.writerow([
                i,
                G_losses[i],
                D_losses[i],
                epochs_list[i] if i < len(epochs_list) else None,
                fid_scores[i] if i < len(fid_scores) else None,
                is_scores[i] if i < len(is_scores) else None
            ])
            
    print(f"Losses and metrics exported to: {os.path.join(output_folder, 'losses_and_metrics.csv')}")
            