import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.utils as vutils


def set_seed(seed):
	random.seed(seed)
	torch.manual_seed(seed)
	np.random.seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def smooth(x, window=50):
	return np.convolve(x, np.ones(window) / window, mode='valid')


def moving_variance(loss_list, window=10):
	return [np.var(loss_list[i:i + window]) for i in range(len(loss_list) - window)]


def save_plots(output_folder, g_losses, d_losses, fid_scores, is_scores, epochs_list, timestamp):
	# Loss Curve
	plt.figure()
	plt.plot(g_losses, label="Generator Loss")
	plt.plot(d_losses, label="Discriminator Loss")
	plt.title("GAN Loss Curves")
	plt.xlabel("Iteration")
	plt.ylabel("Loss")
	plt.legend()
	plt.savefig(os.path.join(output_folder, f"loss_curves_{timestamp}.png"), dpi=300, bbox_inches='tight')
	plt.close()

	# Variance
	g_var = moving_variance(g_losses)
	d_var = moving_variance(d_losses)
	plt.figure()
	plt.plot(g_var, label="Generator Variance")
	plt.plot(d_var, label="Discriminator Variance")
	plt.title(f"Moving Variance over Losses")
	plt.xlabel("Iteration")
	plt.ylabel("Variance")
	plt.legend()
	plt.savefig(os.path.join(output_folder, f"loss_variance_{timestamp}.png"), dpi=300, bbox_inches='tight')
	plt.close()

	# FID and IS Scores
	if fid_scores and is_scores:
		fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
		ax1.plot(epochs_list, fid_scores, marker='o', label="FID Score", color='blue')
		ax1.set_xlabel("Epoch")
		ax1.set_ylabel("FID Score (Lower is better)")
		ax1.set_title("Frechet Inception Distance")
		ax1.legend()
		ax1.grid(True)

		ax2.plot(epochs_list, is_scores, marker='o', label="IS Score", color='green')
		ax2.set_xlabel("Epoch")
		ax2.set_ylabel("IS Score (Higher is better)")
		ax2.set_title("Inception Score")
		ax2.legend()
		ax2.grid(True)

		plt.tight_layout()
		plt.savefig(os.path.join(output_folder, f"fid_is_{timestamp}.png"), dpi=300)
		plt.close()


def save_image_grid(real_batch, fake_batch, output_folder, timestamp):
	plt.figure(figsize=(15, 6))
	plt.subplot(1, 2, 1)
	plt.title("Real Images")
	plt.axis("off")
	plt.imshow(np.transpose(
		vutils.make_grid(real_batch[:64], padding=2, normalize=True).cpu(),
		(1, 2, 0)
	))

	plt.subplot(1, 2, 2)
	plt.title("Fake Images")
	plt.axis("off")
	plt.imshow(np.transpose(vutils.make_grid(fake_batch[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

	plt.savefig(os.path.join(output_folder, f"real_vs_fake_{timestamp}.png"), dpi=300)
	plt.close()


def save_epoch_image(gen_batch, output_folder, epoch):
	filename = os.path.join(output_folder, f"epoch_{epoch:03d}.png")

	grid = vutils.make_grid(gen_batch, padding=2, normalize=True).cpu()

	plt.figure(figsize=(8, 8))
	plt.axis("off")
	plt.title(f"Epoch {epoch}")
	plt.imshow(grid.permute(1, 2, 0))

	plt.savefig(filename, dpi=150, bbox_inches='tight')
	plt.close()
