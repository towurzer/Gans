from dataclasses import dataclass
import torch


@dataclass
class Config:
	# Reproducibility
	manual_seed: int = 999

	# System
	device: str = "cuda" if torch.cuda.is_available() else "cpu"
	num_workers: int = 4 if torch.cuda.is_available() else 0

	# Hyperparameters
	batch_size: int = 128
	image_size: int = 32
	nc: int = 3  # Number of channels (RGB vs Grayscale)
	noise_dim: int = 100  # Noise dimension
	num_epochs: int = 1
	num_samples_eval: int = 500

	# Optimization
	lr_d: float = 0.00015  # Learning Rate Discriminator
	lr_g: float = 0.0002  # Learning Rate Generator
	beta1: float = 0.5  # Standard is .9 but Paper suggests .5 is better for stability

	# Data
	dataroot: str = "./data"
	# CIFAR-10 labels: 0=airplane, 1=car, 2=bird, 3=cat, 4=deer, 5=dog, 6=frog, 7=horse, 8=ship, 9=truck
	target_class: int = 1  # 1 = car

	# Paths & Logging
	save_dir: str = "../logs"
	model_save_path: str = "../config"
	metric_eval_freq: int = 25  # Evaluate FID/IS every N epochs
