import torch
from config import Config
from dataset import get_dataloader
from model_trainer import DCGANTrainer
import utils

if __name__ == "__main__":
	# Load Configuration
	cfg = Config()

	# Setup Reproducibility
	utils.set_seed(cfg.manual_seed)

	# Enable performance optimizations
	if cfg.device == "cuda":
		torch.backends.cudnn.benchmark = True
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.set_float32_matmul_precision('high')

	print(f"Running DCGAN on {cfg.device} for Class {cfg.target_class}")

	# Prepare Data
	dataloader = get_dataloader(cfg)

	# Initialize Trainer
	model_trainer = DCGANTrainer(cfg, dataloader)

	# Train
	model_trainer.train()
