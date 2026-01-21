from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


def get_dataloader(cfg):
	"""
	Creates and returns the DataLoader for a specific class of CIFAR-10.
	The Target class is specified in config.py under 'target_class'
	"""
	transform = transforms.Compose([
		transforms.Resize(cfg.image_size),  # added resize in case we do in fact now go for 64x64
		transforms.ToTensor(),  # converts the PIL image to [0, 255] to a Tensor [0, 1]
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalises the [0, 1] to a [-1, 1] tensor
	])

	# load the full dataset
	dataset = CIFAR10(
		root=cfg.dataroot,
		train=True,
		download=True,
		transform=transform
	)

	# Filter for specific class
	indices = [i for i, t in enumerate(dataset.targets) if t == cfg.target_class]
	dataset = Subset(dataset, indices)

	dataloader = DataLoader(
		dataset,
		batch_size=cfg.batch_size,
		shuffle=True,
		num_workers=cfg.num_workers,
		drop_last=True,
		pin_memory=cfg.device == "cuda",
		persistent_workers=cfg.num_workers > 0,
		prefetch_factor=4 if cfg.num_workers > 0 else None
	)

	return dataloader
