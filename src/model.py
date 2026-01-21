import torch.nn as nn


class Generator(nn.Module):
	"""
	Full DC GAN Generator.
	((batch_size, noise_dimension, 1, 1) -> (batch_size, number_of_channels, 32, 32))

	The Generator maps the noise distribution (latent space) to a generated Distribution (data space) which then gets
	aligned with the target Distribution (original data), using the discriminator, it plays against. The goal is to
	minimize the Discriminators Performance, meaning maximize its Loss (Binary Cross Entropy Loss, for Derivation see
	extra file) in order to maximize the Probability of it making a mistake.


	"""
	def __init__(self, noise_dim, number_of_channels):
		super(Generator, self).__init__()
		self.network = nn.Sequential(
			# Project the Latent Vector (1x1) into a 4x4 feature map.
			# bias=False because BatchNorm centers the data anyway.
			nn.ConvTranspose2d(noise_dim, 512, 4, 1, 0, bias=False),
			# Normalizes output to mean 0 and var 1 to stabilize training and help gradient flow.
			nn.BatchNorm2d(512),
			# Standard ReLU for the Generator. Allows learning non-linear patterns by zeroing negatives.
			nn.ReLU(True),

			# Upsamples 4x4 -> 8x8. (out = (in−1)×Stride−(2×Padding) + Kernel; out = 2in - 2 -  2 + 4)
			nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
			nn.BatchNorm2d(256),
			nn.ReLU(True),

			# Upsamples 8x8 -> 16x16. Reducing channels (depth) as image gets larger.
			nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
			nn.BatchNorm2d(128),
			nn.ReLU(True),

			# Final Upsample 16x16 -> 32x32 (Target Image Size).
			# No BatchNorm here: we want the raw pixel, color values, not a normalized distribution.
			nn.ConvTranspose2d(128, number_of_channels, 4, 2, 1, bias=False),
			#  Maps all Pixel Values into the range [-1, 1], (pixels have fixed boundary, the red channel can not be 999)
			nn.Tanh()
		)

	def forward(self, x):
		"""
		Forward pass of the Generator.

		Takes a batch of random noise vectors (Tensor of shape (batch_size, noise_dimension, 1, 1)) and
		returns A batch of generated flattened images (Tensor of shape (batch_size, number_of_channels, 32, 32)),
		where the pixel values are capped at [-1, 1].
		"""
		return self.network(x)


class Discriminator(nn.Module):
	"""
	Full DC GAN Discriminator.
	((batch_size, input_dimension, 32, 32) -> (batch_size, 1, 1, 1))

	The Discriminator acts as a Binary Classifier, taking an input vector and returning
	the probability that this image (the vector) is Real.
	"""
	def __init__(self, output_dimension, includeSigmoid):
		super().__init__()
		layers = [
			# Slowly reducing the Amount of Features to 1
			nn.Conv2d(output_dimension, 128, 4, 2, 1, bias=False),
			# LeakyReLU allows a small gradient when the unit is not active,
			# preventing "dead neurons" in the discriminator.
			nn.LeakyReLU(0.2, inplace=True),

			# Downsample: (128, 16, 16) -> (256, 8, 8)
			nn.Conv2d(128, 256, 4, 2, 1, bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True),

			# Downsample: (256, 8, 8) -> (512, 4, 4)
			nn.Conv2d(256, 512, 4, 2, 1, bias=False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace=True),

			# Final Downsample: (512, 4, 4) -> (1, 1, 1)
			nn.Conv2d(512, 1, 4, 1, 0, bias=False)
		]

		if includeSigmoid:
			# Maps Output to a Probability [0, 1]
			layers.append(nn.Sigmoid())

		self.network = nn.Sequential(*layers)

	def forward(self, x):
		"""
		Forward pass of the Discriminator.

		Takes an image tensor (Shape: (batch_size, number_of_channels, 32, 32)) and returns
		a probability estimation (Shape: (batch_size)).
		"""
		return self.network(x).view(-1)


def weights_init(layer):
	"""
	Weight Initialization scheme from the paper, in order to stabilize training,
	the layers get automatically passed via .apply()
	"""
	classname = layer.__class__.__name__

	# Conv2d / ConvTranspose2d Layers
	if classname.find("Conv") != -1:
		# weight ~ 0
		nn.init.normal_(layer.weight.data, 0.0, 0.02)

	# Batchnorm Layers
	elif classname.find("BatchNorm") != -1:
		# weight ~ 1 ; shift = 0
		nn.init.normal_(layer.weight.data, 1.0, 0.02)
		nn.init.constant_(layer.bias.data, 0)
