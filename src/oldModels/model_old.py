import torch
import torch.nn as nn


class Generator(nn.Module):
	"""
	Full Vanilla GAN Generator.
	((batch_size, noise_dimension) -> (batch_size, output_dimension))

	The Generator maps the noise distribution (latent space) to a generated Distribution (data space) which then gets
	aligned with the target Distribution (original data), using the discriminator, it plays against. The goal is to
	minimize the Discriminators Performance, meaning maximize its Loss (Binary Cross Entropy Loss, for Derivation see
	extra file) in order to maximize the Probability of it making a mistake.

	Since the Latent Space (noise Distribution) often has fewer dimensions than the Data Space it uses a Funnel Style
	Architecture in Order to Unpack the Information of the noise Vector into a larger data vector
	"""

	def __init__(self, noise_dimension, output_dimension, hidden_dimension, leak_factor):
		super(Generator, self).__init__()
		self.noise_dimension = noise_dimension

		self.network = nn.Sequential(
			# Linear Layer, transforms Tensor of input_dimension into output_dimension
			nn.Linear(noise_dimension, hidden_dimension),
			# BatchNorm normalizes Output of the linear Layer to have mean 0 and variance 1
			# in order to limit how much the signal can shift around to avoid exploding or vanishing gradients.
			# It helps with avoiding Mode Collapse
			nn.BatchNorm1d(hidden_dimension),
			# LeakyReLu Helps if the Discriminator is too Strong. If it is it sends large negative Gradients for all
			# images which would all get mapped to 0, leaving no Feedback for the Generator, LeakyRelu multiplies all
			# negative numbers by .2 and keeps all positive numbers as they are.
			nn.LeakyReLU(leak_factor, inplace=True),

			nn.Linear(hidden_dimension, hidden_dimension * 2),
			nn.BatchNorm1d(hidden_dimension * 2),
			nn.LeakyReLU(leak_factor, inplace=True),

			nn.Linear(hidden_dimension * 2, hidden_dimension * 4),
			nn.BatchNorm1d(hidden_dimension * 4),
			nn.LeakyReLU(leak_factor, inplace=True),

			# The last hidden Layer gets projected to the image size,
			# No BatchNorm since we want the raw pixel values, no normalized distribution
			nn.Linear(hidden_dimension * 4, output_dimension),
			# Maps all Pixel Values into the range [-1, 1], (pixels have fixed boundary, the red channel can not be 999)
			nn.Tanh()
		)

	def forward(self, generator_input):
		"""Forward pass of the Generator.

		Takes a batch of random noise vectors (Tensor of shape (batch_size, noise_dimension)) and
		returns A batch of generated flattened images (Tensor of shape (batch_size, output_dimension)),
		where the pixel values are capped at [-1, 1]."""
		return self.network(generator_input)

	def generate(self, noise_input=None, num_samples=1, device="cpu"):
		"""
		Function to generate samples.
		:param noise_input: Optional Parameter, the specific noise Tensor to generate from.
			(Shape: (num_samples, noise_dimension))
		:param num_samples: Optional Parameter to specify the amount of Samples to Generate.
		:param device: Optional Parameter to specify the device to generate the samples on (cpu / cuda)
		:return: Returns a Tensor with num_samples of generated Data in range [-1, 1]
			(Shape: (num_samples, output_dimension)).
		"""
		self.eval()  # Do not train on these test images

		with torch.no_grad():  # Disable gradient calculation for generating testing images

			if noise_input is not None:  # Input Given
				if not isinstance(noise_input, torch.Tensor):  # -> cast to Tensor
					noise_input = torch.tensor(noise_input, dtype=torch.float32)

				generator_input = noise_input.to(device)
			else:  # No Input, Generate Random Samples
				generator_input = torch.randn(num_samples, self.noise_dimension).to(device)

			generated_data = self.forward(generator_input)

		self.train()  # Set Model back to Train Mode
		return generated_data


class Discriminator(nn.Module):
	"""
	Full Vanilla GAN Discriminator.
	((batch_size, input_dimension) -> (batch_size, 1))

	The Discriminator acts as a Binary Classifier, taking an input vector and returning
	the probability that this image (the vector) is Real.
	"""
	def __init__(self, input_dimension, hidden_dimension, dropout, leak_factor):
		super(Discriminator, self).__init__()

		self.net = nn.Sequential(
			# Slowly reducing the Amount of Features to 1
			nn.Linear(input_dimension, hidden_dimension * 2),
			nn.LeakyReLU(leak_factor, inplace=True),
			# The Dropout helps with vanishing Gradients, ensuring the Discriminator does not learn too fast
			nn.Dropout(dropout),

			nn.Linear(hidden_dimension * 2, hidden_dimension),
			nn.LeakyReLU(leak_factor, inplace=True),
			nn.Dropout(dropout),

			nn.Linear(hidden_dimension, 1),
			# Maps Output to a Probability [0, 1] (0 -> Fake, 1 -> Real)
			nn.Sigmoid()
		)

	def forward(self, discriminator_input):
		"""Forward pass of the Discriminator.

		Takes a flattened image tensor (Shape: (batch_size, input_dimension)) and returns
		a probability estimation (Shape: (batch_size, 1))."""
		return self.net(discriminator_input)


class GAN:
	@staticmethod
	def build_gan(data_dimension, noise_dimension, hidden_dimension, device='cpu', dropout=0.3, leak_factor=0.2):
		"""
		Builds and Returns the GAN Models

		Args:
			:param data_dimension: The size of the real images
			:param noise_dimension: Size of the noise vector
			:param hidden_dimension: Width of the hidden Layers
			:param device: 'cpu' or 'cuda'
			:param dropout: Dropout probability for the Discriminator
			:param leak_factor: Factor for the LeakyReLu

		Returns:
			:return Returns a Tuple consisting of the Generator, followed by the Discriminator
		"""
		G = Generator(noise_dimension, data_dimension, hidden_dimension, leak_factor).to(device)
		D = Discriminator(data_dimension, hidden_dimension, dropout, leak_factor).to(device)

		# Initialize Parameters
		G.apply(GAN.init_weights)  # Try to Initialize weights on every part of the NN.
		D.apply(GAN.init_weights)

		return G, D

	@staticmethod
	def init_weights(layer):
		"""
		Initialize the Parameters of the Linear Layers,
		Xavier-initialization is used to maintain a good variance balance in weights,
		since it led to good results in literature.
		"""
		if isinstance(layer, nn.Linear):
			# Fills the weights with values drawn from a normal distribution
			nn.init.xavier_uniform_(layer.weight)
			# Set bias to a small constant (or 0) to prevent dead neurons at start
			layer.bias.data.fill_(0.01)
