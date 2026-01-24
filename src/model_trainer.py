import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
import time
import os
import csv
from datetime import datetime

from model import Generator, Discriminator, weights_init
import utils


class DCGANTrainer:
	"""
	Handles the training of the DCGAN.
	Handling the model initialization, model training, evaluation, and logging.
	"""
	def __init__(self, cfg, dataloader):
		self.cfg = cfg
		self.dataloader = dataloader
		self.device = torch.device(cfg.device)

		# Setup
		self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		self.output_folder = os.path.join(self.cfg.save_dir, f"{self.timestamp}_DCGAN")
		self.image_folder = os.path.join(self.output_folder, "progress_images")

		os.makedirs(self.output_folder, exist_ok=True)
		os.makedirs(self.image_folder, exist_ok=True)

		print(f"Logging to: {self.output_folder}")

		# Build models
		self.netG = Generator(cfg.noise_dim, cfg.nc).to(self.device)
		# In- or Exclude Sigmoid depending on the Loss Function
		self.netD = Discriminator(cfg.nc, includeSigmoid=False).to(self.device)

		# Setup models depending on device (cpu vs gpu)
		self.setup_device_specifics()
		# Load saved weights or initiate random weights if none are saved.
		self.initialize_weights()
		# Exponential Moving Average of generator weights for stabler evals
		self.init_ema()

		print(self.netG)
		print(self.netD)

		# Optimizers & Loss
		# Hinge loss: max(0, 1 - D_real) + max(0, 1 + D_fake)
		# Works better with spectral norm D and gives cleaner gradients than BCEWithLogits
		self.use_hinge_loss = True
		self.optimizerD = optim.Adam(self.netD.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, 0.999))
		self.optimizerG = optim.Adam(self.netG.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, 0.999))
		# Cosine decay for D to taper smoothly across full training
		self.schedulerD = optim.lr_scheduler.CosineAnnealingLR(
			self.optimizerD, T_max=self.cfg.num_epochs, eta_min=self.cfg.lr_d * 0.1
		)

		# Best FID tracking (EMA generator)
		self.best_fid = float("inf")
		self.best_epoch = -1
		self.best_ema_path = None

		# Metrics
		self.fixed_noise = torch.randn(64, cfg.noise_dim, 1, 1, device=self.device)
		self.fid_metric = FrechetInceptionDistance(normalize=True).to(self.device)
		self.is_metric = InceptionScore(normalize=True).to(self.device)

		# Tracking
		self.g_losses = []
		self.d_losses = []
		self.fid_scores = []
		self.is_scores = []
		self.epochs_list = []

	def setup_device_specifics(self):
		if self.device.type == "cuda":
			self.netG = self.netG.to(memory_format=torch.channels_last)
			self.netD = self.netD.to(memory_format=torch.channels_last)
			self.scaler = torch.cuda.amp.GradScaler()

			# Compile models for faster execution
			try:
				self.netG = torch.compile(self.netG)
				self.netD = torch.compile(self.netD)
				print("Models compiled successfully")
			except Exception as e:
				print(f"Model compilation not available: {e}")
		else:
			self.scaler = None

	def initialize_weights(self):
		# Load saved weights if they exist
		g_path = os.path.join(self.cfg.model_save_path, 'generator.pth')
		d_path = os.path.join(self.cfg.model_save_path, 'discriminator.pth')
		if os.path.exists(g_path) and os.path.exists(d_path):
			try:
				self.netG.load_state_dict(torch.load(g_path, map_location=self.device))
				self.netD.load_state_dict(torch.load(d_path, map_location=self.device))
				print("Loaded pre-trained models.")
			except Exception as _:
				print("Error loading models, starting from scratch")
				self.netG.apply(weights_init)
				self.netD.apply(weights_init)
		else:
			print("No saved models found, starting from scratch")
			self.netG.apply(weights_init)
			self.netD.apply(weights_init)

	def init_ema(self):
		self.ema_decay = 0.999
		self.ema_netG = Generator(self.cfg.noise_dim, self.cfg.nc).to(self.device)
		self.ema_netG.load_state_dict(self.netG.state_dict())
		self.ema_netG.eval()
		for p in self.ema_netG.parameters():
			p.requires_grad = False

	@torch.no_grad()
	def update_ema(self):
		# Update EMA weights with current generator parameters
		model_params = dict(self.netG.named_parameters())
		ema_params = dict(self.ema_netG.named_parameters())
		for name, param in model_params.items():
			ema_params[name].data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

		# Keep BatchNorm/other buffers in sync
		model_buffers = dict(self.netG.named_buffers())
		ema_buffers = dict(self.ema_netG.named_buffers())
		for name, buf in model_buffers.items():
			ema_buf = ema_buffers[name]
			if buf.is_floating_point():
				ema_buf.data.mul_(self.ema_decay).add_(buf.data, alpha=1 - self.ema_decay)
			else:
				ema_buf.data.copy_(buf.data)

	def train_epoch(self, epoch):
		"""Runs one full epoch of training."""
		for i, data in enumerate(self.dataloader):
			b_size = data[0].size(0)

			# Non_blocking allows data transfer to GPU to overlap with computations
			real = data[0].to(self.device, non_blocking=True)
			if self.device.type == "cuda":
				real = real.to(memory_format=torch.channels_last)
			real.requires_grad_(True)

			# Generate inputs
			noise = torch.randn(b_size, self.cfg.noise_dim, 1, 1, device=self.device)
			label_real = torch.full((b_size,), 0.9, device=self.device)
			label_fake = torch.full((b_size,), 0.1, device=self.device)

			# Train Discriminator
			self.netD.zero_grad(set_to_none=True)

			# Train with Real Batch
			# Use AMP if available
			with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
				output_real = self.netD(real)
				# Hinge loss: penalize D if it underestimates real score (want D_real >= 1)
				errD_real = torch.nn.functional.relu(1.0 - output_real).mean()

				# R1 penalty on real images applied sparsely to avoid over-regularizing D
				r1_gamma = 0.5
				if i % 16 == 0:
					grad_real = torch.autograd.grad(outputs=output_real.sum(), inputs=real, create_graph=True)[0]
					grad_penalty = grad_real.pow(2).reshape(b_size, -1).sum(dim=1).mean()
					errD_real = errD_real + 0.5 * r1_gamma * grad_penalty

				# Backward pass
				if self.scaler:
					self.scaler.scale(errD_real).backward()
				else:
					errD_real.backward()

			# Train with Fake Batch
			with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
				fake = self.netG(noise)
				output_fake = self.netD(fake.detach())  # Detach to not update Generator weights while training the Discriminator
				# Hinge loss: penalize D if it overestimates fake score (want D_fake <= -1)
				errD_fake = torch.nn.functional.relu(1.0 + output_fake).mean()

				# Backward pass
				if self.scaler:
					self.scaler.scale(errD_fake).backward()
					# Gradient clipping
					self.scaler.unscale_(self.optimizerD)
					torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=1.0)
					self.scaler.step(self.optimizerD)
					self.scaler.update()
				else:
					errD_fake.backward()
					# Gradient clipping
					torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=1.0)
					self.optimizerD.step()

			# calculate total loss (not used for training, only logging)
			errD = errD_real.item() + errD_fake.item()

			# Train Generator
			self.netG.zero_grad(set_to_none=True)
			with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
				# Discriminator gets fake images
				output = self.netD(fake)
				# Hinge loss for G: penalize if D_fake < -1 (want D_fake high, ideally close to 1)
				errG = -output.mean()

			# Backward pass
			if self.scaler:
				self.scaler.scale(errG).backward()
				# Gradient clipping
				self.scaler.unscale_(self.optimizerG)
				torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
				self.scaler.step(self.optimizerG)
				self.scaler.update()
			else:
				errG.backward()
				# Gradient clipping
				torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
				self.optimizerG.step()

			# Train Generator a second time per iteration with fresh noise until epoch 80
			if epoch < 80:
				self.netG.zero_grad(set_to_none=True)
				with torch.cuda.amp.autocast(enabled=(self.scaler is not None)):
					noise2 = torch.randn(b_size, self.cfg.noise_dim, 1, 1, device=self.device)
					fake2 = self.netG(noise2)
					output2 = self.netD(fake2)
					errG2 = -output2.mean()

				# Backward pass for second generator update
				if self.scaler:
					self.scaler.scale(errG2).backward()
					# Gradient clipping
					self.scaler.unscale_(self.optimizerG)
					torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
					self.scaler.step(self.optimizerG)
					self.scaler.update()
				else:
					errG2.backward()
					# Gradient clipping
					torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0)
					self.optimizerG.step()

			# Update EMA once per iteration using the latest generator weights
			self.update_ema()


			if i % 50 == 0:
				print(f"[{epoch}/{self.cfg.num_epochs}] "
				      f"[{i}/{len(self.dataloader)}] "
				      f"Loss_D: {errD:.4f} "
				      f"Loss_G: {errG:.4f}")

			# Store scalar values on CPU to avoid GPU tensors leaking into numpy/plotting
			self.g_losses.append(errG.item())
			self.d_losses.append(errD)

	def evaluate(self, epoch):
		"""Computes FID and IS scores."""
		print(f"Evaluating at epoch {epoch}...")

		# Switch to Eval mode and pick EMA generator if available
		gen_for_eval = getattr(self, "ema_netG", self.netG)
		self.netG.eval()
		if gen_for_eval is not self.netG:
			gen_for_eval.eval()

		num_samples = self.cfg.num_samples_eval
		batch_size_eval = 64
		fake_images_list = []
		real_images_list = []

		# get real images
		# We create a new iterator so we don't mess up the main training loop state
		temp_loader = iter(self.dataloader)

		try:
			while len(real_images_list) * self.cfg.batch_size < num_samples:
				# Get next batch
				batch_data = next(temp_loader)
				real_batch = batch_data[0].to(self.device)
				real_images_list.append(real_batch)
		except StopIteration:
			pass  # dataset is empty (all samples used)

		# Concatenate real images and crop to exact num_samples
		real_images = torch.cat(real_images_list, dim=0)[:num_samples]

		# Generate fake images
		num_batches = (num_samples + batch_size_eval - 1) // batch_size_eval

		# No Gradients, since we won't train on the results
		with torch.no_grad():
			for _ in range(num_batches):
				noise_values = torch.randn(batch_size_eval, self.cfg.noise_dim, 1, 1, device=self.device)
				fake_batch = gen_for_eval(noise_values)
				fake_images_list.append(fake_batch.detach())

		fake_images = torch.cat(fake_images_list, dim=0)[:num_samples]

		# Compute Metrics
		# Denormalize images for metric computation (scale from [-1,1] to [0,255])
		real_images_denorm = ((real_images + 1) / 2 * 255).to(torch.uint8)
		fake_images_denorm = ((fake_images + 1) / 2 * 255).to(torch.uint8)

		# Compute FID and IS with error handling
		try:
			# Update metric states
			self.fid_metric.update(real_images_denorm, real=True)
			self.fid_metric.update(fake_images_denorm, real=False)

			fid_score = self.fid_metric.compute()
			fid_value = fid_score.item()
			self.fid_scores.append(fid_value)
			self.fid_metric.reset()

			self.is_metric.update(fake_images_denorm)
			is_score = self.is_metric.compute()  # Tuple is (mean, std)
			is_value = is_score[0].item()
			self.is_scores.append(is_value)
			self.is_metric.reset()

			self.epochs_list.append(epoch)

			# Best FID checkpointing (EMA generator)
			if fid_value < self.best_fid:
				self.best_fid = fid_value
				self.best_epoch = epoch
				best_path = os.path.join(self.output_folder_out, "best_generator_ema.pth")
				torch.save(gen_for_eval.state_dict(), best_path)
				self.best_ema_path = best_path

			print(f"Epoch [{epoch}/{self.cfg.num_epochs}] - FID: {fid_score:.4f}, IS: {is_score[0]:.4f}")
		except Exception as e:
			print(f"Warning: Could not compute metrics at epoch {epoch}: {e}")
			# Reset Metrics in case of exception
			self.fid_metric.reset()
			self.is_metric.reset()

		# Switch back to Train mode to continue training in the next epoch
		self.netG.train()

	def train(self):
		"""Handles the training Loop"""
		print("Starting Training Loop...")
		start_time = time.monotonic()

		output_folder = os.path.join(self.cfg.save_dir, f"{self.timestamp}_DCGAN_output")
		self.output_folder_out = output_folder
		os.makedirs(output_folder, exist_ok=True)

		for epoch in range(self.cfg.num_epochs):
			self.train_epoch(epoch)
			#self.save_progress(epoch)

			if epoch % self.cfg.metric_eval_freq == 0:
				self.evaluate(epoch)

			# Step LR schedulers at epoch boundaries
			self.schedulerD.step()

		end_time = time.monotonic()
		total_time = end_time - start_time
		print(f"Training finished in {total_time:.2f}seconds")

		self.save_results(output_folder, total_time)

	def save_progress(self, epoch):
		"""Generates and saves a grid using fixed_noise."""
		self.netG.eval()  # Switch to eval mode for generation
		with torch.no_grad():
			fake = self.netG(self.fixed_noise)
			utils.save_epoch_image(fake, self.image_folder, epoch)
		self.netG.train()  # Switch back to train mode

	def save_results(self, output_folder, total_time):
		gen_to_save = getattr(self, "ema_netG", self.netG)
		# Save Models
		os.makedirs(self.cfg.model_save_path, exist_ok=True)
		torch.save(self.netG.state_dict(), os.path.join(self.cfg.model_save_path, 'generator.pth'))
		torch.save(gen_to_save.state_dict(), os.path.join(self.cfg.model_save_path, 'generator_ema.pth'))
		torch.save(self.netD.state_dict(), os.path.join(self.cfg.model_save_path, 'discriminator.pth'))

		# Save Plots
		utils.save_plots(output_folder, self.g_losses, self.d_losses,
		                 self.fid_scores, self.is_scores, self.epochs_list, self.timestamp)

		# Generate final sample grid
		with torch.no_grad():
			fake_batch = gen_to_save(self.fixed_noise)
			real_batch = next(iter(self.dataloader))[0].to(self.device)
			utils.save_image_grid(real_batch, fake_batch, output_folder, self.timestamp)

		print(f"All plots saved to: {output_folder}")

		# Save CSV
		csv_path = os.path.join(output_folder, "losses_and_metrics.csv")

		with open(csv_path, "w", newline="") as f:
			writer = csv.writer(f)
			writer.writerow(["Iteration", "Generator Loss", "Discriminator Loss", "Epoch", "FID Score", "IS Score"])

			max_len = max(len(self.g_losses), len(self.epochs_list))

			for i in range(max_len):
				row = [
					i,
					self.g_losses[i] if i < len(self.g_losses) else "",
					self.d_losses[i] if i < len(self.d_losses) else "",
					self.epochs_list[i] if i < len(self.epochs_list) else "",
					self.fid_scores[i] if i < len(self.fid_scores) else "",
					self.is_scores[i] if i < len(self.is_scores) else ""
				]
				writer.writerow(row)

		print(f"Losses and metrics exported to: {csv_path}")

		param_file = os.path.join(output_folder, f"training_parameters_{self.timestamp}.txt")

		with open(param_file, "w") as f:
			f.write("DCGAN Training Parameters\n")
			f.write("=========================\n\n")
			f.write(f"Date & Time: {self.timestamp}\n")
			f.write(f"Target Class (CIFAR-10): {self.cfg.target_class}\n")
			f.write(f"Number of Epochs: {self.cfg.num_epochs}\n")
			f.write(f"Batch Size: {self.cfg.batch_size}\n")
			f.write(f"Learning Rate Generator: {self.cfg.lr_g}\n")
			f.write(f"Learning Rate Discriminator: {self.cfg.lr_d}\n")
			f.write(f"Beta1: {self.cfg.beta1}\n")
			f.write(f"Noise Dimension: {self.cfg.noise_dim}\n")
			f.write(f"Image Size: {self.cfg.image_size}x{self.cfg.image_size}\n")
			f.write(f"Number of Channels: {self.cfg.nc}\n")

			f.write(f"Optimizer Discriminator: {self.optimizerD.__str__()}\n")
			f.write(f"Optimizer Generator: {self.optimizerG.__str__()}\n")
			#f.write(f"Loss Function: {self.criterion.__str__()}\n")
			f.write(f"Total Training Time: {total_time:.2f} seconds\n\n")

			f.write("Network Architecture:\n")
			f.write(self.netG.__str__() + "\n\n")
			f.write(self.netD.__str__() + "\n")

		print(f"Training parameters saved to: {param_file}")