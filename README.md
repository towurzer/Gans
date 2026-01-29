# DCGAN for CIFAR-10 Car Generation

This project implements a (modernized) Deep Convolutional Generative Adversarial Network (DCGAN), designed and trained to generate 
RGB Images of cars in the Format of 32x32 pixels.

It is trained on the car class of the
[CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html), which consists of 6000 rgb images per class.

It is based upon the [original GAN Paper](https://arxiv.org/pdf/1406.2661) and more importantly the standard Architecture
established in the [original DCGAN Paper](https://arxiv.org/pdf/1511.06434) , wöhich was then improved by modern ideas (like 
Spectral Normalization, Hinge Loss, EMA) in order to handle common Training issues with GAN's, like training instability mainly due to 
too strong discriminators and mode collapse.

The repo includes a training pipeline which is equiped with metric tracking (FID and IS) and logging and a 
Flask-based webserver for presentation.
It is Modulized and based on object oriented software engineering principles to make it readable, servicable and expandable. 
We tried to follow [python naming conventions](https://peps.python.org/pep-0008/#prescriptive-naming-conventions)
as closely as possible (but since our backgrounds come from different programming languages we did not succeed everywhere).

## Project Structure
	


```text
src/
	/data 					# Auto Generated Folder with the dataset
	/oldModels
		model_old.py			# Old Vanilla GAN, used beofore upgrading to a DC-Gan
	config.py				# Model / Training Settings and Hyperparameters
	dataset.py				# Data Pipeline: Downloads and filters the CIFAR-10 Dataset
	evaluation_snippets.py			# (Depricated) Evaluation Image generation, mostly moved to utils.py^
	main.py					# Manages the whole pipeline, loading, setup, data preparation, and model trainer initialization
	model.py				# Generator and Discriminator Architectures
	model_trainer.py			# Manages the training loop, evaluation and logging
	utils.py				# Utility functions to help visualize the progress, seed the model trainer and more helper functions
webserver/
	/config					# Includes the generator to load and run when generating images on the webserver (.pth files)
	/src/webapp.py				# Flask Webapp, Loading the Model and Serving generated Images
	/templates/indes.html			# The website itself
config/						# trained models of Generator and Discriminator (.pth)
logs/						# includes folders, for each training process, which includes following files:
							# The Generator Model after training is complete,
							# images of the FID, Loss Curves, Loss Varainces,
							# as well ese as a .csv file consisting the losses of the models for each iteration
							# of the training process, some final image comparison and the 
							# config and Architecture at the time of training.
requirements.txt				# Dependencies
```

## Getting Started
**1. Installation**

Run
```bash
pip install -r requirements.txt
```
to install neccessary requirements.

**2. Training the Model**

To reproduce the results, run the main training script.
```bash
python src/main.py
```
+ This will Load the current Configuration
+ automatically download the dataset
+ setup the trainer
	+ which will automatically try to load preexisting models 
		and generate new ones if none are found.
+ start the training process on the device avaliable.
	+ which will automaticall produce a new log folder in /logs with the resulting data

<br>

**3. Run the webserver (optional)**

To run the web server run 
```bash
python webserver/src/webapp.py
```
and open it using the browser of your choice on http://localhost:5000/

Refresh the page to generate a new sample.


## Architecture
### Model Design
The Generator uses transposed convolutional layers to upsample a 120 dimensional noise vector to a 32x32 Image, across 3 channels. 
It uses Instance Normalization (instead of Batch Norm) this significantly improved training stability when testing, since it 
forces the generator to learn better features by removing instance specific contrast information.

The Discriminator takes the 3x32x32 image and is is a standard binary (CNN)classifier, downsampling it to a scalar probability score.
It uses spectral Normalization on the Convolutional layers preventingit from becomming too strong too quickly. Which was a main and persistent problem across training.

### Training Strategy
The proposed BCE Loss was Replaced with Hing Loss, provides cleaner gradients even when the discriminator is very confident. 

We also Implemented an R1 Gradient Penalty (using lazy regularization every 16 steps, to improve training time), to stabilize training by penalizing the gradients on real data.

Additionally we added EMA (Exponential Moving Average), keeping a copy of the generators weights, which updates very slowly and avoids oscillations around the optimum weights.

## Results
![Comparison of Real and Fake images of the final GAN](logs/20260124_183417_DCGAN_output/real_vs_fake_20260124_183417.png)
    
    Final FID Score: xxx

    Best Epoch: xxx

    Hardware Used: xxx
