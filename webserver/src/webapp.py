from flask import Flask, render_template
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.model import Generator
import src.utils
import src.config

template_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))
app = Flask(__name__, template_folder=template_path)

# Initialize generator during startup
cfg = src.config.Config() # Load default config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Generator = Generator(cfg.noise_dim, cfg.nc).to(device)
weights_path = os.path.join('..', 'config', 'best_generator_ema.pth')

try:
    Generator.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    Generator.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model not found at {weights_path}")


@app.route("/")
def hello():
    if Generator is None:
        return "No model loaded"

    img = src.utils.generate_single_image(Generator, device, cfg.noise_dim)
    return render_template('index.html', img_base64=img)


if __name__ == "__main__":
    app.run(debug=True)
