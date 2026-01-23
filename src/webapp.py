from flask import Flask, render_template
import torch
from model import Generator
import torchvision.utils as vutils
import io
import base64
from PIL import Image

app = Flask(__name__, template_folder="templates")

# Initialize generator during startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(noise_dim=100, number_of_channels=3).to(device)
generator.load_state_dict(torch.load('config/generator.pth', map_location=device, weights_only=True))
generator.eval()


@app.route("/")
def hello():
    # Generate a new image on each request
    with torch.no_grad():
        noise = torch.randn(1, 100, 1, 1, device=device)
        fake_image = generator(noise)
        
    # Convert tensor to image
    img_tensor = fake_image[0].cpu()
    img_tensor = (img_tensor + 1) / 2  # Normalize from [-1, 1] to [0, 1]
    img_tensor = torch.clamp(img_tensor, 0, 1)
    
    # Convert to PIL Image
    img = vutils.make_grid(img_tensor, normalize=False)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype('uint8')
    pil_img = Image.fromarray(img)
    
    # Convert to base64 for HTML display
    buffer = io.BytesIO()
    pil_img.save(buffer, format='PNG')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    
    return render_template('index.html', img_base64=img_base64)


if __name__ == "__main__":
    app.run(debug=True)
