from flask import Flask, render_template, jsonify
import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

app = Flask(__name__)

# =========================================
# SETTINGS
# =========================================

nz = 100
ngf = 64
nc = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# =========================================
# GENERATOR
# =========================================

class Generator(nn.Module):

    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()

        self.main = nn.Sequential(

            nn.ConvTranspose2d(
                nz, ngf * 8,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),

            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                ngf * 8, ngf * 4,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                ngf * 4, ngf * 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                ngf * 2, ngf,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                ngf, nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


# =========================================
# LOAD MODEL
# =========================================

model_path = "generator.pth"

if not os.path.exists(model_path):
    raise FileNotFoundError(
        "generator.pth not found in the project folder!"
    )

print("Loading generator.pth...")

netG = Generator(nz, ngf, nc).to(device)

netG.load_state_dict(
    torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )
)

netG.eval()

print("Generator loaded successfully!")


# =========================================
# HOME PAGE
# =========================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================
# GENERATE FACE
# =========================================

@app.route("/generate")
def generate():

    print("Generating face...")

    # Random noise
    noise = torch.randn(
        1,
        nz,
        1,
        1,
        device=device
    )

    # Generate image
    with torch.no_grad():
        fake_image = netG(noise)

    # Create static folder
    os.makedirs("static", exist_ok=True)

    # Save image
    image_path = "static/generated_face.png"

    save_image(
        fake_image,
        image_path,
        normalize=True,
        value_range=(-1, 1)
    )

    print("Face generated successfully!")

    return jsonify({
        "success": True,
        "image": "/static/generated_face.png"
    })


# =========================================
# RUN APP
# =========================================

if __name__ == "__main__":
    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )