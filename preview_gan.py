import torch
from torchvision.utils import save_image

from app.gan_model import Generator


NOISE_DIM = 100

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

generator = Generator(noise_dim=NOISE_DIM).to(device)

generator.load_state_dict(
    torch.load(
        "models/mnist_generator.pth",
        map_location=device,
        weights_only=True,
    )
)

generator.eval()

with torch.no_grad():
    noise = torch.randn(
        16,
        NOISE_DIM,
        device=device,
    )

    generated_images = generator(noise)

    generated_images = (generated_images + 1) / 2

save_image(
    generated_images,
    "generated_digits.png",
    nrow=4,
)

print("Generated image saved to generated_digits.png")