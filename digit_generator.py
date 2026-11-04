from io import BytesIO
import math
import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
from app.gan_model import Generator


class DigitGenerator:
    def __init__(
        self,
        model_path="models/mnist_generator.pth",
        noise_dim=100,
    ):
        self.device = torch.device("cpu")
        self.noise_dim = noise_dim

        self.model = Generator(
            noise_dim=self.noise_dim
        ).to(self.device)

        model_weights = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True,
        )

        self.model.load_state_dict(model_weights)
        self.model.eval()

    def generate(self, num_images=16):
        with torch.no_grad():
            noise = torch.randn(
                num_images,
                self.noise_dim,
                device=self.device,
            )

            generated_images = self.model(noise)
            generated_images = (generated_images + 1) / 2

        images_per_row = math.ceil(
            math.sqrt(num_images)
        )

        image_grid = make_grid(
            generated_images,
            nrow=images_per_row,
            padding=2,
        )

        pil_image = to_pil_image(
            image_grid.cpu()
        )

        image_buffer = BytesIO()
        pil_image.save(
            image_buffer,
            format="PNG",
        )

        return image_buffer.getvalue()