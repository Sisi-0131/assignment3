import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from app.gan_model import Generator, Discriminator


BATCH_SIZE = 128
NOISE_DIM = 100
LEARNING_RATE = 0.0002
EPOCHS = 10

device = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

if __name__ == "__main__":
    images, labels = next(iter(train_loader))

    print("Using device:", device)
    print("Number of training images:", len(train_dataset))
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Minimum pixel value:", images.min().item())
    print("Maximum pixel value:", images.max().item())
    
    generator = Generator(noise_dim=NOISE_DIM).to(device)
    discriminator = Discriminator().to(device)

    loss_function = torch.nn.BCELoss()

    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    for epoch in range(EPOCHS):
        generator_loss_total = 0.0
        discriminator_loss_total = 0.0

        for batch_index, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(device)
            current_batch_size = real_images.size(0)

            real_labels = torch.ones(
                current_batch_size,
                1,
                device=device,
            )

            fake_labels = torch.zeros(
                current_batch_size,
                1,
                device=device,
            )

            # Train the Discriminator
            discriminator_optimizer.zero_grad()

            real_predictions = discriminator(real_images)
            real_loss = loss_function(
                real_predictions,
                real_labels,
            )

            noise = torch.randn(
                current_batch_size,
                NOISE_DIM,
                device=device,
            )

            fake_images = generator(noise)
            fake_predictions = discriminator(fake_images.detach())
            fake_loss = loss_function(
                fake_predictions,
                fake_labels,
            )

            discriminator_loss = (real_loss + fake_loss) / 2
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # Train the Generator
            generator_optimizer.zero_grad()

            generator_predictions = discriminator(fake_images)
            generator_loss = loss_function(
                generator_predictions,
                real_labels,
            )

            generator_loss.backward()
            generator_optimizer.step()

            generator_loss_total += generator_loss.item()
            discriminator_loss_total += discriminator_loss.item()

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS}, "
                    f"Batch {batch_index + 1}/{len(train_loader)}, "
                    f"D Loss: {discriminator_loss.item():.4f}, "
                    f"G Loss: {generator_loss.item():.4f}"
                )

        average_generator_loss = (
            generator_loss_total / len(train_loader)
        )
        average_discriminator_loss = (
            discriminator_loss_total / len(train_loader)
        )

        print(
            f"Epoch {epoch + 1} completed. "
            f"Average D Loss: {average_discriminator_loss:.4f}, "
            f"Average G Loss: {average_generator_loss:.4f}"
        )

    torch.save(
        generator.state_dict(),
        "models/mnist_generator.pth",
    )

    print("Generator saved to models/mnist_generator.pth")