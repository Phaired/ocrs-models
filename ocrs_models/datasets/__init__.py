import io
import random

from PIL import Image
import torch
from torchvision import transforms


class SaltAndPepperNoise:
    """Add salt-and-pepper noise to an image tensor."""

    def __init__(self, amount: float = 0.02):
        self.amount = amount

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        noise_mask = torch.rand_like(img)
        salt = noise_mask < self.amount / 2
        pepper = noise_mask > 1 - self.amount / 2
        img = img.clone()
        img[salt] = 0.5  # White in [-0.5, 0.5] range
        img[pepper] = -0.5  # Black in [-0.5, 0.5] range
        return img


class MotionBlur:
    """Apply motion blur via a directional convolution kernel."""

    def __init__(self, kernel_size: int = 7):
        self.kernel_size = kernel_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        k = random.choice([3, 5, self.kernel_size])
        # Random angle: horizontal or vertical motion
        if random.random() < 0.5:
            kernel = torch.zeros(1, 1, 1, k)
        else:
            kernel = torch.zeros(1, 1, k, 1)
        kernel.fill_(1.0 / k)

        c = img.shape[0] if img.ndim == 3 else 1
        if img.ndim == 3:
            kernel = kernel.expand(c, -1, -1, -1)

        padding = [s // 2 for s in kernel.shape[2:]]
        return torch.nn.functional.conv2d(
            img.unsqueeze(0) if img.ndim == 3 else img.unsqueeze(0).unsqueeze(0),
            kernel,
            padding=padding,
            groups=c,
        ).squeeze(0)


class JPEGCompressionArtifacts:
    """Simulate JPEG compression artifacts by encoding/decoding with random quality."""

    def __init__(self, quality_range: tuple[int, int] = (20, 95)):
        self.quality_range = quality_range

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        quality = random.randint(*self.quality_range)

        # Convert from [-0.5, 0.5] to [0, 255] uint8 for JPEG encoding
        pil_img = Image.fromarray(
            ((img.squeeze(0).numpy() + 0.5) * 255).clip(0, 255).astype("uint8"),
            mode="L",
        )

        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)

        # Convert back to [-0.5, 0.5] tensor
        import numpy as np

        result = torch.from_numpy(np.array(compressed, dtype=np.float32))
        result = result / 255.0 - 0.5
        return result.unsqueeze(0)


def text_recognition_data_augmentations():
    """
    Create a set of data augmentations for use with text recognition.
    """

    # Fill color for empty space created by transforms.
    # This is the "black" value for normalized images.
    transform_fill = -0.5

    augmentations = transforms.RandomApply(
        [
            transforms.RandomChoice(
                [
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.RandomRotation(
                        degrees=5,
                        fill=transform_fill,
                        expand=True,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.Pad(padding=(5, 5), fill=transform_fill),
                    transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                    SaltAndPepperNoise(amount=0.02),
                    MotionBlur(kernel_size=7),
                    JPEGCompressionArtifacts(quality_range=(20, 95)),
                ]
            )
        ],
        p=0.5,
    )
    return augmentations
