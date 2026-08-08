import torch
import pytest
from PIL import Image
from src.clip_encoder import CLIPEncoder


@pytest.fixture(scope="module")
def clip_encoder():
    """Fixture providing a CPU-bound CLIPEncoder instance for fast unit testing."""
    return CLIPEncoder(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu")


def test_clip_image_encoding_shape_and_norm(clip_encoder):
    """Verify single image encoding output shape [1, 512] and L2 norm = 1.0."""
    dummy_img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    vec = clip_encoder.encode_image(dummy_img)

    assert vec.shape == (1, 512)
    norm = torch.norm(vec, p=2, dim=-1)
    assert torch.allclose(norm, torch.tensor([1.0]), atol=1e-5)


def test_clip_text_encoding_shape_and_norm(clip_encoder):
    """Verify text query encoding output shape [1, 512] and L2 norm = 1.0."""
    prompt = "a small black bird"
    vec = clip_encoder.encode_text(prompt)

    assert vec.shape == (1, 512)
    norm = torch.norm(vec, p=2, dim=-1)
    assert torch.allclose(norm, torch.tensor([1.0]), atol=1e-5)


def test_clip_batch_image_encoding(clip_encoder):
    """Verify batch image encoding output shape [B, 512] and L2 norm = 1.0."""
    images = [
        Image.new("RGB", (224, 224), color=(255, 0, 0)),
        Image.new("RGB", (224, 224), color=(0, 255, 0)),
    ]
    vecs = clip_encoder.encode_images_batch(images)

    assert vecs.shape == (2, 512)
    norms = torch.norm(vecs, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)
