import torch
import torch.nn.functional as F
from PIL import Image
import open_clip


class CLIPEncoder:
    """Wrapper for OpenCLIP for frozen multimodal feature extraction.
    Provides an interface for extracting L2-normalized embeddings from images and text.

    Inspired by: https://github.com/mlfoundations/open_clip/tree/main

    Args:
        model_name: Name of the CLIP vision architecture (default: "ViT-B-32").
        pretrained: Pretrained weights tag (default: "laion2b_s34b_b79k").
        device: Target computation device ("cuda" or "cpu"). Defaults to CUDA if available.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.pretrained = pretrained

        # Load OpenCLIP model, transform, and tokenizer
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(model_name)

        # 
        model.eval()
        model.requires_grad_(False)

        self.model = model.to(self.device)
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    @torch.no_grad()
    def encode_image(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        """Extracts L2-normalized 512D embedding vector from a single image.

        Args:
            image: PIL Image instance or preprocessed PyTorch image tensor [3, H, W].

        Returns:
            L2-normalized embedding tensor of shape [1, 512].
        """
        if isinstance(image, Image.Image):
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image_tensor = image.unsqueeze(0).to(self.device)
            else:
                image_tensor = image.to(self.device)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        features = self.model.encode_image(image_tensor)
        return F.normalize(features, p=2, dim=-1)

    @torch.no_grad()
    def encode_images_batch(self, images: list[Image.Image] | torch.Tensor) -> torch.Tensor:
        """Extracts L2-normalized 512D embeddings from a batch of images.

        Args:
            images: List of PIL Image instances or batch tensor [B, 3, H, W].

        Returns:
            L2-normalized embeddings tensor of shape [B, 512].
        """
        if isinstance(images, list):
            tensors = [self.preprocess(img) for img in images]
            batch_tensor = torch.stack(tensors).to(self.device)
        elif isinstance(images, torch.Tensor):
            batch_tensor = images.to(self.device)
        else:
            raise TypeError(f"Unsupported images type: {type(images)}")

        features = self.model.encode_image(batch_tensor)
        return F.normalize(features, p=2, dim=-1)

    @torch.no_grad()
    def encode_text(self, text: str | list[str]) -> torch.Tensor:
        """Extracts L2-normalized 512D embedding vector from text query.

        Args:
            text: Text string or list of text strings (e.g., "a yellow bird").

        Returns:
            L2-normalized embedding tensor of shape [1, 512] or [B, 512].
        """
        if isinstance(text, str):
            text_list = [text]
        else:
            text_list = text

        tokens = self.tokenizer(text_list).to(self.device)
        features = self.model.encode_text(tokens)
        return F.normalize(features, p=2, dim=-1)
