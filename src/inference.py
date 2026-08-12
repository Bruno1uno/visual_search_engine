import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import faiss
import numpy as np
import torch
from PIL import Image

from src.clip_encoder import CLIPEncoder
from src.dataset import get_transforms
from src.model import load_resnet_model


@dataclass
class SearchResult:
    """Dataclass representing a single visual search result."""

    id: int
    image_id: int
    rel_path: str
    abs_path: str
    class_id: int
    class_name: str
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


class VisualSearchEngine:
    """Stable inference engine for visual similarity search and text-to-image retrieval.

    Loads trained ResNet encoder, OpenCLIP encoder, FAISS indices, and metadata lookup map
    once upon initialization into RAM for fast search queries.
    """

    def __init__(
        self,
        resnet_checkpoint_path: str | Path = "checkpoints/best_resnet34_proxy_anchor.pt",
        resnet_faiss_path: str | Path = "indices/resnet_cub200.faiss",
        clip_faiss_path: str | Path = "indices/clip_cub200.faiss",
        metadata_path: str | Path = "indices/id_to_metadata.json",
        device: str | torch.device | None = None,
    ):
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Load metadata mapping (ID -> image info)
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata mapping file not found at: {metadata_path}")
        
        with open(metadata_path, "r", encoding="utf-8") as f:
            raw_metadata = json.load(f)
            # Ensure keys are integer FAISS vector IDs
            self.id_to_metadata = {int(k): v for k, v in raw_metadata.items()}

        # Load FAISS search indices
        resnet_faiss_path = Path(resnet_faiss_path)
        clip_faiss_path = Path(clip_faiss_path)

        if not resnet_faiss_path.exists():
            raise FileNotFoundError(f"ResNet FAISS index not found at: {resnet_faiss_path}")
        if not clip_faiss_path.exists():
            raise FileNotFoundError(f"CLIP FAISS index not found at: {clip_faiss_path}")

        self.resnet_faiss = faiss.read_index(str(resnet_faiss_path))
        self.clip_faiss = faiss.read_index(str(clip_faiss_path))

        # Load ResNet model checkpoint and transforms
        resnet_checkpoint_path = Path(resnet_checkpoint_path)
        if not resnet_checkpoint_path.exists():
            raise FileNotFoundError(f"ResNet checkpoint not found at: {resnet_checkpoint_path}")

        self.resnet_model = load_resnet_model(str(resnet_checkpoint_path), device=self.device)
        self.resnet_model.eval()

        _, self.eval_transform = get_transforms()

        # Load OpenCLIP model wrapper
        self.clip_encoder = CLIPEncoder(
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            device=str(self.device),
        )

    @torch.inference_mode()
    def search_by_image(
        self,
        image: Image.Image,
        engine_type: str = "resnet",
        top_k: int = 8,
    ) -> list[dict]:
        """Performs image-to-image vector similarity search.

        Args:
            image: Query image (PIL Image instance).
            engine_type: Search engine variant ('resnet' or 'clip').
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of SearchResult dictionaries sorted by similarity score.
        """
        engine_type = engine_type.lower()
        if engine_type not in ("resnet", "clip"):
            raise ValueError(f"Invalid engine_type '{engine_type}'. Choose 'resnet' or 'clip'.")

        image_rgb = image.convert("RGB")

        if engine_type == "resnet":
            tensor_img = self.eval_transform(image_rgb).unsqueeze(0).to(self.device)
            vec = self.resnet_model(tensor_img).cpu().numpy().astype(np.float32)
            faiss.normalize_L2(vec)
            distances, indices = self.resnet_faiss.search(vec, top_k)
        else:
            vec_tensor = self.clip_encoder.encode_image(image_rgb)
            vec = vec_tensor.cpu().numpy().astype(np.float32)
            faiss.normalize_L2(vec)
            distances, indices = self.clip_faiss.search(vec, top_k)

        return self._format_results(indices[0], distances[0])

    @torch.inference_mode()
    def search_by_text(
        self,
        text_query: str,
        top_k: int = 8,
    ) -> list[dict]:
        """Performs text-to-image vector similarity search using OpenCLIP text encoder.

        Args:
            text_query: Search prompt string (e.g. "a yellow bird with black wings").
            top_k: Number of top matching images to return.

        Returns:
            List of SearchResult dictionaries sorted by similarity score.
        """
        if not text_query or not text_query.strip():
            return []

        text_vec_tensor = self.clip_encoder.encode_text(text_query.strip())
        vec = text_vec_tensor.cpu().numpy().astype(np.float32)
        faiss.normalize_L2(vec)

        distances, indices = self.clip_faiss.search(vec, top_k)
        return self._format_results(indices[0], distances[0])

    def _format_results(self, indices: np.ndarray, distances: np.ndarray) -> list[dict]:
        """Maps numerical FAISS result indices and similarity distances to metadata dicts.

        Args:
            indices: Array of FAISS result indices.
            distances: Array of similarity distances.

        Returns:
            List of SearchResult dictionaries.  
        """
        results = []
        for idx, dist in zip(indices, distances):
            idx_int = int(idx)
            if idx_int in self.id_to_metadata:
                meta = self.id_to_metadata[idx_int]
                res = SearchResult(
                    id=idx_int,
                    image_id=meta.get("image_id", idx_int),
                    rel_path=meta.get("rel_path", ""),
                    abs_path=meta.get("abs_path", ""),
                    class_id=meta.get("class_id", -1),
                    class_name=meta.get("class_name", "unknown"),
                    score=round(float(dist), 4),
                )
                results.append(res.to_dict())
        return results
