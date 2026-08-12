import os
import json
import argparse
from pathlib import Path
import numpy as np
import faiss
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

from src.dataset import download_and_extract_cub200, parse_cub200_metadata, CUB200Dataset, get_transforms
from src.model import load_resnet_model
from src.clip_encoder import CLIPEncoder


def build_indices(
    data_dir: str = "data",
    output_dir: str = "indices",
    checkpoint_path: str = "checkpoints/best_resnet34_proxy_anchor.pt",
    batch_size: int = 64,
    num_workers: int = 2,
    device: str | None = None,
) -> tuple[str, str, str]:
    """Extracts features for all dataset images and builds offline FAISS search indices.

    Creates two separate indices:
    1. resnet_cub200.faiss: 128D vectors from fine-tuned ResNet34 metric learning encoder.
    2. clip_cub200.faiss: 512D vectors from frozen OpenCLIP ViT-B-32 encoder.
    Also creates id_to_metadata.json mapping sequential integer IDs (0..N) to image details.

    Args:
        data_dir: Path to CUB-200-2011 local dataset directory.
        output_dir: Destination directory for FAISS indices and metadata mapping.
        checkpoint_path: Model weights filepath for custom ResNet encoder.
        batch_size: Batch size for feature extraction.
        num_workers: PyTorch DataLoader background worker threads.
        device: Computation device ('cuda' or 'cpu').

    Returns:
        Tuple of (resnet_index_path, clip_index_path, metadata_path).
    """
    comp_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device for indexing: {comp_device}")

    # Download/parse dataset metadata
    cub_dir = download_and_extract_cub200(data_dir)
    records = parse_cub200_metadata(cub_dir)
    print(f"Found {len(records)} image records across CUB-200-2011.")

    # Extract 128D ResNet embeddings
    resnet_model = load_resnet_model(checkpoint_path, comp_device)
    _, eval_transform = get_transforms()
    dataset = CUB200Dataset(records, transform=eval_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    resnet_embeddings_list = []
    with torch.inference_mode():
        for images, _, _ in tqdm(dataloader, desc="Extracting ResNet (128D) embeddings"):
            images = images.to(comp_device)
            embeds = resnet_model(images)
            resnet_embeddings_list.append(embeds.cpu().numpy())

    resnet_vectors = np.concatenate(resnet_embeddings_list, axis=0).astype(np.float32)
    faiss.normalize_L2(resnet_vectors)

    # Extract 512D OpenCLIP embeddings
    print("Loading OpenCLIP model (ViT-B-32)...")
    clip_encoder = CLIPEncoder(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device=str(comp_device))
    clip_embeddings_list = []

    for i in tqdm(range(0, len(records), batch_size), desc="Extracting CLIP (512D) embeddings"):
        batch_records = records[i : i + batch_size]
        pil_imgs = [Image.open(r["abs_path"]).convert("RGB") for r in batch_records]
        clip_embeds = clip_encoder.encode_image(pil_imgs)
        clip_embeddings_list.append(clip_embeds.cpu().numpy())

    clip_vectors = np.concatenate(clip_embeddings_list, axis=0).astype(np.float32)
    faiss.normalize_L2(clip_vectors)

    # Build FAISS IndexFlatIP indices
    d_resnet = resnet_vectors.shape[1]
    resnet_index = faiss.IndexFlatIP(d_resnet)
    resnet_index.add(resnet_vectors)

    d_clip = clip_vectors.shape[1]
    clip_index = faiss.IndexFlatIP(d_clip)
    clip_index.add(clip_vectors)

    # Save indices and metadata JSON to output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    resnet_index_path = output_path / "resnet_cub200.faiss"
    clip_index_path = output_path / "clip_cub200.faiss"
    metadata_path = output_path / "id_to_metadata.json"

    faiss.write_index(resnet_index, str(resnet_index_path))
    faiss.write_index(clip_index, str(clip_index_path))

    id_to_metadata = {}
    for i, rec in enumerate(records):
        id_to_metadata[str(i)] = {
            "image_id": rec["image_id"],
            "rel_path": rec["rel_path"],
            "abs_path": rec["abs_path"],
            "class_id": rec["class_id"],
            "class_name": rec["class_name"],
        }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(id_to_metadata, f, indent=2)

    print(f"\nSuccessfully built and saved FAISS indices:")
    print(f"  - ResNet Index (128D, {resnet_index.ntotal} vectors): {resnet_index_path}")
    print(f"  - CLIP Index (512D, {clip_index.ntotal} vectors): {clip_index_path}")
    print(f"  - Metadata mapping ({len(id_to_metadata)} entries): {metadata_path}")

    return str(resnet_index_path), str(clip_index_path), str(metadata_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build offline FAISS indices for CUB-200 dataset.")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing CUB-200 dataset.")
    parser.add_argument("--output_dir", type=str, default="indices", help="Directory to save FAISS indices.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="checkpoints/best_resnet34_proxy_anchor.pt",
        help="Path to trained ResNet checkpoint.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for feature extraction.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader num_workers.")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on (cuda/cpu).")

    args = parser.parse_args()
    build_indices(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
