import os
import json
import numpy as np
import faiss
import torch
import pytest

from src.indexer import build_indices
from src.model import EmbeddingNet, load_resnet_model


def test_load_resnet_model(tmp_path):
    device = torch.device("cpu")

    # 1. Non-existent checkpoint fallback
    model = load_resnet_model("non_existent_checkpoint.pt", device)
    assert isinstance(model, EmbeddingNet)
    assert not model.training

    # 2. Valid checkpoint dict
    ckpt_path = os.path.join(tmp_path, "dummy_ckpt.pt")
    dummy_model = EmbeddingNet(embedding_dim=128, backbone_name="resnet34")
    torch.save({"model_state_dict": dummy_model.state_dict(), "epoch": 1}, ckpt_path)

    loaded_model = load_resnet_model(ckpt_path, device)
    assert isinstance(loaded_model, EmbeddingNet)
    assert not loaded_model.training


def test_faiss_index_properties_and_query(tmp_path):
    """Verifies creation, vector normalization, saving, loading, and querying of FAISS IndexFlatIP."""
    num_vectors = 50
    dim_resnet = 128
    dim_clip = 512

    # Generate random vectors and L2 normalize
    resnet_vecs = np.random.randn(num_vectors, dim_resnet).astype(np.float32)
    faiss.normalize_L2(resnet_vecs)

    clip_vecs = np.random.randn(num_vectors, dim_clip).astype(np.float32)
    faiss.normalize_L2(clip_vecs)

    # Build FAISS IP indices
    idx_resnet = faiss.IndexFlatIP(dim_resnet)
    idx_resnet.add(resnet_vecs)

    idx_clip = faiss.IndexFlatIP(dim_clip)
    idx_clip.add(clip_vecs)

    resnet_path = os.path.join(tmp_path, "resnet.faiss")
    clip_path = os.path.join(tmp_path, "clip.faiss")
    metadata_path = os.path.join(tmp_path, "metadata.json")

    faiss.write_index(idx_resnet, resnet_path)
    faiss.write_index(idx_clip, clip_path)

    metadata = {
        str(i): {"image_id": i + 1, "rel_path": f"img_{i}.jpg", "class_id": i % 5}
        for i in range(num_vectors)
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    # Reload indices from disk
    loaded_resnet_idx = faiss.read_index(resnet_path)
    loaded_clip_idx = faiss.read_index(clip_path)

    with open(metadata_path, "r") as f:
        loaded_metadata = json.load(f)

    # Number of vectors in FAISS index == number of items in id_to_metadata.json
    assert loaded_resnet_idx.ntotal == num_vectors
    assert loaded_clip_idx.ntotal == num_vectors
    assert len(loaded_metadata) == num_vectors

    # Test query accuracy (query vector equal to 1st indexed vector should yield score ~ 1.0 at top 1)
    query_vec = resnet_vecs[:1]  # shape [1, 128]
    scores, indices = loaded_resnet_idx.search(query_vec, k=3)

    assert indices[0][0] == 0
    assert pytest.approx(scores[0][0], abs=1e-5) == 1.0

@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists("data/CUB_200_2011/images"), reason="Requires local CUB-200 dataset")
def test_build_indices_integration(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = os.path.join(tmp_path, "indices")
    resnet_path, clip_path, meta_path = build_indices(
        data_dir="data",
        output_dir=output_dir,
        checkpoint_path="checkpoints/best_resnet34_proxy_anchor.pt",
        batch_size=64,
        num_workers=0,
        device=device,
    )

    assert os.path.exists(resnet_path)
    assert os.path.exists(clip_path)
    assert os.path.exists(meta_path)

    res_idx = faiss.read_index(resnet_path)
    clip_idx = faiss.read_index(clip_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    assert res_idx.ntotal == clip_idx.ntotal == len(meta) == 11788
