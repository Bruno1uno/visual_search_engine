import pytest
from pathlib import Path
from PIL import Image
import numpy as np
import torch

from src.inference import VisualSearchEngine, SearchResult

CHECKPOINT_PATH = "checkpoints/best_resnet34_proxy_anchor.pt"
RESNET_FAISS_PATH = "indices/resnet_cub200.faiss"
CLIP_FAISS_PATH = "indices/clip_cub200.faiss"
METADATA_PATH = "indices/id_to_metadata.json"


def test_missing_files_error_handling(tmp_path):
    """Test that VisualSearchEngine raises FileNotFoundError on missing files."""
    dummy_file = tmp_path / "non_existent.faiss"
    with pytest.raises(FileNotFoundError):
        VisualSearchEngine(
            resnet_checkpoint_path=dummy_file,
            resnet_faiss_path=dummy_file,
            clip_faiss_path=dummy_file,
            metadata_path=dummy_file,
        )


@pytest.mark.skipif(
    not Path(CHECKPOINT_PATH).exists() or not Path(RESNET_FAISS_PATH).exists(),
    reason="Real checkpoints/indices not available in environment",
)
def test_visual_search_engine_search_by_image():
    """Integration test for search_by_image using real model and FAISS indices."""
    engine = VisualSearchEngine(
        resnet_checkpoint_path=CHECKPOINT_PATH,
        resnet_faiss_path=RESNET_FAISS_PATH,
        clip_faiss_path=CLIP_FAISS_PATH,
        metadata_path=METADATA_PATH,
    )

    # Synthetic RGB image
    img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

    # Test ResNet engine
    results_resnet = engine.search_by_image(img, engine_type="resnet", top_k=4)
    assert len(results_resnet) == 4
    for item in results_resnet:
        assert "id" in item
        assert "class_name" in item
        assert "score" in item
        assert -1.0 <= item["score"] <= 1.05

    # Test CLIP engine
    results_clip = engine.search_by_image(img, engine_type="clip", top_k=4)
    assert len(results_clip) == 4


@pytest.mark.skipif(
    not Path(CHECKPOINT_PATH).exists() or not Path(RESNET_FAISS_PATH).exists(),
    reason="Real checkpoints/indices not available in environment",
)
def test_visual_search_engine_search_by_text():
    """Integration test for search_by_text using CLIP text encoder."""
    engine = VisualSearchEngine(
        resnet_checkpoint_path=CHECKPOINT_PATH,
        resnet_faiss_path=RESNET_FAISS_PATH,
        clip_faiss_path=CLIP_FAISS_PATH,
        metadata_path=METADATA_PATH,
    )

    results = engine.search_by_text("a yellow bird with black wings", top_k=5)
    assert len(results) == 5
    for item in results:
        assert "rel_path" in item
        assert "class_name" in item
        assert isinstance(item["score"], float)

    # Empty text query check
    empty_results = engine.search_by_text("", top_k=5)
    assert empty_results == []
