import pytest
from src.dataset import parse_cub200_metadata, create_disjoint_splits


def test_disjoint_splits_class_and_file_isolation():
    """Verify zero-shot split rules:
    1. Seen classes (1..100) vs Unseen classes (101..200) have ZERO class overlap.
    2. Train, Val, and Seen_test have ZERO file path overlap.
    """
    dummy_records = []
    # Create 100 seen classes (ids 1..100) with 10 images each
    for cls_id in range(1, 101):
        for img_idx in range(10):
            dummy_records.append({
                "image_id": cls_id * 100 + img_idx,
                "rel_path": f"seen_{cls_id}_{img_idx}.jpg",
                "abs_path": f"/fake/seen_{cls_id}_{img_idx}.jpg",
                "class_id": cls_id,
                "class_name": f"bird_seen_{cls_id}"
            })

    # Create 100 unseen classes (ids 101..200) with 10 images each
    for cls_id in range(101, 201):
        for img_idx in range(10):
            dummy_records.append({
                "image_id": cls_id * 100 + img_idx,
                "rel_path": f"unseen_{cls_id}_{img_idx}.jpg",
                "abs_path": f"/fake/unseen_{cls_id}_{img_idx}.jpg",
                "class_id": cls_id,
                "class_name": f"bird_unseen_{cls_id}"
            })

    train_recs, val_recs, seen_test_recs, unseen_recs = create_disjoint_splits(dummy_records, seed=42)

    seen_classes = set(r["class_id"] for r in train_recs + val_recs + seen_test_recs)
    unseen_classes = set(r["class_id"] for r in unseen_recs)

    # 1. Zero class overlap between seen and unseen
    assert seen_classes.isdisjoint(unseen_classes)
    assert seen_classes == set(range(1, 101))
    assert unseen_classes == set(range(101, 201))

    # 2. Zero file path overlap between train, val, and seen_test
    train_paths = set(r["abs_path"] for r in train_recs)
    val_paths = set(r["abs_path"] for r in val_recs)
    seen_test_paths = set(r["abs_path"] for r in seen_test_recs)

    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(seen_test_paths)
    assert val_paths.isdisjoint(seen_test_paths)


def test_get_cub200_dataloaders_sampler_option(mocker=None):
    """Verify get_cub200_dataloaders toggles between MPerClassSampler and standard sampler."""
    from unittest.mock import patch
    from src.dataset import get_cub200_dataloaders
    from pytorch_metric_learning.samplers import MPerClassSampler

    dummy_records = [
        {
            "image_id": i,
            "rel_path": f"img_{i}.jpg",
            "abs_path": f"/fake/img_{i}.jpg",
            "class_id": (i % 100) + 1,
            "class_name": f"bird_{(i % 100) + 1}"
        }
        for i in range(200)
    ]

    with patch("src.dataset.download_and_extract_cub200", return_value="/fake/cub"), \
         patch("src.dataset.parse_cub200_metadata", return_value=dummy_records), \
         patch("src.dataset.CUB200Dataset.__getitem__", return_value=(None, 0, "fake")):

        # 1. Test with MPerClassSampler (Triplet mode)
        train_l, _, _, _ = get_cub200_dataloaders(use_m_per_class_sampler=True, batch_size=16)
        assert isinstance(train_l.sampler, MPerClassSampler)

        # 2. Test with standard RandomSampler (Proxy Anchor mode)
        train_l_no_m, _, _, _ = get_cub200_dataloaders(use_m_per_class_sampler=False, batch_size=16)
        assert not isinstance(train_l_no_m.sampler, MPerClassSampler)

