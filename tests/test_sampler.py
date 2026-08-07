from collections import Counter
from pytorch_metric_learning.samplers import MPerClassSampler


def test_mperclass_sampler_batch_structure():
    """Verify MPerClassSampler produces batches with M instances per class."""
    # 10 classes, 20 samples each = 200 samples
    targets = []
    for cls_id in range(10):
        targets.extend([cls_id] * 20)

    m = 4
    batch_size = 16

    sampler = MPerClassSampler(
        labels=targets,
        m=m,
        batch_size=batch_size,
        length_before_new_iter=len(targets)
    )

    indices = list(sampler)
    assert len(indices) > 0

    # Group into batches of size 16 and check class counts
    for i in range(0, len(indices) - batch_size, batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_labels = [targets[idx] for idx in batch_indices]

        counts = Counter(batch_labels)
        # Every class present in the batch must have exactly m=4 instances
        for cls_id, count in counts.items():
            assert count == m, f"Class {cls_id} has count {count}, expected {m}"
