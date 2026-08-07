import os
import tarfile
import urllib.request
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pytorch_metric_learning.samplers import MPerClassSampler

CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"

class CUB200Dataset(Dataset):
    """PyTorch Dataset for CUB-200-2011 image retrieval."""

    def __init__(self, records: list[dict], transform=None):
        self.records = records
        self.transform = transform

        # Map class_ids to contiguous 0..C-1 targets for metric learning sampler compatibility
        unique_classes = sorted(list(set(r["class_id"] for r in records)))
        self.class_to_target = {cls_id: i for i, cls_id in enumerate(unique_classes)}

        self.targets = [self.class_to_target[r["class_id"]] for r in records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        rec = self.records[idx]
        image = Image.open(rec["abs_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        target = self.targets[idx]
        return image, target, rec["abs_path"]


def download_and_extract_cub200(data_dir: str = "data") -> str:
    """Downloads and extracts CUB-200-2011 dataset if not already present.

    Returns:
        Path to extracted CUB_200_2011 directory.
    """
    cub_dir = os.path.join(data_dir, "CUB_200_2011")
    images_dir = os.path.join(cub_dir, "images")

    if os.path.exists(images_dir):
        return cub_dir

    os.makedirs(data_dir, exist_ok=True)
    tgz_path = os.path.join(data_dir, "CUB_200_2011.tgz")

    if not os.path.exists(tgz_path):
        print(f"Downloading CUB-200-2011 dataset from {CUB_URL}...")
        urllib.request.urlretrieve(CUB_URL, tgz_path)
        print("Download complete.")

    print(f"Extracting {tgz_path}...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print("Extraction complete.")

    return cub_dir


def parse_cub200_metadata(cub_dir: str) -> list[dict]:
    """Parses CUB-200-2011 metadata files into a list of image records.

    Returns:
        List of dicts with keys: 'image_id', 'rel_path', 'abs_path', 'class_id', 'class_name'.
    """
    images_file = os.path.join(cub_dir, "images.txt")
    labels_file = os.path.join(cub_dir, "image_class_labels.txt")
    classes_file = os.path.join(cub_dir, "classes.txt")

    if not (os.path.exists(images_file) and os.path.exists(labels_file)):
        raise FileNotFoundError(f"Missing metadata files in {cub_dir}")

    # Load classes: class_id -> class_name
    # Example structure: {1: 'black_throated_sparrow', 2: 'california_towhee', ...}
    class_names = {}
    if os.path.exists(classes_file):
        with open(classes_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    class_names[int(parts[0])] = parts[1]

    # Load image paths: image_id -> rel_path
    # Example structure: {1: '001.Black_footed_Albatross/ABal_0001_45.jpg', 2: '002.Anna_hummingbird/ABHum_0001_78.jpg', ...}
    images = {}
    with open(images_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                images[int(parts[0])] = parts[1]

    # Load image labels: image_id -> class_id
    # Example structure: {1: 285, 2: 325, 3: 285, ...}
    records = []
    with open(labels_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_id = int(parts[0])
                cls_id = int(parts[1])
                rel_path = images[img_id]
                abs_path = os.path.join(cub_dir, "images", rel_path)
                records.append({
                    "image_id": img_id,
                    "rel_path": rel_path,
                    "abs_path": abs_path,
                    "class_id": cls_id,
                    "class_name": class_names.get(cls_id, f"class_{cls_id}")
                })

                """
                Example structure:
                {
                    "image_id": 1, 
                    "rel_path": "001.Black_footed_Albatross/ABal_0001_45.jpg", 
                    "abs_path": "data/CUB_200_2011/images/001.Black_footed_Albatross/ABal_0001_45.jpg", 
                    "class_id": 285, 
                    "class_name": "black_throated_sparrow"
                }
                """
    return records


# Default ImageNet normalization & spatial transformation constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RESIZE_SIZE = (256, 256)
CROP_SIZE = (224, 224)
COLOR_JITTER = 0.2

def get_transforms():
    """Returns train and evaluation PyTorch image transformations."""
    train_transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.RandomCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=COLOR_JITTER, contrast=COLOR_JITTER, saturation=COLOR_JITTER),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def create_disjoint_splits(records: list[dict], seed: int = 42) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Splits records into Seen classes (1..100) and Unseen classes (101..200).

    - Seen classes (1..100): Split per class into train (70%), val (10%), seen_test (20%).
    - Unseen classes (101..200): All assigned to unseen_test.

    This is the standard split for the CUB-200-2011 dataset for zero-shot learning.

    Returns:
        (train_records, val_records, seen_test_records, unseen_test_records)
    """
    g = torch.Generator().manual_seed(seed)


    # Sort data to seen classes (1..100) and unseen classes (101..200)
    seen_records_by_class = {}
    unseen_records = []

    for r in records:
        cls_id = r["class_id"]
        if 1 <= cls_id <= 100:
            # Each seen class is a key in the dictionary and its value is a list of records for that class
            seen_records_by_class.setdefault(cls_id, []).append(r)
        elif 101 <= cls_id <= 200:
            unseen_records.append(r)

    # Split seen classes into train (70%), val (10%), seen_test (20%)
    train_records = []
    val_records = []
    seen_test_records = []

    for cls_id, cls_items in seen_records_by_class.items():
        # Deterministically shuffle items for this class
        perm = torch.randperm(len(cls_items), generator=g).tolist()
        shuffled = [cls_items[i] for i in perm]

        n = len(shuffled)
        n_train = int(n * 0.70)
        n_val = int(n * 0.10)

        train_records.extend(shuffled[:n_train])
        val_records.extend(shuffled[n_train:n_train + n_val])
        seen_test_records.extend(shuffled[n_train + n_val:])

    return train_records, val_records, seen_test_records, unseen_records


def get_cub200_dataloaders(
    data_dir: str = "data",
    batch_size: int = 64,
    samples_per_class: int = 4,
    use_m_per_class_sampler: bool = True,
    num_workers: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Main pipeline entrypoint to get DataLoaders for all splits.

    Args:
        data_dir: Path to directory storing CUB_200_2011 dataset.
        batch_size: Batch size for DataLoader instances.
        samples_per_class: Number of samples per class when using MPerClassSampler.
        use_m_per_class_sampler: If True, uses MPerClassSampler for Triplet Loss.
            If False, uses standard RandomSampler (shuffle=True) for Proxy-based losses.
        num_workers: Number of worker processes for data loading.
        seed: Random seed for reproducible class splitting.

    Returns:
        (train_loader, val_loader, seen_test_loader, unseen_test_loader)
    """
    cub_dir = download_and_extract_cub200(data_dir)
    all_records = parse_cub200_metadata(cub_dir)

    train_recs, val_recs, seen_test_recs, unseen_recs = create_disjoint_splits(all_records, seed=seed)

    train_tf, eval_tf = get_transforms()

    train_dataset = CUB200Dataset(train_recs, transform=train_tf)
    val_dataset = CUB200Dataset(val_recs, transform=eval_tf)
    seen_test_dataset = CUB200Dataset(seen_test_recs, transform=eval_tf)
    unseen_test_dataset = CUB200Dataset(unseen_recs, transform=eval_tf)

    if use_m_per_class_sampler:
        sampler = MPerClassSampler(
            labels=train_dataset.targets,
            m=samples_per_class,
            batch_size=batch_size,
            length_before_new_iter=len(train_dataset),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    seen_test_loader = DataLoader(
        seen_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    unseen_test_loader = DataLoader(
        unseen_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, seen_test_loader, unseen_test_loader
