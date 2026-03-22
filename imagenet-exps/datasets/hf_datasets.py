import os
import re
import sys
from typing import Callable, List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms


# MEMO-MODIFICATION: ensure we load HF datasets even though this repo has a datasets/ folder.
def _import_hf_datasets_module():
    local_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    removed_paths = []
    if local_root in sys.path:
        sys.path.remove(local_root)
        removed_paths.append(local_root)

    if "datasets" in sys.modules and hasattr(sys.modules["datasets"], "__file__"):
        if "imagenet-exps" in sys.modules["datasets"].__file__:
            del sys.modules["datasets"]

    try:
        import datasets as hf_datasets
    except Exception as exc:
        raise ImportError(
            "Failed to import HuggingFace 'datasets'. Reinstall with "
            "pip install -U datasets==2.14.7 pyarrow==12.0.1"
        ) from exc
    finally:
        for path in removed_paths:
            sys.path.insert(0, path)

    return hf_datasets


_HF_DATASETS_MODULE = _import_hf_datasets_module()
load_dataset = _HF_DATASETS_MODULE.load_dataset
ClassLabel = _HF_DATASETS_MODULE.ClassLabel


# MEMO-MODIFICATION: HF dataset support with model-specific transforms.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


_HF_DATASETS = {
    "patternnet": {
        "hf_path": "blanchon/PatternNet",
        "preferred_split": "train",
        "image_keys": ["image", "img", "images"],
        "label_keys": ["label", "labels", "class", "category"],
    },
    "rsicd": {
        "hf_path": "arampacha/rsicd",
        "preferred_split": "test",
        "image_keys": ["image", "img", "images"],
        "label_keys": ["label", "labels", "class", "category"],
    },
    "resisc45": {
        "hf_path": "timm/resisc45",
        "preferred_split": "test",
        "image_keys": ["image", "img", "images"],
        "label_keys": ["label", "labels", "class", "category"],
    },
    "mlrsnet": {
        "hf_path": "jonathan-roberts1/MLRSNet",
        "preferred_split": "train",
        "image_keys": ["image", "img", "images"],
        "label_keys": ["label", "labels", "class", "category"],
    },
}

RSICD_FILENAME_ALIASES = {
    "airport": "airport",
    "bareland": "bare land",
    "baseballfield": "baseball field",
    "beach": "beach",
    "bridge": "bridge",
    "center": "center",
    "church": "church",
    "commercial": "commercial area",
    "denseresidential": "dense residential",
    "desert": "desert",
    "farmland": "farmland",
    "forest": "forest",
    "industrial": "industrial area",
    "meadow": "meadow",
    "mediumresidential": "medium residential",
    "mountain": "mountain",
    "park": "park",
    "parking": "parking lot",
    "playground": "playground",
    "pond": "pond",
    "port": "port",
    "railwaystation": "railway station",
    "resort": "resort",
    "river": "river",
    "school": "school",
    "sparseresidential": "sparse residential",
    "square": "square",
    "stadium": "stadium",
    "storagetanks": "storage tanks",
    "viaduct": "viaduct",
}


def _resolve_split(dataset_dict, preferred_split: Optional[str]) -> str:
    if preferred_split and preferred_split in dataset_dict:
        return preferred_split
    for candidate in ("test", "validation", "train"):
        if candidate in dataset_dict:
            return candidate
    raise ValueError("No usable split found in HuggingFace dataset.")


def _pick_column(column_names: List[str], candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in column_names:
            return name
    return None


def _infer_columns(dataset_name: str, dataset) -> Tuple[str, str]:
    config = _HF_DATASETS[dataset_name]
    column_names = list(dataset.column_names)

    image_key = _pick_column(column_names, config["image_keys"])
    label_key = _pick_column(column_names, config["label_keys"])

    if image_key is None:
        for key, feature in dataset.features.items():
            if getattr(feature, "_type", None) == "Image":
                image_key = key
                break
    if label_key is None:
        for key, feature in dataset.features.items():
            if isinstance(feature, ClassLabel):
                label_key = key
                break

    if image_key is None or label_key is None:
        raise ValueError(
            f"Unable to infer image/label columns for dataset '{dataset_name}'."
        )
    return image_key, label_key


def _infer_image_column(dataset_name: str, dataset) -> str:
    config = _HF_DATASETS[dataset_name]
    column_names = list(dataset.column_names)

    image_key = _pick_column(column_names, config["image_keys"])
    if image_key is None:
        for key, feature in dataset.features.items():
            if getattr(feature, "_type", None) == "Image":
                image_key = key
                break
    if image_key is None:
        raise ValueError(
            f"Unable to infer image column for dataset '{dataset_name}'."
        )
    return image_key


def _get_class_names(dataset, label_key: str) -> List[str]:
    feature = dataset.features[label_key]
    if isinstance(feature, ClassLabel):
        return list(feature.names)
    unique_labels = sorted(set(dataset[label_key]))
    return [str(label) for label in unique_labels]


def _build_resnet_vit_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    tr_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    te_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return tr_transform, te_transform


def _build_clip_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    tr_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )
    te_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )
    return tr_transform, te_transform


def build_clip_prompts(class_names: List[str]) -> List[str]:
    # MEMO-MODIFICATION: CLIP prompts are built per dataset.
    return [f"a satellite image of a {name}" for name in class_names]


class HFDatasetWrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset,
        image_key: str,
        label_key: str,
        label_resolver: Optional[Callable[[dict], int]] = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.dataset = dataset
        self.image_key = image_key
        self.label_key = label_key
        self.label_resolver = label_resolver
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        row = self.dataset[idx]
        image = row[self.image_key]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.label_resolver is not None:
            label = self.label_resolver(row)
        else:
            label = row[self.label_key]
        return image, int(label)


def _resolve_rsicd_label(row: dict) -> int:
    filename = None
    for key in ("filename", "file_name", "file"):
        if key in row:
            filename = row[key]
            break
    if filename is None:
        raise KeyError("RSICD sample is missing a filename field.")

    basename = os.path.basename(filename)
    stem = re.sub(r"\.[^.]+$", "", basename)
    prefix = re.sub(r"_\d+$", "", stem).lower()
    label_to_index = {token: index for index, token in enumerate(RSICD_FILENAME_ALIASES)}
    if prefix not in label_to_index:
        raise KeyError(f"Unknown RSICD filename prefix: {prefix}")
    return label_to_index[prefix]


def load_hf_dataset(
    dataset_name: str,
    split: Optional[str],
    model_name: str,
    use_transforms: bool = True,
    batch_size: int = 256,
    workers: int = 8,
) -> Tuple[torch.utils.data.Dataset, Optional[torch.utils.data.DataLoader], List[str], transforms.Compose, transforms.Compose, List[str]]:
    if dataset_name not in _HF_DATASETS:
        raise ValueError(f"Unsupported HF dataset: {dataset_name}")

    config = _HF_DATASETS[dataset_name]
    dataset_dict = load_dataset(config["hf_path"])
    split_name = _resolve_split(dataset_dict, split or config["preferred_split"])
    dataset = dataset_dict[split_name]

    label_resolver = None
    if dataset_name == "rsicd":
        image_key = _infer_image_column(dataset_name, dataset)
        label_key = "filename"
        class_names = [RSICD_FILENAME_ALIASES[token] for token in RSICD_FILENAME_ALIASES]
        label_resolver = _resolve_rsicd_label
    else:
        image_key, label_key = _infer_columns(dataset_name, dataset)
        class_names = _get_class_names(dataset, label_key)

    if model_name in ("clip_resnet50", "clip_vitb16"):
        tr_transform, te_transform = _build_clip_transforms()
    else:
        tr_transform, te_transform = _build_resnet_vit_transforms()

    dataset_transform = te_transform if use_transforms else None
    teset = HFDatasetWrapper(
        dataset,
        image_key,
        label_key,
        label_resolver=label_resolver,
        transform=dataset_transform,
    )

    teloader = None
    if use_transforms:
        teloader = torch.utils.data.DataLoader(
            teset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
        )

    clip_prompts = build_clip_prompts(class_names)
    return teset, teloader, class_names, tr_transform, te_transform, clip_prompts
