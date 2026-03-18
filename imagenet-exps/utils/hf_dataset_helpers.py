import io
import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

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


# MEMO-MOD: Registry for the Hugging Face remote-sensing datasets requested by the user.
RSICD_FILENAME_ALIASES = {
    'airport': 'airport',
    'bareland': 'bare land',
    'baseballfield': 'baseball field',
    'beach': 'beach',
    'bridge': 'bridge',
    'center': 'center',
    'church': 'church',
    'commercial': 'commercial area',
    'denseresidential': 'dense residential',
    'desert': 'desert',
    'farmland': 'farmland',
    'forest': 'forest',
    'industrial': 'industrial area',
    'meadow': 'meadow',
    'mediumresidential': 'medium residential',
    'mountain': 'mountain',
    'park': 'park',
    'parking': 'parking lot',
    'playground': 'playground',
    'pond': 'pond',
    'port': 'port',
    'railwaystation': 'railway station',
    'resort': 'resort',
    'river': 'river',
    'school': 'school',
    'sparseresidential': 'sparse residential',
    'square': 'square',
    'stadium': 'stadium',
    'storagetanks': 'storage tanks',
    'viaduct': 'viaduct',
}

DATASET_REGISTRY = {
    'patternnet': {
        'hf_id': 'blanchon/PatternNet',
        'train_split': 'train',
        'eval_split': None,
        'task_type': 'single_label',
        'metric_name': 'accuracy',
        'synthetic_eval_ratio': 0.2,
    },
    'rsicd': {
        'hf_id': 'arampacha/rsicd',
        'train_split': 'train',
        'eval_split': 'test',
        'task_type': 'single_label',
        'metric_name': 'accuracy',
        'synthetic_eval_ratio': None,
    },
    'eurosat': {
        'hf_id': 'tanganke/eurosat',
        'train_split': 'train',
        'eval_split': 'test',
        'task_type': 'single_label',
        'metric_name': 'accuracy',
        'synthetic_eval_ratio': None,
    },
    'resisc45': {
        'hf_id': 'timm/resisc45',
        'train_split': 'train',
        'eval_split': 'test',
        'task_type': 'single_label',
        'metric_name': 'accuracy',
        'synthetic_eval_ratio': None,
    },
    'mlrsnet': {
        'hf_id': 'jonathan-roberts1/MLRSNet',
        'train_split': 'train',
        'eval_split': None,
        'task_type': 'multi_label',
        'metric_name': 'mAP',
        'synthetic_eval_ratio': 0.2,
    },
}


def canonical_dataset_name(name: str) -> str:
    return name.lower().replace('-', '').replace('_', '')


def dataset_choices() -> List[str]:
    return list(DATASET_REGISTRY.keys())


def get_dataset_config(dataset_name: str) -> Dict[str, object]:
    canonical_name = canonical_dataset_name(dataset_name)
    if canonical_name not in DATASET_REGISTRY:
        raise ValueError(f'Unsupported dataset: {dataset_name}')
    return DATASET_REGISTRY[canonical_name]


@dataclass
class DatasetBundle:
    train_dataset: 'RemoteSensingDataset'
    eval_dataset: 'RemoteSensingDataset'
    class_names: List[str]
    task_type: str
    metric_name: str
    train_split_name: str
    eval_split_name: str
    hf_id: str


class RemoteSensingDataset(Dataset):
    # MEMO-MOD: Torch Dataset wrapper around Hugging Face datasets so the existing
    # DataLoader-based evaluation flow can be reused without manual downloads.
    def __init__(
        self,
        hf_split,
        class_names: Sequence[str],
        task_type: str,
        transform: Optional[Callable] = None,
        indices: Optional[Sequence[int]] = None,
        label_resolver: Optional[Callable[[dict], object]] = None,
    ):
        self.hf_split = hf_split
        self.class_names = list(class_names)
        self.task_type = task_type
        self.transform = transform
        self.indices = list(indices) if indices is not None else None
        self.label_resolver = label_resolver or (lambda row: row['label'])
        self.num_classes = len(self.class_names)

    def clone_with_transform(self, transform: Optional[Callable]) -> 'RemoteSensingDataset':
        return RemoteSensingDataset(
            hf_split=self.hf_split,
            class_names=self.class_names,
            task_type=self.task_type,
            transform=transform,
            indices=self.indices,
            label_resolver=self.label_resolver,
        )

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.hf_split)

    def __getitem__(self, index: int):
        row_index = self.indices[index] if self.indices is not None else index
        row = self.hf_split[row_index]
        image = _resolve_image(row['image'])
        label = self.label_resolver(row)

        if self.transform is not None:
            image = self.transform(image)

        if self.task_type == 'multi_label':
            encoded = torch.zeros(self.num_classes, dtype=torch.float32)
            for class_index in label:
                encoded[int(class_index)] = 1.0
            label = encoded
        else:
            label = int(label)

        return image, label


def load_remote_sensing_dataset(
    dataset_name: str,
    split_seed: int,
    hf_cache_dir: Optional[str] = None,
    eval_split_override: Optional[str] = None,
) -> DatasetBundle:
    config = get_dataset_config(dataset_name)
    dataset_dict = load_dataset(config['hf_id'], cache_dir=hf_cache_dir)

    class_names, label_resolver = _resolve_class_metadata(canonical_dataset_name(dataset_name), dataset_dict)
    train_split_name = config['train_split']
    requested_eval_split = eval_split_override or config['eval_split']

    if requested_eval_split is not None:
        if requested_eval_split not in dataset_dict:
            available = ', '.join(dataset_dict.keys())
            raise ValueError(f'Split {requested_eval_split} not found for {dataset_name}. Available splits: {available}')
        train_dataset = RemoteSensingDataset(
            dataset_dict[train_split_name],
            class_names=class_names,
            task_type=config['task_type'],
            transform=None,
            label_resolver=label_resolver,
        )
        eval_dataset = RemoteSensingDataset(
            dataset_dict[requested_eval_split],
            class_names=class_names,
            task_type=config['task_type'],
            transform=None,
            label_resolver=label_resolver,
        )
        eval_split_name = requested_eval_split
    else:
        train_dataset, eval_dataset = _build_synthetic_split(
            dataset_name=canonical_dataset_name(dataset_name),
            hf_split=dataset_dict[train_split_name],
            class_names=class_names,
            task_type=config['task_type'],
            label_resolver=label_resolver,
            eval_ratio=config['synthetic_eval_ratio'],
            split_seed=split_seed,
        )
        eval_split_name = 'synthetic-test'

    return DatasetBundle(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        class_names=class_names,
        task_type=config['task_type'],
        metric_name=config['metric_name'],
        train_split_name=train_split_name,
        eval_split_name=eval_split_name,
        hf_id=config['hf_id'],
    )


def _resolve_class_metadata(dataset_name: str, dataset_dict) -> Tuple[List[str], Callable[[dict], object]]:
    sample_split = next(iter(dataset_dict.values()))
    features = sample_split.features

    if dataset_name == 'rsicd':
        class_names = [RSICD_FILENAME_ALIASES[token] for token in RSICD_FILENAME_ALIASES]
        label_to_index = {token: index for index, token in enumerate(RSICD_FILENAME_ALIASES)}

        def resolve_label(row: dict) -> int:
            filename = row['filename']
            prefix = re.sub(r'_\d+\.jpg$', '', filename.split('/')[-1])
            if prefix not in label_to_index:
                raise KeyError(f'Unknown RSICD filename prefix: {prefix}')
            return label_to_index[prefix]

        return class_names, resolve_label

    label_feature = features['label']
    if dataset_name == 'mlrsnet':
        class_names = list(label_feature.feature.names)

        def resolve_label(row: dict) -> List[int]:
            return [int(label) for label in row['label']]

        return class_names, resolve_label

    class_names = list(label_feature.names)

    def resolve_label(row: dict) -> int:
        return int(row['label'])

    return class_names, resolve_label


def _build_synthetic_split(
    dataset_name: str,
    hf_split,
    class_names: Sequence[str],
    task_type: str,
    label_resolver: Callable[[dict], object],
    eval_ratio: float,
    split_seed: int,
) -> Tuple[RemoteSensingDataset, RemoteSensingDataset]:
    if task_type == 'single_label':
        labels = [int(label_resolver(hf_split[index])) for index in range(len(hf_split))]
        train_indices, eval_indices = _stratified_indices(labels, eval_ratio, split_seed)
    else:
        all_indices = list(range(len(hf_split)))
        rng = random.Random(split_seed)
        rng.shuffle(all_indices)
        eval_size = max(1, int(round(len(all_indices) * eval_ratio)))
        eval_indices = sorted(all_indices[:eval_size])
        train_indices = sorted(all_indices[eval_size:])

    train_dataset = RemoteSensingDataset(
        hf_split=hf_split,
        class_names=class_names,
        task_type=task_type,
        transform=None,
        indices=train_indices,
        label_resolver=label_resolver,
    )
    eval_dataset = RemoteSensingDataset(
        hf_split=hf_split,
        class_names=class_names,
        task_type=task_type,
        transform=None,
        indices=eval_indices,
        label_resolver=label_resolver,
    )
    return train_dataset, eval_dataset


def _stratified_indices(labels: Sequence[int], eval_ratio: float, split_seed: int) -> Tuple[List[int], List[int]]:
    grouped_indices: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        grouped_indices.setdefault(label, []).append(index)

    train_indices: List[int] = []
    eval_indices: List[int] = []
    for label, indices in grouped_indices.items():
        rng = random.Random(split_seed + label)
        shuffled = list(indices)
        rng.shuffle(shuffled)
        eval_count = max(1, int(round(len(shuffled) * eval_ratio)))
        eval_indices.extend(shuffled[:eval_count])
        train_indices.extend(shuffled[eval_count:])

    return sorted(train_indices), sorted(eval_indices)


def _resolve_image(image_value) -> Image.Image:
    if isinstance(image_value, Image.Image):
        return image_value.convert('RGB')

    if isinstance(image_value, dict):
        if image_value.get('bytes') is not None:
            return Image.open(io.BytesIO(image_value['bytes'])).convert('RGB')
        if image_value.get('path') is not None:
            return Image.open(image_value['path']).convert('RGB')
        if image_value.get('array') is not None:
            return Image.fromarray(np.asarray(image_value['array'])).convert('RGB')

    raise TypeError(f'Unsupported image payload type: {type(image_value)}')