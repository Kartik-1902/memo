import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms

import clip
from timm import create_model
from torch.utils.data import DataLoader, TensorDataset


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# MEMO-MOD: Generalized model registry for the requested backbones.
MODEL_REGISTRY = {
    'resnet50': {
        'display_name': 'ResNet-50',
        'optimizer': 'sgd',
        'lr': 2.5e-4,
        'weight_decay': 0.0,
        'requires_linear_probe': True,
    },
    'vit_b16': {
        'display_name': 'ViT-B/16',
        'optimizer': 'adamw',
        'lr': 1e-5,
        'weight_decay': 0.01,
        'requires_linear_probe': True,
    },
    'clip_resnet50': {
        'display_name': 'CLIP ResNet-50',
        'optimizer': 'adamw',
        'lr': 5e-6,
        'weight_decay': 0.01,
        'requires_linear_probe': False,
    },
    'clip_vit_b16': {
        'display_name': 'CLIP ViT-B/16',
        'optimizer': 'adamw',
        'lr': 5e-6,
        'weight_decay': 0.01,
        'requires_linear_probe': False,
    },
}

CLIP_PROMPT_TEMPLATES = [
    'a satellite image of {}.',
    'an aerial photo of {}.',
    'a remote sensing image of {}.',
    'an overhead view of {}.',
]


def model_choices() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def get_model_config(model_name: str) -> Dict[str, object]:
    canonical_name = canonical_model_name(model_name)
    if canonical_name not in MODEL_REGISTRY:
        raise ValueError(f'Unsupported model: {model_name}')
    return MODEL_REGISTRY[canonical_name]


def canonical_model_name(name: str) -> str:
    return name.lower().replace('-', '_').replace('/', '_')


class FeatureBackboneClassifier(nn.Module):
    # MEMO-MOD: Wrap feature extractors with a dataset-specific linear classifier.
    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode_images(images))


class ClipZeroShotClassifier(nn.Module):
    # MEMO-MOD: CLIP wrapper that turns remote-sensing class names into zero-shot logits.
    def __init__(self, clip_model: nn.Module, class_names: List[str], device: torch.device):
        super().__init__()
        self.clip_model = clip_model
        self.device = device
        self.register_buffer('text_features', self._encode_text_features(class_names), persistent=True)

    @torch.no_grad()
    def _encode_text_features(self, class_names: List[str]) -> torch.Tensor:
        prompts: List[str] = []
        for class_name in class_names:
            prompts.extend(template.format(class_name) for template in CLIP_PROMPT_TEMPLATES)
        tokens = clip.tokenize(prompts).to(self.device)
        text_features = self.clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.view(len(class_names), len(CLIP_PROMPT_TEMPLATES), -1).mean(dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return self.clip_model.logit_scale.exp() * image_features @ self.text_features.t()


def build_model(
    model_name: str,
    class_names: List[str],
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, object]]:
    canonical_name = canonical_model_name(model_name)
    model_config = get_model_config(canonical_name)

    if canonical_name == 'resnet50':
        backbone = tv_models.resnet50(pretrained=True)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        model = FeatureBackboneClassifier(backbone, feature_dim, len(class_names))
        eval_transform = build_imagenet_transform()
    elif canonical_name == 'vit_b16':
        backbone = create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        feature_dim = getattr(backbone, 'num_features', 768)
        model = FeatureBackboneClassifier(backbone, feature_dim, len(class_names))
        eval_transform = build_imagenet_transform()
    elif canonical_name == 'clip_resnet50':
        clip_model, eval_transform = clip.load('RN50', device=device, jit=False)
        model = ClipZeroShotClassifier(clip_model, class_names, device)
    elif canonical_name == 'clip_vit_b16':
        clip_model, eval_transform = clip.load('ViT-B/16', device=device, jit=False)
        model = ClipZeroShotClassifier(clip_model, class_names, device)
    else:
        raise ValueError(f'Unsupported model: {model_name}')

    model = model.to(device)
    return model, {
        'eval_transform': eval_transform,
        'requires_linear_probe': model_config['requires_linear_probe'],
        'optimizer': model_config['optimizer'],
        'lr': model_config['lr'],
        'weight_decay': model_config['weight_decay'],
        'display_name': model_config['display_name'],
    }


def build_imagenet_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_task_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> Dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return checkpoint


def save_task_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    model_name: str,
    dataset_name: str,
    class_names: List[str],
    task_type: str,
    metric_name: str,
) -> None:
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(
        {
            'state_dict': model.state_dict(),
            'model_name': model_name,
            'dataset_name': dataset_name,
            'class_names': class_names,
            'task_type': task_type,
            'metric_name': metric_name,
        },
        checkpoint_path,
    )


def train_linear_probe(
    model: FeatureBackboneClassifier,
    train_loader: DataLoader,
    device: torch.device,
    task_type: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    feature_batch_size: int,
) -> None:
    # MEMO-MOD: Freeze the pretrained backbone, fit a dataset-specific head once,
    # and cache it so later MEMO runs can reuse the same checkpoint.
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    train_features, train_labels = extract_features(model, train_loader, device)
    train_dataset = TensorDataset(train_features, train_labels)
    feature_loader = DataLoader(train_dataset, batch_size=feature_batch_size, shuffle=True)

    classifier = nn.Linear(model.classifier.in_features, model.classifier.out_features).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss() if task_type == 'single_label' else nn.BCEWithLogitsLoss()

    classifier.train()
    for _ in range(epochs):
        for batch_features, batch_labels in feature_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = classifier(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

    model.classifier.load_state_dict(classifier.state_dict())
    for parameter in model.backbone.parameters():
        parameter.requires_grad = True


@torch.no_grad()
def extract_features(
    model: FeatureBackboneClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    features = []
    labels = []
    for images, targets in loader:
        images = images.to(device)
        encoded = model.encode_images(images).cpu().float()
        features.append(encoded)
        if isinstance(targets, torch.Tensor):
            labels.append(targets.cpu())
        else:
            labels.append(torch.as_tensor(targets))
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)