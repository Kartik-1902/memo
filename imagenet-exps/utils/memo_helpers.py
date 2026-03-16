import contextlib
import random
from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from utils.third_party import augmentations


# MEMO-MOD: Preserve the original BatchNorm forward so BN-prior adaptation can be scoped safely.
ORIGINAL_BATCHNORM_FORWARD = nn.BatchNorm2d.forward


def build_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if optimizer_name == 'sgd':
        return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == 'adamw':
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f'Unsupported optimizer: {optimizer_name}')


def build_memo_augmentation(eval_transform: Callable) -> Callable:
    # MEMO-MOD: Keep the existing AugMix-style MEMO augmentation pattern, but apply
    # the active model preprocessing afterwards so CLIP and ImageNet models both work.
    preaugment = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ])

    def memo_transform(image):
        image = image.convert('RGB')
        x_orig = preaugment(image)
        x_processed = eval_transform(x_orig)
        weights = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
        mixture = torch.zeros_like(x_processed)
        for mix_index in range(3):
            x_aug = x_orig.copy()
            for _ in range(np.random.randint(1, 4)):
                x_aug = random.choice(augmentations)(x_aug)
            mixture += weights[mix_index] * eval_transform(x_aug)
        mix_ratio = np.float32(np.random.beta(1.0, 1.0))
        return mix_ratio * x_processed + (1.0 - mix_ratio) * mixture

    return memo_transform


def evaluate_model(model: nn.Module, loader, task_type: str, device: torch.device) -> Dict[str, float]:
    model.eval()
    logits_buffer: List[torch.Tensor] = []
    targets_buffer: List[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            logits_buffer.append(outputs.cpu())
            if isinstance(labels, torch.Tensor):
                targets_buffer.append(labels.cpu())
            else:
                targets_buffer.append(torch.as_tensor(labels))

    logits = torch.cat(logits_buffer, dim=0)
    targets = torch.cat(targets_buffer, dim=0)
    return compute_metrics(logits, targets, task_type)


def evaluate_with_memo(
    model: nn.Module,
    dataset,
    eval_transform: Callable,
    memo_transform: Callable,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    niter: int,
    batch_size: int,
    prior_strength: int,
    task_type: str,
    device: torch.device,
) -> Dict[str, float]:
    base_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    logits_buffer: List[torch.Tensor] = []
    targets_buffer: List[torch.Tensor] = []

    for index in range(len(dataset)):
        model.load_state_dict(base_state)
        optimizer = build_optimizer(model, optimizer_name, learning_rate, weight_decay)
        image, label = dataset[index]
        adapt_single(model, image, optimizer, memo_transform, task_type, niter, batch_size, prior_strength, device)
        logits = predict_single(model, image, eval_transform, prior_strength, device)
        logits_buffer.append(logits.cpu())
        if isinstance(label, torch.Tensor):
            targets_buffer.append(label.unsqueeze(0).cpu())
        else:
            targets_buffer.append(torch.tensor([label], dtype=torch.long))

    logits = torch.cat(logits_buffer, dim=0)
    targets = torch.cat(targets_buffer, dim=0)
    return compute_metrics(logits, targets, task_type)


def adapt_single(
    model: nn.Module,
    image,
    optimizer,
    memo_transform: Callable,
    task_type: str,
    niter: int,
    batch_size: int,
    prior_strength: int,
    device: torch.device,
) -> None:
    model.train()
    with batchnorm_prior(prior_strength):
        for _ in range(niter):
            inputs = torch.stack([memo_transform(image) for _ in range(batch_size)]).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if task_type == 'single_label':
                loss = softmax_marginal_entropy(outputs)
            else:
                loss = sigmoid_marginal_entropy(outputs)
            loss.backward()
            optimizer.step()
    model.eval()


def predict_single(
    model: nn.Module,
    image,
    eval_transform: Callable,
    prior_strength: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    inputs = eval_transform(image).unsqueeze(0).to(device)
    with batchnorm_prior(prior_strength):
        with torch.no_grad():
            logits = model(inputs)
    return logits


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, task_type: str) -> Dict[str, float]:
    if task_type == 'single_label':
        predictions = logits.argmax(dim=1)
        accuracy = predictions.eq(targets).float().mean().item()
        return {
            'accuracy': accuracy,
            'error': 1.0 - accuracy,
        }

    average_precision = multilabel_mean_average_precision(logits, targets)
    subset_accuracy = (torch.sigmoid(logits) > 0.5).eq(targets.bool()).all(dim=1).float().mean().item()
    return {
        'mAP': average_precision,
        'subset_accuracy': subset_accuracy,
    }


def softmax_marginal_entropy(outputs: torch.Tensor) -> torch.Tensor:
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def sigmoid_marginal_entropy(outputs: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(outputs).mean(dim=0)
    probabilities = probabilities.clamp(1e-6, 1.0 - 1e-6)
    entropy = -(probabilities * probabilities.log() + (1.0 - probabilities) * (1.0 - probabilities).log())
    return entropy.mean()


def multilabel_mean_average_precision(logits: torch.Tensor, targets: torch.Tensor) -> float:
    probabilities = torch.sigmoid(logits)
    targets = targets.float()
    per_class_ap = []
    for class_index in range(probabilities.shape[1]):
        scores = probabilities[:, class_index]
        labels = targets[:, class_index]
        positive_count = int(labels.sum().item())
        if positive_count == 0:
            continue
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_indices]
        cumulative_true_positives = torch.cumsum(sorted_labels, dim=0)
        ranks = torch.arange(1, len(sorted_labels) + 1, dtype=torch.float32)
        precision_at_k = cumulative_true_positives / ranks
        ap = (precision_at_k * sorted_labels).sum() / positive_count
        per_class_ap.append(ap.item())
    return float(np.mean(per_class_ap)) if per_class_ap else 0.0


@contextlib.contextmanager
def batchnorm_prior(prior_strength: int):
    if prior_strength is None or prior_strength < 0:
        yield
        return

    nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)
    nn.BatchNorm2d.forward = modified_batchnorm_forward
    try:
        yield
    finally:
        nn.BatchNorm2d.forward = ORIGINAL_BATCHNORM_FORWARD
        nn.BatchNorm2d.prior = 1.0


def modified_batchnorm_forward(self, inputs):
    estimated_mean = torch.zeros(self.running_mean.shape, device=self.running_mean.device)
    estimated_var = torch.ones(self.running_var.shape, device=self.running_var.device)
    nn.functional.batch_norm(inputs, estimated_mean, estimated_var, None, None, True, 1.0, self.eps)
    running_mean = self.prior * self.running_mean + (1 - self.prior) * estimated_mean
    running_var = self.prior * self.running_var + (1 - self.prior) * estimated_var
    return nn.functional.batch_norm(inputs, running_mean, running_var, self.weight, self.bias, False, 0, self.eps)