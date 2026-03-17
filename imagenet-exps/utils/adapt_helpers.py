import torch
import torch.nn as nn

from utils.train_helpers import tr_transforms, te_transforms, te_transforms_inc, common_corruptions
from utils.third_party import indices_in_1k, imagenet_r_mask


def adapt_single(model, image, optimizer, criterion,
                 corruption, niter, batch_size, prior_strength,
                 tr_transform=None, apply_imagenet_masks=True):
    model.eval()

    # MEMO-MODIFICATION: allow custom transforms for HF datasets.
    if tr_transform is None:
        tr_transform = tr_transforms

    if prior_strength < 0:
        nn.BatchNorm2d.prior = 1
    else:
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)

    for iteration in range(niter):
        inputs = [tr_transform(image) for _ in range(batch_size)]
        inputs = torch.stack(inputs).cuda()
        optimizer.zero_grad()
        outputs = model(inputs)

        if apply_imagenet_masks:
            if corruption == 'rendition':
                outputs = outputs[:, imagenet_r_mask]
            elif corruption == 'adversarial':
                outputs = outputs[:, indices_in_1k]
        loss, logits = criterion(outputs)
        loss.backward()
        optimizer.step()
    nn.BatchNorm2d.prior = 1

def test_single(model, image, label, corruption, prior_strength,
                te_transform=None, apply_imagenet_masks=True):
    model.eval()

    if prior_strength < 0:
        nn.BatchNorm2d.prior = 1
    else:
        nn.BatchNorm2d.prior = float(prior_strength) / float(prior_strength + 1)
    if te_transform is None:
        transform = te_transforms_inc if corruption in common_corruptions else te_transforms
    else:
        transform = te_transform
    inputs = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(inputs.cuda())

        if apply_imagenet_masks:
            if corruption == 'rendition':
                outputs = outputs[:, imagenet_r_mask]
            elif corruption == 'adversarial':
                outputs = outputs[:, indices_in_1k]
        _, predicted = outputs.max(1)
        confidence = nn.functional.softmax(outputs, dim=1).squeeze()[predicted].item()
    correctness = 1 if predicted.item() == label else 0
    nn.BatchNorm2d.prior = 1
    return correctness, confidence
