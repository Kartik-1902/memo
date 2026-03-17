import time

import numpy as np
import torch
import torch.nn as nn

from utils.third_party import AverageMeter, ProgressMeter, imagenet_r_mask, indices_in_1k
from utils.train_helpers import get_device
from tqdm import tqdm


def test(
    teloader,
    model,
    corruption,
    verbose=False,
    print_freq=10,
    apply_imagenet_masks=True,
    show_progress=False,
    progress_desc=None,
):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(teloader), batch_time, top1, prefix='Test: ')
    one_hot = []
    losses = []
    device = get_device()
    criterion = nn.CrossEntropyLoss(reduction='none').to(device)
    end = time.time()

    iterator = enumerate(teloader)
    if show_progress:
        iterator = tqdm(iterator, total=len(teloader), desc=progress_desc, leave=True)

    for i, (inputs, labels) in iterator:
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if apply_imagenet_masks:
                if corruption == 'rendition':
                    outputs = outputs[:, imagenet_r_mask]
                elif corruption == 'adversarial':
                    outputs = outputs[:, indices_in_1k]
            _, predicted = outputs.max(1)
            losses.append(criterion(outputs, labels).cpu())
            one_hot.append(predicted.eq(labels).cpu())
        acc1 = one_hot[-1].sum().item() / len(labels)
        top1.update(acc1, len(labels))
        batch_time.update(time.time() - end)
        end = time.time()

        if show_progress:
            iterator.set_postfix(acc=f"{top1.avg:.4f}")

        if not show_progress and i % print_freq == 0:
            progress.print(i)
    print(f' * Acc@1 {top1.avg:.3f}')

    if verbose:
        one_hot = torch.cat(one_hot).numpy()
        losses = torch.cat(losses).numpy()
        return 1-top1.avg, one_hot, losses
    else:
        return 1-top1.avg
