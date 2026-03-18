from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from tqdm import tqdm
from utils.adapt_helpers import adapt_single, test_single
from utils.hf_dataset_helpers import load_remote_sensing_dataset
from utils.memo_helpers import (
    batchnorm_prior,
    build_memo_augmentation,
    build_optimizer,
    compute_metrics,
    sigmoid_marginal_entropy,
    softmax_marginal_entropy,
)
from utils.model_helpers import (
    build_model as build_rs_model,
    canonical_model_name,
    load_task_checkpoint,
    save_task_checkpoint,
    train_linear_probe,
)
from utils.train_helpers import build_model, prepare_test_data

# MEMO-MODIFICATION: prefer local datasets/ over HF package name collision.
_LOCAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _LOCAL_ROOT not in sys.path:
    sys.path.insert(0, _LOCAL_ROOT)


parser = argparse.ArgumentParser()
# MEMO-MODIFICATION: dataset/model args are additive; ImageNet path remains default.
parser.add_argument('--dataset', default=None, choices=['patternnet', 'rsicd', 'resisc45', 'mlrsnet'])
parser.add_argument('--model', default=None, choices=['resnet50', 'vitb16', 'clip_resnet50', 'clip_vitb16'])
parser.add_argument('--dataroot', default=None)
parser.add_argument('--use_rvt', action='store_true')
parser.add_argument('--use_resnext', action='store_true')
parser.add_argument('--level', default=0, type=int)
parser.add_argument('--corruption', default='original')
parser.add_argument('--resume', default=None)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--prior_strength', default=16, type=int)
parser.add_argument('--optimizer', default='sgd')
parser.add_argument('--lr', default=0.00025, type=float)
parser.add_argument('--weight_decay', default=0.0, type=float)
parser.add_argument('--niter', default=1, type=int)
parser.add_argument('--results_dir', default='imagenet-exps/results/generated')
parser.add_argument('--hf_cache_dir', default=None)
parser.add_argument('--split_seed', default=13, type=int)
parser.add_argument('--eval_split', default=None)
parser.add_argument('--linear_probe_epochs', default=5, type=int)
parser.add_argument('--linear_probe_lr', default=5e-3, type=float)
parser.add_argument('--linear_probe_weight_decay', default=1e-4, type=float)
parser.add_argument('--linear_probe_batch_size', default=512, type=int)
args = parser.parse_args()

if args.dataset:
    if args.model is None:
        args.model = 'resnet50'

    # MEMO-MODIFICATION: map legacy model names to the registry used by model_helpers.
    model_name = canonical_model_name(args.model)
    if model_name == 'vitb16':
        model_name = 'vit_b16'
    elif model_name == 'clip_vitb16':
        model_name = 'clip_vit_b16'

    dataset_bundle = load_remote_sensing_dataset(
        dataset_name=args.dataset,
        split_seed=args.split_seed,
        hf_cache_dir=args.hf_cache_dir,
        eval_split_override=args.eval_split,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, model_info = build_rs_model(model_name, dataset_bundle.class_names, device)

    checkpoint_path = args.resume or os.path.join(
        args.results_dir,
        f'{canonical_model_name(model_name)}_{args.dataset}',
        'ckpt.pth',
    )

    if model_info['requires_linear_probe']:
        if os.path.exists(checkpoint_path):
            print(f'Resuming task head from {checkpoint_path}')
            load_task_checkpoint(net, checkpoint_path, device)
        else:
            print('Training a dataset-specific linear probe because no cached checkpoint was found.')
            train_dataset = dataset_bundle.train_dataset.clone_with_transform(model_info['eval_transform'])
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=torch.cuda.is_available(),
            )
            train_linear_probe(
                model=net,
                train_loader=train_loader,
                device=device,
                task_type=dataset_bundle.task_type,
                epochs=args.linear_probe_epochs,
                learning_rate=args.linear_probe_lr,
                weight_decay=args.linear_probe_weight_decay,
                feature_batch_size=args.linear_probe_batch_size,
            )
            save_task_checkpoint(
                model=net,
                checkpoint_path=checkpoint_path,
                model_name=model_name,
                dataset_name=args.dataset,
                class_names=dataset_bundle.class_names,
                task_type=dataset_bundle.task_type,
                metric_name=dataset_bundle.metric_name,
            )
            print(f'Saved task head to {checkpoint_path}')
else:
    if not args.dataroot:
        raise ValueError('dataroot is required when --dataset is not provided.')
    if not args.resume:
        raise ValueError('resume is required when --dataset is not provided.')

    net = build_model(args)
    teset, _ = prepare_test_data(args, use_transforms=False)

    print(f'Resuming from {args.resume}...')
    ckpt = torch.load(f'{args.resume}/ckpt.pth')

def _build_marginal_entropy(task_type: str):
    def _softmax_entropy(outputs):
        return softmax_marginal_entropy(outputs), outputs

    def _sigmoid_entropy(outputs):
        return sigmoid_marginal_entropy(outputs), outputs

    return _softmax_entropy if task_type == 'single_label' else _sigmoid_entropy

print('Running...')
if args.dataset:
    eval_transform = model_info['eval_transform']
    memo_transform = build_memo_augmentation(eval_transform)
    criterion = _build_marginal_entropy(dataset_bundle.task_type)
    base_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
    logits_buffer = []
    targets_buffer = []

    progress_desc = f"{args.model}-{args.dataset}-adapt"
    pbar = tqdm(range(len(dataset_bundle.eval_dataset)), desc=progress_desc, leave=True)
    for i in pbar:
        net.load_state_dict(base_state)
        optimizer = build_optimizer(net, args.optimizer, args.lr, args.weight_decay)
        image, label = dataset_bundle.eval_dataset[i]
        adapt_single(
            net,
            image,
            optimizer,
            criterion,
            args.corruption,
            args.niter,
            args.batch_size,
            args.prior_strength,
            tr_transform=memo_transform,
            apply_imagenet_masks=False,
        )
        inputs = eval_transform(image).unsqueeze(0).to(device)
        with batchnorm_prior(args.prior_strength):
            with torch.no_grad():
                logits = net(inputs)
        logits_buffer.append(logits.cpu())
        if isinstance(label, torch.Tensor):
            targets_buffer.append(label.unsqueeze(0).cpu())
        else:
            targets_buffer.append(torch.tensor([label], dtype=torch.long))

        metrics = compute_metrics(torch.cat(logits_buffer, dim=0), torch.cat(targets_buffer, dim=0), dataset_bundle.task_type)
        if dataset_bundle.task_type == 'single_label':
            pbar.set_postfix(acc=f"{metrics['accuracy']:.4f}")
        else:
            pbar.set_postfix(mAP=f"{metrics['mAP']:.4f}")

    final_metrics = compute_metrics(torch.cat(logits_buffer, dim=0), torch.cat(targets_buffer, dim=0), dataset_bundle.task_type)
    if dataset_bundle.task_type == 'single_label':
        print(f"MEMO adapt test error {(1 - final_metrics['accuracy']) * 100:.2f}")
    else:
        print(f"MEMO adapt test mAP {final_metrics['mAP']:.4f}")
else:
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    correct = []
    progress_desc = f"imagenet-{args.corruption}-adapt"
    pbar = tqdm(range(len(teset)), desc=progress_desc, leave=True)
    for i in pbar:
        if ckpt is not None:
            net.load_state_dict(ckpt['state_dict'])
        image, label = teset[i]
        adapt_single(net, image, optimizer, _build_marginal_entropy('single_label'),
                     args.corruption, args.niter, args.batch_size, args.prior_strength,
                     tr_transform=None,
                     apply_imagenet_masks=True)
        correct.append(
            test_single(
                net,
                image,
                label,
                args.corruption,
                args.prior_strength,
                te_transform=None,
                apply_imagenet_masks=True,
            )[0]
        )
        pbar.set_postfix(acc=f"{float(np.mean(correct)):.4f}")

    print(f'MEMO adapt test error {(1-np.mean(correct))*100:.2f}')
