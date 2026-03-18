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
from utils.model_helpers import build_model as build_rs_model, canonical_model_name
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
args = parser.parse_args()

if args.dataset:
    # MEMO-MODIFICATION: HF dataset path.
    from datasets.hf_datasets import load_hf_dataset

    if args.model is None:
        args.model = 'resnet50'

    # MEMO-MODIFICATION: map legacy model names to the registry used by model_helpers.
    model_name = canonical_model_name(args.model)
    if model_name == 'vitb16':
        model_name = 'vit_b16'
    elif model_name == 'clip_vitb16':
        model_name = 'clip_vit_b16'

    teset, _, class_names, tr_transform, te_transform, _ = load_hf_dataset(
        args.dataset,
        split=None,
        model_name=args.model,
        use_transforms=False,
        batch_size=args.batch_size,
        workers=8,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net, _ = build_rs_model(model_name, class_names, device)

    if args.resume and not args.model.startswith('clip'):
        print(f'Resuming from {args.resume}...')
        ckpt = torch.load(f'{args.resume}/ckpt.pth', map_location=device)
        net.load_state_dict(ckpt['state_dict'])
    else:
        ckpt = None
else:
    if not args.dataroot:
        raise ValueError('dataroot is required when --dataset is not provided.')
    if not args.resume:
        raise ValueError('resume is required when --dataset is not provided.')

    net = build_model(args)
    teset, _ = prepare_test_data(args, use_transforms=False)

    print(f'Resuming from {args.resume}...')
    ckpt = torch.load(f'{args.resume}/ckpt.pth')

def marginal_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits

if args.optimizer == 'sgd':
    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'adamw':
    optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

print('Running...')
correct = []
progress_desc = f"{args.model}-{args.dataset}-adapt" if args.dataset else f"imagenet-{args.corruption}-adapt"
pbar = tqdm(range(len(teset)), desc=progress_desc, leave=True)
for i in pbar:
    if ckpt is not None:
        net.load_state_dict(ckpt['state_dict'])
    image, label = teset[i]
    adapt_single(net, image, optimizer, marginal_entropy,
                 args.corruption, args.niter, args.batch_size, args.prior_strength,
                 tr_transform=tr_transform if args.dataset else None,
                 apply_imagenet_masks=not bool(args.dataset))
    correct.append(
        test_single(
            net,
            image,
            label,
            args.corruption,
            args.prior_strength,
            te_transform=te_transform if args.dataset else None,
            apply_imagenet_masks=not bool(args.dataset),
        )[0]
    )
    pbar.set_postfix(acc=f"{float(np.mean(correct)):.4f}")

print(f'MEMO adapt test error {(1-np.mean(correct))*100:.2f}')
