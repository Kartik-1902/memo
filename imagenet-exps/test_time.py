from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from utils.hf_dataset_helpers import dataset_choices, load_remote_sensing_dataset
from utils.memo_helpers import build_memo_augmentation, evaluate_model, evaluate_with_memo
from utils.model_helpers import (
    build_model,
    canonical_model_name,
    load_task_checkpoint,
    model_choices,
    save_task_checkpoint,
    train_linear_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=model_choices())
    parser.add_argument('--dataset', required=True, choices=dataset_choices())
    parser.add_argument('--resume', default=None)
    parser.add_argument('--results_dir', default='imagenet-exps/results/generated')
    parser.add_argument('--hf_cache_dir', default=None)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--adapt_batch_size', default=64, type=int)
    parser.add_argument('--niter', default=1, type=int)
    parser.add_argument('--prior_strength', default=16, type=int)
    parser.add_argument('--optimizer', default=None)
    parser.add_argument('--lr', default=None, type=float)
    parser.add_argument('--weight_decay', default=None, type=float)
    parser.add_argument('--linear_probe_epochs', default=5, type=int)
    parser.add_argument('--linear_probe_lr', default=5e-3, type=float)
    parser.add_argument('--linear_probe_weight_decay', default=1e-4, type=float)
    parser.add_argument('--linear_probe_batch_size', default=512, type=int)
    parser.add_argument('--split_seed', default=13, type=int)
    parser.add_argument('--eval_split', default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_bundle = load_remote_sensing_dataset(
        dataset_name=args.dataset,
        split_seed=args.split_seed,
        hf_cache_dir=args.hf_cache_dir,
        eval_split_override=args.eval_split,
    )
    model, model_info = build_model(args.model, dataset_bundle.class_names, device)

    optimizer_name = args.optimizer or model_info['optimizer']
    learning_rate = args.lr if args.lr is not None else model_info['lr']
    weight_decay = args.weight_decay if args.weight_decay is not None else model_info['weight_decay']

    print(f'Model: {model_info["display_name"]}')
    print(f'Dataset: {args.dataset} ({dataset_bundle.hf_id})')
    print(f'Train split: {dataset_bundle.train_split_name} | Eval split: {dataset_bundle.eval_split_name}')
    print(f'Task type: {dataset_bundle.task_type} | Metric: {dataset_bundle.metric_name}')

    checkpoint_path = args.resume or os.path.join(
        args.results_dir,
        f'{canonical_model_name(args.model)}_{args.dataset}',
        'ckpt.pth',
    )

    if model_info['requires_linear_probe']:
        train_dataset = dataset_bundle.train_dataset.clone_with_transform(model_info['eval_transform'])
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=torch.cuda.is_available(),
        )

        if os.path.exists(checkpoint_path):
            print(f'Resuming task head from {checkpoint_path}')
            load_task_checkpoint(model, checkpoint_path, device)
        else:
            print('Training a dataset-specific linear probe because no cached checkpoint was found.')
            train_linear_probe(
                model=model,
                train_loader=train_loader,
                device=device,
                task_type=dataset_bundle.task_type,
                epochs=args.linear_probe_epochs,
                learning_rate=args.linear_probe_lr,
                weight_decay=args.linear_probe_weight_decay,
                feature_batch_size=args.linear_probe_batch_size,
            )
            save_task_checkpoint(
                model=model,
                checkpoint_path=checkpoint_path,
                model_name=args.model,
                dataset_name=args.dataset,
                class_names=dataset_bundle.class_names,
                task_type=dataset_bundle.task_type,
                metric_name=dataset_bundle.metric_name,
            )
            print(f'Saved task head to {checkpoint_path}')

    eval_dataset = dataset_bundle.eval_dataset.clone_with_transform(model_info['eval_transform'])
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )
    baseline_metrics = evaluate_model(model, eval_loader, dataset_bundle.task_type, device)

    memo_transform = build_memo_augmentation(model_info['eval_transform'])
    adapted_metrics = evaluate_with_memo(
        model=model,
        dataset=dataset_bundle.eval_dataset,
        eval_transform=model_info['eval_transform'],
        memo_transform=memo_transform,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        niter=args.niter,
        batch_size=args.adapt_batch_size,
        prior_strength=args.prior_strength,
        task_type=dataset_bundle.task_type,
        device=device,
    )

    print('Baseline metrics:')
    for metric_name, value in baseline_metrics.items():
        print(f'  {metric_name}: {value:.4f}')

    print('MEMO metrics:')
    for metric_name, value in adapted_metrics.items():
        print(f'  {metric_name}: {value:.4f}')


if __name__ == '__main__':
    # MEMO-MOD: Unified test-time entrypoint for the new remote-sensing experiments.
    main()