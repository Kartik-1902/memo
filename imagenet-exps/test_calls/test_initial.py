from __future__ import print_function
import argparse
import os
import sys

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from utils.hf_dataset_helpers import load_remote_sensing_dataset
from utils.memo_helpers import evaluate_model
from utils.model_helpers import (
	build_model as build_rs_model,
	canonical_model_name,
	load_task_checkpoint,
	save_task_checkpoint,
	train_linear_probe,
)
from utils.test_helpers import test
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
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--workers', default=8, type=int)
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
				num_workers=args.workers,
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

	eval_dataset = dataset_bundle.eval_dataset.clone_with_transform(model_info['eval_transform'])
	eval_loader = torch.utils.data.DataLoader(
		eval_dataset,
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.workers,
		pin_memory=torch.cuda.is_available(),
	)
	metrics = evaluate_model(net, eval_loader, dataset_bundle.task_type, device)
	if dataset_bundle.task_type == 'single_label':
		print(f"Error on HF test split {(1 - metrics['accuracy']) * 100:.2f}")
	else:
		print(f"HF test split mAP {metrics['mAP']:.4f}")
else:
	if not args.dataroot:
		raise ValueError('dataroot is required when --dataset is not provided.')
	if not args.resume:
		raise ValueError('resume is required when --dataset is not provided.')

	net = build_model(args)
	teset, teloader = prepare_test_data(args)

	print(f'Resuming from {args.resume}...')
	ckpt = torch.load(f'{args.resume}/ckpt.pth')
	net.load_state_dict(ckpt['state_dict'])
	progress_desc = f"imagenet-{args.corruption}-initial"
	cls_initial, _, _ = test(
		teloader,
		net,
		args.corruption,
		verbose=True,
		show_progress=True,
		progress_desc=progress_desc,
	)

	print(f'Error on corrupted test set {cls_initial*100:.2f}')
