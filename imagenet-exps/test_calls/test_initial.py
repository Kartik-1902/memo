from __future__ import print_function
import argparse

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from utils.test_helpers import test
from utils.train_helpers import build_model, prepare_test_data


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
args = parser.parse_args()

if args.dataset:
	# MEMO-MODIFICATION: HF dataset path.
	from datasets.hf_datasets import load_hf_dataset

	if args.model is None:
		args.model = 'resnet50'

	teset, teloader, class_names, tr_transform, te_transform, clip_prompts = load_hf_dataset(
		args.dataset,
		split=None,
		model_name=args.model,
		use_transforms=True,
		batch_size=args.batch_size,
		workers=args.workers,
	)
	net = build_model(args, class_names=class_names, clip_prompts=clip_prompts)

	if args.resume and not args.model.startswith('clip'):
		print(f'Resuming from {args.resume}...')
		ckpt = torch.load(f'{args.resume}/ckpt.pth')
		net.load_state_dict(ckpt['state_dict'])

	progress_desc = f"{args.model}-{args.dataset}-initial"
	cls_initial, _, _ = test(
		teloader,
		net,
		args.corruption,
		verbose=True,
		apply_imagenet_masks=False,
		show_progress=True,
		progress_desc=progress_desc,
	)
	print(f'Error on HF test split {cls_initial*100:.2f}')
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
