from __future__ import print_function
import argparse
import os
import sys

import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from utils.test_helpers import test
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
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--workers', default=8, type=int)
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

	teset, teloader, class_names, tr_transform, te_transform, _ = load_hf_dataset(
		args.dataset,
		split=None,
		model_name=args.model,
		use_transforms=True,
		batch_size=args.batch_size,
		workers=args.workers,
	)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	net, _ = build_rs_model(model_name, class_names, device)

	if args.resume and not args.model.startswith('clip'):
		print(f'Resuming from {args.resume}...')
		ckpt = torch.load(f'{args.resume}/ckpt.pth', map_location=device)
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
