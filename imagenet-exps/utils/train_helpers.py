import copy
import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from utils.third_party import aug  # also runs timm model registry for RVT*-small


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
tr_transforms = aug
te_transforms = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    normalize])
# ImageNet-C has already been resized and center cropper, the center crop below is a no op
te_transforms_inc = transforms.Compose([transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize])

common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness',
                      'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']


# https://github.com/bethgelab/robustness/blob/main/robusta/batchnorm/bn.py#L175
def _modified_bn_forward(self, input):
    est_mean = torch.zeros(self.running_mean.shape, device=self.running_mean.device)
    est_var = torch.ones(self.running_var.shape, device=self.running_var.device)
    nn.functional.batch_norm(input, est_mean, est_var, None, None, True, 1.0, self.eps)
    running_mean = self.prior * self.running_mean + (1 - self.prior) * est_mean
    running_var = self.prior * self.running_var + (1 - self.prior) * est_var
    return nn.functional.batch_norm(input, running_mean, running_var, self.weight, self.bias, False, 0, self.eps)


class ClipZeroShot(nn.Module):
    # MEMO-MODIFICATION: Wrap CLIP image encoder with per-dataset text features.
    def __init__(self, clip_model: nn.Module, text_features: torch.Tensor):
        super().__init__()
        self.clip_model = clip_model
        self.register_buffer("text_features", text_features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.clip_model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        return logit_scale * image_features @ self.text_features.t()


def _build_clip_text_features(clip_model: nn.Module, clip_prompts: List[str]) -> torch.Tensor:
    import clip

    tokens = clip.tokenize(clip_prompts).cuda()
    with torch.no_grad():
        text_features = clip_model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


def get_device() -> torch.device:
    # MEMO-MODIFICATION: force GPU-only execution.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Install CUDA-enabled PyTorch or set CUDA_VISIBLE_DEVICES.")
    return torch.device("cuda")


def build_model(args, class_names: Optional[List[str]] = None, clip_prompts: Optional[List[str]] = None):
    model_name = getattr(args, 'model', None)
    use_pretrained = bool(getattr(args, 'dataset', None)) and not bool(getattr(args, 'resume', None))
    device = get_device()

    if model_name == 'resnet50':
        net = models.resnet50(pretrained=use_pretrained).to(device)
    elif model_name == 'vitb16':
        from timm.models import create_model
        net = create_model('vit_base_patch16_224', pretrained=use_pretrained).to(device)
    elif model_name in ('clip_resnet50', 'clip_vitb16'):
        if not clip_prompts:
            raise ValueError('clip_prompts are required for CLIP zero-shot models.')
        import clip

        clip_id = 'RN50' if model_name == 'clip_resnet50' else 'ViT-B/16'
        clip_model, _ = clip.load(clip_id, device=device, jit=False)
        text_features = _build_clip_text_features(clip_model, clip_prompts)
        net = ClipZeroShot(clip_model, text_features).to(device)
    elif hasattr(args, 'use_rvt') and args.use_rvt:
        print('constructing rvt+ small')
        from timm.models import create_model
        net = create_model('rvt_small_plus', drop_path_rate=0.1).to(device)
    elif hasattr(args, 'use_resnext') and args.use_resnext:
        net = models.resnext101_32x8d().to(device)
    else:
        net = models.resnet50(pretrained=False).to(device)
    if device.type == "cuda":
        net = torch.nn.DataParallel(net)

    if hasattr(args, 'prior_strength'):
        if (not args.prior_strength is None and args.prior_strength >= 0):
            print('modifying BN forward pass')
            nn.BatchNorm2d.prior = float(args.prior_strength) / float(args.prior_strength + 1)
            nn.BatchNorm2d.forward = _modified_bn_forward
    return net

def prepare_test_data(args, use_transforms=True):
    if args.corruption in common_corruptions:
        te_transforms_local = te_transforms_inc if use_transforms else None	
        print(f'Test on {args.corruption} level {args.level}')
        validdir = os.path.join(args.dataroot, 'imagenet-c', args.corruption, str(args.level))
        teset = datasets.ImageFolder(validdir, te_transforms_local)
    elif args.corruption == 'rendition':
        te_transforms_local = te_transforms if use_transforms else None	
        validdir = os.path.join(args.dataroot, 'imagenet-r')
        teset = datasets.ImageFolder(validdir, te_transforms_local)
    elif args.corruption == 'adversarial':
        te_transforms_local = te_transforms if use_transforms else None	
        validdir = os.path.join(args.dataroot, 'imagenet-a')
        teset = datasets.ImageFolder(validdir, te_transforms_local)
    else:
        raise Exception('Corruption not found!')

    if not hasattr(args, 'workers'):
        args.workers = 8
    collate_fn = None if use_transforms else lambda x: x
    teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    return teset, teloader
