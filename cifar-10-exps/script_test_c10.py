import argparse
import os
import sys

from subprocess import call

# Add utils to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.download_dataset import download_cifar10_from_hf, get_default_dataroot


parser = argparse.ArgumentParser()
parser.add_argument('--experiment', default='cifar10')
parser.add_argument('--resume', default='rn26_gn')
parser.add_argument('--dataroot', default=None, help='Path to dataset root. If not provided, will download to ~/btp')
parser.add_argument('--download', action='store_true', help='Download CIFAR-10 from Hugging Face')
args = parser.parse_args()
experiment = args.experiment
resume = args.resume

# Set dataroot - either use provided path or download to ~/btp/Memo
if args.dataroot:
    dataroot = args.dataroot
else:
    # Default path for DGX server: ~/btp/Memo
    dataroot = get_default_dataroot()

# Download dataset from Hugging Face if requested or if it doesn't exist
if args.download or not os.path.exists(dataroot):
    print(f"Downloading CIFAR-10 from Hugging Face to ~/btp/Memo...")
    dataroot = download_cifar10_from_hf(os.path.expanduser("~/btp"))

print(f"Using dataroot: {dataroot}")

if experiment == 'cifar10':
    corruptions = ['original']
    levels = [0]
elif experiment == 'cifar101':
    corruptions = ['cifar_new']
    levels = [0]
elif experiment == 'cifar10c':
    corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                   'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                   'snow', 'frost', 'fog', 'brightness',
                   'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    levels = [1, 2, 3, 4, 5]

for corruption in corruptions:
    for level in levels:
        print(corruption, level)
        call(' '.join(['python', 'test_calls/test_initial.py',
                       f'--dataroot {dataroot}',
                       f'--level {level}',
                       f'--corruption {corruption}',
                       f'--resume results/cifar10_{resume}/']),
             shell=True)

        call(' '.join(['python', 'test_calls/test_adapt.py',
                       f'--dataroot {dataroot}',
                       f'--level {level}',
                       f'--corruption {corruption}',
                       f'--resume results/cifar10_{resume}/']),
             shell=True)
