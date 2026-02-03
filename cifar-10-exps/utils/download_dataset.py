"""
Utility to download CIFAR-10 dataset from Hugging Face and convert it to PyTorch format.
"""

import os
import numpy as np
from pathlib import Path


def download_cifar10_from_hf(save_dir: str) -> str:
    """
    Download CIFAR-10 dataset from Hugging Face and save it in a format
    compatible with torchvision.datasets.CIFAR10.
    
    Args:
        save_dir: Directory where the dataset will be saved (e.g., ~/btp)
    
    Returns:
        str: Path to the dataset root directory
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Please install the 'datasets' package: pip install datasets"
        )
    
    # Expand user path and create directory
    save_dir = os.path.expanduser(save_dir)
    dataset_root = os.path.join(save_dir, "Memo")
    os.makedirs(dataset_root, exist_ok=True)
    
    # Check if dataset already exists
    cifar10_dir = os.path.join(dataset_root, "cifar-10-batches-py")
    if os.path.exists(cifar10_dir):
        print(f"CIFAR-10 dataset already exists at {dataset_root}")
        return dataset_root
    
    print(f"Downloading CIFAR-10 from Hugging Face to {dataset_root}...")
    
    # Load dataset from Hugging Face
    dataset = load_dataset("cifar10", cache_dir=os.path.join(save_dir, ".hf_cache"))
    
    # Create the directory structure expected by torchvision
    os.makedirs(cifar10_dir, exist_ok=True)
    
    # Process train data
    train_data = dataset["train"]
    train_images = np.array([np.array(img) for img in train_data["img"]])
    train_labels = np.array(train_data["label"])
    
    # Process test data
    test_data = dataset["test"]
    test_images = np.array([np.array(img) for img in test_data["img"]])
    test_labels = np.array(test_data["label"])
    
    # Save in pickle format compatible with torchvision CIFAR10
    import pickle
    
    # Save training batches (split into 5 batches like original CIFAR-10)
    batch_size = 10000
    for i in range(5):
        batch_data = {
            b'data': train_images[i*batch_size:(i+1)*batch_size].reshape(-1, 3*32*32),
            b'labels': train_labels[i*batch_size:(i+1)*batch_size].tolist(),
            b'batch_label': f'training batch {i+1} of 5'.encode(),
            b'filenames': [f'image_{j}.png'.encode() for j in range(batch_size)]
        }
        with open(os.path.join(cifar10_dir, f"data_batch_{i+1}"), 'wb') as f:
            pickle.dump(batch_data, f)
    
    # Save test batch
    test_batch_data = {
        b'data': test_images.reshape(-1, 3*32*32),
        b'labels': test_labels.tolist(),
        b'batch_label': b'testing batch 1 of 1',
        b'filenames': [f'image_{j}.png'.encode() for j in range(len(test_labels))]
    }
    with open(os.path.join(cifar10_dir, "test_batch"), 'wb') as f:
        pickle.dump(test_batch_data, f)
    
    # Save meta data
    meta = {
        b'num_cases_per_batch': 10000,
        b'label_names': [b'airplane', b'automobile', b'bird', b'cat', b'deer',
                         b'dog', b'frog', b'horse', b'ship', b'truck'],
        b'num_vis': 3072
    }
    with open(os.path.join(cifar10_dir, "batches.meta"), 'wb') as f:
        pickle.dump(meta, f)
    
    print(f"CIFAR-10 dataset saved to {dataset_root}")
    return dataset_root


def get_default_dataroot() -> str:
    """
    Get the default dataroot path based on server structure.
    For DGX server: ~/btp/Memo
    
    Returns:
        str: Default dataroot path
    """
    home = os.path.expanduser("~")
    return os.path.join(home, "btp", "Memo")


if __name__ == "__main__":
    # Test download
    dataroot = download_cifar10_from_hf("~/btp")
    print(f"Dataset downloaded to: {dataroot}")
