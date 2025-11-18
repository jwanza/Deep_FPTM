import argparse
import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    import timm
except ImportError:
    timm = None


def build_dataset(name: str, data_root: str, image_size: Tuple[int, int], batch_size: int) -> DataLoader:
    name = name.lower()
    if name == "mnist":
        dataset = datasets.MNIST(data_root, train=True, download=True, transform=transforms.ToTensor())
    elif name == "fashionmnist":
        dataset = datasets.FashionMNIST(data_root, train=True, download=True, transform=transforms.ToTensor())
    elif name == "cifar10":
        dataset = datasets.CIFAR10(
            data_root,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Resize(image_size)]),
        )
    else:
        raise ValueError(f"Unsupported dataset '{name}'.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher pseudo-labels dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (mnist/fashionmnist/cifar10).")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root directory.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for inference.")
    parser.add_argument("--teacher", type=str, required=True, help="timm model name to use as teacher.")
    parser.add_argument(
        "--img-size", type=int, nargs=2, default=(32, 32), metavar=("H", "W"), help="Image size for resizing."
    )
    parser.add_argument("--output", type=str, required=True, help="Path to output .pt file.")
    parser.add_argument("--device", type=str, default="cuda", help="Device identifier.")
    args = parser.parse_args()

    if timm is None:
        raise ImportError("timm is required to generate teacher pseudo-labels.")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loader = build_dataset(args.dataset, args.data_root, tuple(args.img_size), args.batch_size)

    teacher = timm.create_model(args.teacher, pretrained=True)
    teacher.reset_classifier(loader.dataset.targets.unique().numel() if hasattr(loader.dataset, "targets") else 10)
    teacher.to(device)
    teacher.eval()

    images = []
    logits = []
    with torch.inference_mode():
        for batch, _ in loader:
            batch = batch.to(device)
            out = teacher(batch)
            out = out[0] if isinstance(out, (tuple, list)) else out
            images.append(batch.cpu())
            logits.append(out.cpu())

    images_tensor = torch.cat(images, dim=0)
    logits_tensor = torch.cat(logits, dim=0)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images_tensor, "logits": logits_tensor}, output_path)
    print(f"Saved pseudo-label dataset to {output_path} (samples={images_tensor.size(0)})")


if __name__ == "__main__":
    main()

