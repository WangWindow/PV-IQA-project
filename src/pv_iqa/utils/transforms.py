from __future__ import annotations

from torchvision import transforms


def build_transforms(*, image_size: int, is_train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    common = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ]
    if not is_train:
        return transforms.Compose(common)

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.RandomResizedCrop(
                size=image_size,
                scale=(0.9, 1.0),
                ratio=(0.95, 1.05),
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )
