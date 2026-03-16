import json
import logging
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from domino.base_piece import BasePiece
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from .models import InputModel, OutputModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[]
)
logger = logging.getLogger(__name__)

try:
    try:
        from ..utils import apply_inscribed_circle_mask, ensure_parent_dir, validate_crop_box
    except ImportError:  # pragma: no cover
        from pieces.utils import apply_inscribed_circle_mask, ensure_parent_dir, validate_crop_box
except Exception as e:
    logger.exception(f"Could not import utils.py: {e}")
    raise e

_ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
_NORMALIZE_STD = [0.229, 0.224, 0.225]


class FixedCrop:
    def __init__(self, crop_box: tuple[int, int, int, int]):
        self.crop_box = crop_box

    def __call__(self, img: Image.Image) -> Image.Image:
        img = img.convert('RGB')
        width, height = img.size
        left, top, right, bottom = validate_crop_box(*self.crop_box, width, height)
        return img.crop((left, top, right, bottom))


class InscribedCircleMask:
    def __init__(self, background: tuple[int, int, int]):
        self.background = background

    def __call__(self, img: Image.Image) -> Image.Image:
        return apply_inscribed_circle_mask(img, background=self.background)


class RandomNonCardinalRotation:
    def __init__(
        self,
        gap_deg: float = 2.0,
        fill: tuple[int, int, int] = (0, 0, 0),
        interpolation: InterpolationMode = InterpolationMode.BICUBIC,
    ):
        if not (0 <= gap_deg < 45):
            raise ValueError('gap_deg must be in [0, 45).')
        self.fill = fill
        self.interpolation = interpolation
        self.ranges = [
            (0.0 + gap_deg, 90.0 - gap_deg),
            (90.0 + gap_deg, 180.0 - gap_deg),
            (180.0 + gap_deg, 270.0 - gap_deg),
            (270.0 + gap_deg, 360.0 - gap_deg),
        ]

    def __call__(self, img: Image.Image) -> Image.Image:
        low, high = random.choice(self.ranges)
        angle = random.uniform(low, high)
        return transforms.functional.rotate(
            img,
            angle=angle,
            interpolation=self.interpolation,
            expand=False,
            fill=self.fill,
        )


class NozzlePairDataset(Dataset):
    def __init__(self, image_paths: list[Path], transform, pairs_per_epoch: int, positive_prob: float):
        self.image_paths = list(image_paths)
        self.transform = transform
        self.pairs_per_epoch = int(pairs_per_epoch)
        self.positive_prob = float(positive_prob)

        if len(self.image_paths) < 2:
            raise ValueError('Need at least 2 images to create negative pairs.')

    def __len__(self):
        return self.pairs_per_epoch

    def _load(self, path: Path) -> Image.Image:
        with Image.open(path) as img:
            return img.convert('RGB')

    def __getitem__(self, index):
        same = random.random() < self.positive_prob

        if same:
            image_path = random.choice(self.image_paths)
            image = self._load(image_path)
            left = self.transform(image)
            right = self.transform(image)
            label = torch.tensor(1.0, dtype=torch.float32)
        else:
            left_path, right_path = random.sample(self.image_paths, 2)
            left = self.transform(self._load(left_path))
            right = self.transform(self._load(right_path))
            label = torch.tensor(0.0, dtype=torch.float32)

        return left, right, label


class EmbeddingNet(nn.Module):
    def __init__(self, backbone_name: str, embedding_dim: int, pretrained: bool = True):
        super().__init__()

        if backbone_name == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet18(weights=weights)
            in_features = backbone.fc.in_features
        elif backbone_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            backbone = models.resnet50(weights=weights)
            in_features = backbone.fc.in_features
        else:
            raise ValueError("Unsupported backbone. Use 'resnet18' or 'resnet50'.")

        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
        )

    def forward_once(self, x):
        embedding = self.backbone(x)
        embedding = self.projector(embedding)
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, left, right):
        return self.forward_once(left), self.forward_once(right)


class SiameseImageSimilarityTrainPiece(BasePiece):
    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _find_image_files(self, root: Path) -> list[Path]:
        image_paths = sorted(
            path for path in root.rglob('*')
            if path.is_file() and path.suffix.lower() in _ALLOWED_EXTENSIONS
        )
        if not image_paths:
            raise RuntimeError(f'No image files found in: {root}')
        return image_paths

    def _split_paths(self, image_paths: list[Path], train_ratio: float, random_seed: int) -> tuple[list[Path], list[Path]]:
        if not 0 < train_ratio < 1:
            raise ValueError('train_ratio must be in the open interval (0, 1).')

        unique_names = {path.name for path in image_paths}
        if len(unique_names) != len(image_paths):
            raise ValueError('Each filename must be unique across the dataset.')

        shuffled_paths = image_paths.copy()
        rng = random.Random(random_seed)
        rng.shuffle(shuffled_paths)

        split_index = max(1, int(len(shuffled_paths) * train_ratio))
        split_index = min(split_index, len(shuffled_paths) - 1)
        train_paths = shuffled_paths[:split_index]
        val_paths = shuffled_paths[split_index:]

        if len(train_paths) < 2 or len(val_paths) < 2:
            raise ValueError('Training and validation splits must each contain at least 2 images.')

        return train_paths, val_paths

    def _create_transforms(self, input_data: InputModel):
        background = tuple(int(value) for value in input_data.circle_background_color)
        crop_box = (
            input_data.crop_left,
            input_data.crop_top,
            input_data.crop_right,
            input_data.crop_bottom,
        )

        train_transform = transforms.Compose([
            FixedCrop(crop_box),
            InscribedCircleMask(background),
            RandomNonCardinalRotation(gap_deg=input_data.cardinal_gap_deg, fill=background),
            transforms.Resize((input_data.image_size, input_data.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.03, 0.03),
                scale=(0.97, 1.03),
                interpolation=InterpolationMode.BICUBIC,
                fill=background,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=_NORMALIZE_MEAN, std=_NORMALIZE_STD),
        ])

        eval_transform = transforms.Compose([
            FixedCrop(crop_box),
            InscribedCircleMask(background),
            RandomNonCardinalRotation(gap_deg=input_data.cardinal_gap_deg, fill=background),
            transforms.Resize((input_data.image_size, input_data.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=_NORMALIZE_MEAN, std=_NORMALIZE_STD),
        ])

        return train_transform, eval_transform

    def _create_data_loaders(self, train_paths: list[Path], val_paths: list[Path], input_data: InputModel):
        train_transform, eval_transform = self._create_transforms(input_data)
        train_dataset = NozzlePairDataset(
            image_paths=train_paths,
            transform=train_transform,
            pairs_per_epoch=input_data.train_pairs_per_epoch,
            positive_prob=input_data.positive_prob,
        )
        val_dataset = NozzlePairDataset(
            image_paths=val_paths,
            transform=eval_transform,
            pairs_per_epoch=input_data.val_pairs_per_epoch,
            positive_prob=input_data.positive_prob,
        )

        num_workers = 2 if torch.cuda.is_available() else 0
        pin_memory = torch.cuda.is_available()

        train_loader = DataLoader(
            train_dataset,
            batch_size=input_data.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=input_data.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader

    def _cosine_targets_from_binary(self, labels: torch.Tensor) -> torch.Tensor:
        return torch.where(labels > 0.5, torch.ones_like(labels), -torch.ones_like(labels))

    def _run_epoch(self, model, loader, device, criterion, optimizer=None, scaler=None, cosine_threshold: float = 0.75):
        is_train = optimizer is not None
        model.train(is_train)

        total_loss = 0.0
        similarities = []
        labels_all = []

        for left, right, labels in loader:
            left = left.to(device, non_blocking=True)
            right = right.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            cosine_targets = self._cosine_targets_from_binary(labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                left_embedding, right_embedding = model(left, right)
                loss = criterion(left_embedding, right_embedding, cosine_targets)
                batch_similarities = F.cosine_similarity(left_embedding, right_embedding)

            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * left.size(0)
            similarities.append(batch_similarities.detach().cpu())
            labels_all.append(labels.detach().cpu())

        similarities_np = torch.cat(similarities).numpy()
        labels_np = torch.cat(labels_all).numpy()
        predictions = (similarities_np >= cosine_threshold).astype(np.float32)
        accuracy = accuracy_score(labels_np, predictions)
        average_loss = total_loss / len(loader.dataset)
        return average_loss, accuracy

    def _save_training_plot(self, history: dict[str, list[float]], plot_path: str, cosine_threshold: float):
        plt.figure(figsize=(14, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train loss')
        plt.plot(history['val_loss'], label='Val loss')
        plt.title('CosineEmbeddingLoss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc_init_thr'], label='Train acc @ initial threshold')
        plt.plot(history['val_acc_init_thr'], label='Val acc @ initial threshold')
        plt.title(f'Accuracy @ threshold={cosine_threshold:.2f}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

    def piece_function(self, input_data: InputModel):
        try:
            logger.info('Starting Siamese Image Similarity Train Piece')
            self._set_seed(input_data.random_seed)

            image_paths = self._find_image_files(Path(input_data.train_data_path))
            train_paths, val_paths = self._split_paths(image_paths, input_data.train_ratio, input_data.random_seed)
            train_loader, val_loader = self._create_data_loaders(train_paths, val_paths, input_data)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = EmbeddingNet(
                backbone_name=input_data.backbone,
                embedding_dim=input_data.embedding_dim,
                pretrained=True,
            ).to(device)

            criterion = nn.CosineEmbeddingLoss(margin=input_data.cosine_margin)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=input_data.learning_rate,
                weight_decay=input_data.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=input_data.epochs,
                eta_min=1e-6,
            )
            scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))

            trained_model_dir = os.path.join(input_data.model_output_path, 'trained_model')
            os.makedirs(trained_model_dir, exist_ok=True)

            best_model_file_path = os.path.join(trained_model_dir, 'best_model.pth')
            last_model_file_path = os.path.join(trained_model_dir, 'last_model.pth')
            config_path = os.path.join(trained_model_dir, 'config.json')
            training_plot_file_path = os.path.join(trained_model_dir, 'training_metrics.png')

            history = {
                'train_loss': [],
                'val_loss': [],
                'train_acc_init_thr': [],
                'val_acc_init_thr': [],
            }
            best_val_loss = float('inf')

            for epoch in range(1, input_data.epochs + 1):
                train_loss, train_acc = self._run_epoch(
                    model=model,
                    loader=train_loader,
                    device=device,
                    criterion=criterion,
                    optimizer=optimizer,
                    scaler=scaler,
                    cosine_threshold=input_data.cosine_threshold,
                )
                val_loss, val_acc = self._run_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    criterion=criterion,
                    optimizer=None,
                    scaler=None,
                    cosine_threshold=input_data.cosine_threshold,
                )
                scheduler.step()

                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_acc_init_thr'].append(train_acc)
                history['val_acc_init_thr'].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {
                            'epoch': epoch,
                            'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'val_loss': val_loss,
                        },
                        best_model_file_path,
                    )

                logger.info(
                    'Epoch %s/%s | train_loss=%.4f | val_loss=%.4f | train_acc=%.4f | val_acc=%.4f',
                    epoch,
                    input_data.epochs,
                    train_loss,
                    val_loss,
                    train_acc,
                    val_acc,
                )

            torch.save(
                {
                    'epoch': input_data.epochs,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                },
                last_model_file_path,
            )

            config_data = input_data.model_dump()
            config_data.update({
                'train_files': [str(path) for path in train_paths],
                'validation_files': [str(path) for path in val_paths],
                'best_val_loss': best_val_loss,
            })
            ensure_parent_dir(config_path)
            with open(config_path, 'w', encoding='utf-8') as file:
                json.dump(config_data, file)

            self._save_training_plot(history, training_plot_file_path, input_data.cosine_threshold)
            self.display_result = {
                'file_type': 'png',
                'file_path': training_plot_file_path,
            }

            return OutputModel(
                best_model_file_path=best_model_file_path,
                last_model_file_path=last_model_file_path,
                config_path=config_path,
                training_plot_file_path=training_plot_file_path,
            )
        except Exception as e:
            logger.exception(f'An error occurred during Siamese training: {e}')
            raise e
