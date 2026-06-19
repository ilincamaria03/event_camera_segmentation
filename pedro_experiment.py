from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

from segmentation_model import SegceptionLite, segmentation_loss
from event_representations import Events, get_num_channels, make_representation

PROJECT_DIR = Path(__file__).resolve().parent
RUN_NAME = "pedro_human_binary_dsec_matched"

# PEDRo uses a DAVIS346 camera W=346, H=260.
PEDRO_HEIGHT = 260
PEDRO_WIDTH = 346
IGNORE_INDEX = 255

# PEDRo samples are 40 ms windows, because labels are tied to grayscale frames.
TRAIN_WINDOW_US = 40_000
TEST_WINDOWS_US = [10_000, 20_000, 40_000]

NUM_BINS = 5
RANDOM_SEED = 42
DEFAULT_SEEDS = [42, 43, 44]

DEFAULT_REPRESENTATIONS = ["recent", "voxel", "evsegnet"]
DEFAULT_EPOCHS = 12
DEFAULT_MICRO_BATCH_SIZE = 8
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_LR = 1e-4
DEFAULT_EVAL_EVERY_EPOCHS = 1.0
DEFAULT_NUM_WORKERS = 4
DEFAULT_PREFETCH_FACTOR = 2
DEFAULT_EARLY_STOPPING_PATIENCE_EVALS = 4
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.002
POLY_POWER = 0.9

CLASS_NAMES = ["background", "human_box"]


@dataclass
class Settings:
    pedro_root: Path = PROJECT_DIR / "pedro"
    output_root: Path = PROJECT_DIR
    run_name: str = RUN_NAME

    representations: tuple[str, ...] = tuple(DEFAULT_REPRESENTATIONS)
    seeds: tuple[int, ...] = tuple(DEFAULT_SEEDS)

    epochs: int = DEFAULT_EPOCHS
    micro_batch_size: int = DEFAULT_MICRO_BATCH_SIZE
    grad_accum_steps: int = DEFAULT_GRAD_ACCUM_STEPS
    learning_rate: float = DEFAULT_LR
    eval_every_epochs: float = DEFAULT_EVAL_EVERY_EPOCHS

    train_height: int = PEDRO_HEIGHT
    train_width: int = PEDRO_WIDTH
    model_base: int = 24

    num_workers: int = DEFAULT_NUM_WORKERS
    prefetch_factor: int = DEFAULT_PREFETCH_FACTOR

    early_stopping_patience_evals: int = DEFAULT_EARLY_STOPPING_PATIENCE_EVALS
    early_stopping_min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA
    dice_weight: float = 1.0
    max_class_weight: float = 20.0
    amp: bool = True


# REPRODUCIBILITY

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_reproducible() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def worker_seed(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def count_steps(num_samples: int, micro_batch_size: int, grad_accum_steps: int, epochs: int) -> tuple[int, int]:
    if num_samples <= 0:
        raise ValueError("Empty training split.")
    if micro_batch_size <= 0 or grad_accum_steps <= 0 or epochs <= 0:
        raise ValueError("micro_batch_size, grad_accum_steps and epochs must be positive.")
    micro_batches_per_epoch = math.ceil(num_samples / micro_batch_size)
    optimizer_steps_per_epoch = max(1, math.ceil(micro_batches_per_epoch / grad_accum_steps))
    return int(optimizer_steps_per_epoch * epochs), int(optimizer_steps_per_epoch)


def count_eval_steps(steps_per_epoch: int, eval_every_epochs: float) -> int:
    return max(1, int(round(float(steps_per_epoch) * float(eval_every_epochs))))


def loader_settings(args: Settings, generator: Optional[torch.Generator]) -> dict:
    kwargs = {
        "num_workers": int(args.num_workers),
        "pin_memory": torch.cuda.is_available(),
        "generator": generator,
        "worker_init_fn": worker_seed,
    }
    if int(args.num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(args.prefetch_factor)
    return kwargs


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_run_folder(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    for suffix in range(1, 10_000):
        candidate = base_dir.with_name(f"{base_dir.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find a free run directory based on {base_dir}")


def get_env_info() -> dict:
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }

# DATA LOADING

@dataclass(frozen=True)
class Sample:
    split: str
    numpy_path: Path
    xml_path: Path
    sample_id: str


def check_data_folder(root: Path) -> None:
    missing: list[Path] = []
    for sub in [
        "numpy/train",
        "numpy/val",
        "numpy/test",
        "xml/train",
        "xml/val",
        "xml/test",
    ]:
        p = root / sub
        if not p.exists():
            missing.append(p)
    if missing:
        raise FileNotFoundError("Dataset format is incorrect")


def load_sample_list(root: Path, split: str) -> list[Sample]:
    numpy_dir = root / "numpy" / split
    xml_dir = root / "xml" / split
    npy_paths = sorted(numpy_dir.glob("*.npy"))
    samples: list[Sample] = []
    for npy_path in npy_paths:
        xml_path = xml_dir / f"{npy_path.stem}.xml"
        if xml_path.exists():
            samples.append(Sample(split=split, numpy_path=npy_path, xml_path=xml_path, sample_id=npy_path.stem))

    return samples


def events_from_array(arr: np.ndarray) -> Optional[Events]:
    if arr.dtype.names is None:
        return None
    names = {name.lower(): name for name in arr.dtype.names}
    required = ["t", "x", "y", "p"]
    if not all(k in names for k in required):
        return None
    return Events(
        x=arr[names["x"]].astype(np.int64),
        y=arr[names["y"]].astype(np.int64),
        t=arr[names["t"]].astype(np.int64),
        p=arr[names["p"]].astype(np.int8),
    )


def load_events(numpy_path: Path) -> Events:
    arr = np.load(numpy_path, allow_pickle=True)

    if arr.shape == () and isinstance(arr.item(), dict):
        d = arr.item()
        keymap = {str(k).lower(): k for k in d.keys()}
        return Events(
            x=np.asarray(d[keymap["x"]], dtype=np.int64),
            y=np.asarray(d[keymap["y"]], dtype=np.int64),
            t=np.asarray(d[keymap["t"]], dtype=np.int64),
            p=np.asarray(d[keymap["p"]], dtype=np.int8),
        )

    structured = events_from_array(arr)
    if structured is not None:
        events = structured
    elif arr.ndim == 2 and arr.shape[1] >= 4:
        events = Events(
            x=arr[:, 1].astype(np.int64),
            y=arr[:, 2].astype(np.int64),
            t=arr[:, 0].astype(np.int64),
            p=arr[:, 3].astype(np.int8),
        )
    else:
        raise RuntimeError(
            "Unsupported dataset format"
        )

    if len(events) == 0:
        return events
    order = np.argsort(events.t, kind="stable")
    return Events(x=events.x[order], y=events.y[order], t=events.t[order], p=events.p[order])


def read_boxes(xml_path: Path) -> tuple[list[tuple[int, int, int, int]], int, int]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = PEDRO_WIDTH
    height = PEDRO_HEIGHT
    size = root.find("size")
    if size is not None:
        w_node = size.find("width")
        h_node = size.find("height")
        if w_node is not None and h_node is not None:
            width = int(float(w_node.text))
            height = int(float(h_node.text))

    boxes: list[tuple[int, int, int, int]] = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = int(float(bndbox.findtext("xmin", "0")))
        ymin = int(float(bndbox.findtext("ymin", "0")))
        xmax = int(float(bndbox.findtext("xmax", "0")))
        ymax = int(float(bndbox.findtext("ymax", "0")))
        xmin = max(0, min(width - 1, xmin))
        xmax = max(0, min(width - 1, xmax))
        ymin = max(0, min(height - 1, ymin))
        ymax = max(0, min(height - 1, ymax))
        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes, width, height


def make_box_mask(xml_path: Path, out_height: int = PEDRO_HEIGHT, out_width: int = PEDRO_WIDTH) -> np.ndarray:
    boxes, src_width, src_height = read_boxes(xml_path)
    mask = np.zeros((src_height, src_width), dtype=np.uint8)
    for xmin, ymin, xmax, ymax in boxes:
        mask[ymin:ymax + 1, xmin:xmax + 1] = 1
    if (src_height, src_width) != (out_height, out_width):
        img = Image.fromarray(mask)
        img = img.resize((out_width, out_height), Image.Resampling.NEAREST)
        mask = np.array(img, dtype=np.uint8)
    return mask


def has_box(sample: Sample) -> bool:
    boxes, _, _ = read_boxes(sample.xml_path)
    return len(boxes) > 0


# DATASET PREPROCESSING

def augment(x: torch.Tensor, y: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> tuple[torch.Tensor, torch.Tensor]:
    y_u8 = y.to(torch.uint8).unsqueeze(0)

    if random.random() < 0.5:
        _, H, W = x.shape
        crop_h = int(random.uniform(0.85, 1.0) * H)
        crop_w = int(random.uniform(0.85, 1.0) * W)
        top = random.randint(0, max(H - crop_h, 0))
        left = random.randint(0, max(W - crop_w, 0))
        x = TF.resized_crop(x, top=top, left=left, height=crop_h, width=crop_w,
                            size=[H, W], interpolation=TF.InterpolationMode.BILINEAR)
        y_u8 = TF.resized_crop(y_u8, top=top, left=left, height=crop_h, width=crop_w,
                               size=[H, W], interpolation=TF.InterpolationMode.NEAREST)

    if random.random() < 0.5:
        x = TF.hflip(x)
        y_u8 = TF.hflip(y_u8)

    if random.random() < 0.5:
        angle = random.uniform(-10.0, 10.0)
        x = TF.rotate(x, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        y_u8 = TF.rotate(y_u8, angle, interpolation=TF.InterpolationMode.NEAREST, fill=ignore_index)

    return x, y_u8.squeeze(0).long()


class PedroDataset(Dataset):
    def __init__(self, samples: list[Sample], representation: str, train: bool,
                 train_height: int, train_width: int, window_us: int = TRAIN_WINDOW_US,
                 num_bins: int = NUM_BINS):
        self.samples = samples
        self.representation = representation
        self.train = train
        self.train_height = int(train_height)
        self.train_width = int(train_width)
        self.window_us = int(window_us)
        self.num_bins = int(num_bins)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        events = load_events(sample.numpy_path)

        if len(events) > 0:
            t_end = int(events.t.max())
        else:
            t_end = self.window_us

        rep = make_representation(
            events,
            representation=self.representation,
            t_end=t_end,
            delta_t=self.window_us,
            height=PEDRO_HEIGHT,
            width=PEDRO_WIDTH,
            num_bins=self.num_bins,
            rectify_map=None,
        )
        mask = make_box_mask(sample.xml_path, out_height=PEDRO_HEIGHT, out_width=PEDRO_WIDTH)

        x = torch.from_numpy(rep).float().unsqueeze(0)
        x = F.interpolate(x, size=(self.train_height, self.train_width), mode="bilinear", align_corners=False).squeeze(0)
        y_img = Image.fromarray(mask.astype(np.uint8))
        y_img = y_img.resize((self.train_width, self.train_height), Image.Resampling.NEAREST)
        y = torch.from_numpy(np.array(y_img)).long()

        if self.train:
            x, y = augment(x, y)
        return x, y


# LOSS, METRICS, SAMPLING

class CEDiceLoss(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor], ignore_index: int,
                 dice_class_ids: tuple[int, ...], dice_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.ignore_index = int(ignore_index)
        self.dice_class_ids = tuple(int(c) for c in dice_class_ids)
        self.dice_weight = float(dice_weight)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, target)
        if not self.dice_class_ids or self.dice_weight <= 0:
            return ce_loss
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        probs = torch.softmax(logits, dim=1)
        valid = (target != self.ignore_index).float()
        dice_losses = []
        for class_id in self.dice_class_ids:
            target_c = ((target == class_id).float()) * valid
            prob_c = probs[:, class_id, :, :] * valid
            intersection = (prob_c * target_c).sum()
            denominator = prob_c.sum() + target_c.sum()
            dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
            dice_losses.append(1.0 - dice)
        return ce_loss + self.dice_weight * torch.stack(dice_losses).mean()


def make_confusion(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
                           ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    if target.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.float64)
    indices = target * num_classes + pred
    return torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes).double()


def get_iou(cm: torch.Tensor) -> tuple[np.ndarray, float, float]:
    cm_np = cm.cpu().numpy()
    intersection = np.diag(cm_np)
    gt_sum = cm_np.sum(axis=1)
    pred_sum = cm_np.sum(axis=0)
    union = gt_sum + pred_sum - intersection
    iou = intersection / np.maximum(union, 1e-6)
    valid_classes = gt_sum > 0
    miou = float(np.mean(iou[valid_classes])) if np.any(valid_classes) else 0.0
    accuracy = float(intersection.sum() / max(cm_np.sum(), 1e-6))
    return iou, miou, accuracy


def count_pixels(samples: list[Sample]) -> np.ndarray:
    counts = np.zeros(2, dtype=np.int64)
    for sample in samples:
        mask = make_box_mask(sample.xml_path)
        values, c = np.unique(mask, return_counts=True)
        for value, count in zip(values, c):
            counts[int(value)] += int(count)
    return counts


def make_class_weights(class_counts: np.ndarray, max_weight: float = 20.0) -> torch.Tensor:
    counts = np.maximum(class_counts.astype(np.float64), 1.0)
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.1, max_weight)
    return torch.tensor(weights, dtype=torch.float32)


def poly_lr(current_step: int, total_steps: int, power: float = POLY_POWER) -> float:
    if total_steps <= 0:
        return 1.0
    return max(0.0, (1.0 - float(current_step) / float(total_steps)) ** power)


# TRAINING

@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
             use_amp: bool) -> tuple[float, np.ndarray, float, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    cm_total = torch.zeros((2, 2), dtype=torch.float64)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(x)
            logits = outputs["out"]
            loss = criterion(logits, y)
        pred = torch.argmax(logits, dim=1)
        cm_total += make_confusion(pred.cpu(), y.cpu(), num_classes=2)
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
    iou, miou, acc = get_iou(cm_total)
    return total_loss / max(n, 1), iou, miou, acc


def save_csv(path: Path, rows: list[dict], fieldnames: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def md_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def save_md_table(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No results yet.\n", encoding="utf-8")
        return
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(md_value(row.get(col, "")) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def window_keys(prefix: str) -> list[str]:
    return [f"{prefix}_{int(window_us / 1000)}ms" for window_us in TEST_WINDOWS_US]


def average_seeds(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    metrics = [
        "best_epoch", "best_optimizer_step", "best_val_human_box_iou", "best_val_miou",
    ]
    for prefix in ["test_loss", "test_human_box_iou", "test_background_iou", "test_miou", "test_acc"]:
        metrics.extend(window_keys(prefix))

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row["experiment"]), str(row["representation"]))
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict] = []
    for (experiment, representation), group in sorted(grouped.items()):
        group = sorted(group, key=lambda r: int(r.get("seed", 0)))
        out: dict[str, object] = {
            "dataset": "PEDRo",
            "experiment": experiment,
            "representation": representation,
            "n_seeds": len(group),
            "seeds": ",".join(str(r.get("seed", "")) for r in group),
            "target_epochs": group[0].get("target_epochs", ""),
            "actual_target_epochs": group[0].get("actual_target_epochs", ""),
            "in_channels": group[0].get("in_channels", ""),
            "trainable_parameters": group[0].get("trainable_parameters", ""),
            "primary_metric_label": "human_box IoU",
        }
        for metric in metrics:
            values = [float(r[metric]) for r in group if metric in r and r[metric] not in ("", None)]
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            out[f"{metric}_mean"] = float(arr.mean())
            out[f"{metric}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            out[f"{metric}_min"] = float(arr.min())
            out[f"{metric}_max"] = float(arr.max())
        aggregated.append(out)
    return aggregated


def mean_std_text(row: dict, metric: str) -> str:
    mean = row.get(f"{metric}_mean", "")
    std = row.get(f"{metric}_std", "")
    if mean == "" or std == "":
        return ""
    return f"{float(mean):.4f} ± {float(std):.4f}"


def seed_window_rows(rows: list[dict]) -> list[dict]:
    flat: list[dict] = []
    for row in sorted(rows, key=lambda r: (str(r.get("representation", "")), int(r.get("seed", 0)))):
        for window_us in TEST_WINDOWS_US:
            ms = int(window_us / 1000)
            flat.append({
                "dataset": row.get("dataset", "PEDRo"),
                "experiment": row.get("experiment", "human_box_segmentation"),
                "representation": row.get("representation", ""),
                "seed": row.get("seed", ""),
                "window_ms": ms,
                "human_box_iou": row.get(f"test_human_box_iou_{ms}ms", ""),
                "mIoU": row.get(f"test_miou_{ms}ms", ""),
                "background_iou": row.get(f"test_background_iou_{ms}ms", ""),
                "accuracy": row.get(f"test_acc_{ms}ms", ""),
                "test_loss": row.get(f"test_loss_{ms}ms", ""),
                "best_val_human_box_iou": row.get("best_val_human_box_iou", ""),
                "best_epoch": row.get("best_epoch", ""),
                "best_optimizer_step": row.get("best_optimizer_step", ""),
                "target_epochs": row.get("target_epochs", ""),
                "trainable_parameters": row.get("trainable_parameters", ""),
            })
    return flat


def average_windows(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for row in rows:
        for window_us in TEST_WINDOWS_US:
            ms = int(window_us / 1000)
            grouped.setdefault((str(row.get("experiment", "")), str(row.get("representation", "")), ms), []).append(row)

    out_rows: list[dict] = []
    for (experiment, representation, ms), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])):
        seeds = [str(r.get("seed", "")) for r in sorted(group, key=lambda r: int(r.get("seed", 0)))]
        out: dict[str, object] = {
            "dataset": "PEDRo",
            "experiment": experiment,
            "representation": representation,
            "window_ms": ms,
            "n_seeds": len(group),
            "seeds": ",".join(seeds),
        }
        for source_key, pretty in [
            (f"test_human_box_iou_{ms}ms", "human_box_iou"),
            (f"test_miou_{ms}ms", "mIoU"),
            (f"test_background_iou_{ms}ms", "background_iou"),
            (f"test_acc_{ms}ms", "accuracy"),
            (f"test_loss_{ms}ms", "test_loss"),
        ]:
            values = [float(r[source_key]) for r in group if source_key in r and r[source_key] not in ("", None)]
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            out[f"{pretty}_mean"] = float(arr.mean())
            out[f"{pretty}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            out[f"{pretty}_min"] = float(arr.min())
            out[f"{pretty}_max"] = float(arr.max())
            out[f"{pretty}_mean_std"] = f"{arr.mean():.4f} ± {(arr.std(ddof=1) if len(arr) > 1 else 0.0):.4f}"
        out_rows.append(out)
    return out_rows


def save_seed_average_md(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No aggregate results yet.\n", encoding="utf-8")
        return
    columns = [
        "experiment", "representation", "n", "seeds", "params",
        "human box IoU 40ms", "mIoU 40ms", "accuracy 40ms",
        "human box IoU 10ms", "human box IoU 20ms",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [
            str(row.get("experiment", "")),
            str(row.get("representation", "")),
            str(row.get("n_seeds", "")),
            str(row.get("seeds", "")),
            str(row.get("trainable_parameters", "")),
            mean_std_text(row, "test_human_box_iou_40ms"),
            mean_std_text(row, "test_miou_40ms"),
            mean_std_text(row, "test_acc_40ms"),
            mean_std_text(row, "test_human_box_iou_10ms"),
            mean_std_text(row, "test_human_box_iou_20ms"),
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_window_average_md(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No window mean/std results yet.\n", encoding="utf-8")
        return
    columns = [
        "representation", "window_ms", "n", "seeds",
        "human_box_iou", "mIoU", "background_iou", "accuracy", "test_loss",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [
            str(row.get("representation", "")),
            str(row.get("window_ms", "")),
            str(row.get("n_seeds", "")),
            str(row.get("seeds", "")),
            str(row.get("human_box_iou_mean_std", "")),
            str(row.get("mIoU_mean_std", "")),
            str(row.get("background_iou_mean_std", "")),
            str(row.get("accuracy_mean_std", "")),
            str(row.get("test_loss_mean_std", "")),
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_compact_md(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No compact results yet.\n", encoding="utf-8")
        return
    columns = ["Representation", "Window", "Human IoU", "mIoU", "Accuracy"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append(
            "| " + " | ".join([
                str(row.get("representation", "")),
                f"{row.get('window_ms', '')} ms",
                str(row.get("human_box_iou_mean_std", "")),
                str(row.get("mIoU_mean_std", "")),
                str(row.get("accuracy_mean_std", "")),
            ]) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normal_curve(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    if std <= 1e-12:
        return np.zeros_like(x)
    return (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)


def save_iou_plot(run_dir: Path, rows: list[dict]) -> list[Path]:
    out_paths: list[Path] = []
    if not rows:
        return out_paths
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("representation", "")), []).append(row)

    for representation, group in sorted(grouped.items()):
        windows = [int(w / 1000) for w in TEST_WINDOWS_US]
        means: list[float] = []
        stds: list[float] = []
        values_by_window: list[list[float]] = []
        for ms in windows:
            vals = [float(r[f"test_human_box_iou_{ms}ms"]) for r in group if f"test_human_box_iou_{ms}ms" in r]
            values_by_window.append(vals)
            means.append(float(np.mean(vals)) if vals else float("nan"))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        ax.errorbar(windows, means, yerr=stds, marker="o", capsize=5, linewidth=2.0, label="mean ± std")
        for ms, vals in zip(windows, values_by_window):
            if not vals:
                continue
            offsets = np.linspace(-1.5, 1.5, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), ms) + offsets, vals, s=42, alpha=0.85, label="individual seeds" if ms == windows[0] else None)
        ax.set_xlabel("Event window (ms)")
        ax.set_ylabel("Human-box IoU")
        ax.set_title(f"PEDRo {representation}: Human-box IoU across event windows")
        ax.set_xticks(windows)
        ax.set_ylim(0.0, min(1.0, max(0.05, np.nanmax(np.asarray(means) + np.asarray(stds)) * 1.15)))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        out_path = figures_dir / f"{representation}_human_iou_by_window_mean_std.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def save_iou_curve(run_dir: Path, rows: list[dict], main_window_ms: int = 40) -> list[Path]:
    out_paths: list[Path] = []
    if not rows:
        return out_paths
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("representation", "")), []).append(row)

    for representation, group in sorted(grouped.items()):
        key = f"test_human_box_iou_{int(main_window_ms)}ms"
        vals = np.asarray([float(r[key]) for r in group if key in r and r[key] not in ("", None)], dtype=np.float64)
        if vals.size == 0:
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        if std <= 1e-12:
            x_min = max(0.0, mean - 0.05)
            x_max = min(1.0, mean + 0.05)
        else:
            x_min = max(0.0, min(vals.min(), mean - 3.0 * std))
            x_max = min(1.0, max(vals.max(), mean + 3.0 * std))
            if x_max - x_min < 1e-4:
                x_min, x_max = max(0.0, mean - 0.05), min(1.0, mean + 0.05)
        x = np.linspace(x_min, x_max, 300)
        y = normal_curve(x, mean, std) if std > 1e-12 else np.zeros_like(x)

        fig, ax = plt.subplots(figsize=(7.0, 4.4))
        if std > 1e-12:
            ax.plot(x, y, linewidth=2.0, label="normal curve from mean/std")
            ax.fill_between(x, 0, y, alpha=0.15)
        else:
            ax.axvline(mean, linewidth=2.0, label="all seeds identical")
        for idx, value in enumerate(vals):
            ax.axvline(value, linestyle=":", linewidth=1.4, alpha=0.75, label="seed result" if idx == 0 else None)
        ax.axvline(mean, linewidth=2.0, label=f"mean={mean:.4f}")
        if std > 1e-12:
            ax.axvline(max(0.0, mean - std), linestyle="--", linewidth=1.2, label=f"±1 std={std:.4f}")
            ax.axvline(min(1.0, mean + std), linestyle="--", linewidth=1.2)
        ax.set_xlabel(f"Human-box IoU at {int(main_window_ms)} ms")
        ax.set_ylabel("Fitted density")
        ax.set_title(f"PEDRo {representation}: seed variability, n={vals.size} seeds")
        ax.text(
            0.02, 0.95,
            "Visual guide only: fitted from three seeds",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        out_path = figures_dir / f"{representation}_human_iou_{int(main_window_ms)}ms_bell_curve_std.png"
        fig.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out_path)
    return out_paths


def save_figure_list(run_dir: Path, paths: list[Path]) -> None:
    if not paths:
        return
    lines = ["# PEDRo figures", ""]
    for path in sorted(paths):
        try:
            lines.append(str(path.relative_to(run_dir)))
        except ValueError:
            lines.append(str(path))
    (run_dir / "figures_manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_outputs(run_dir: Path, rows: list[dict]) -> None:
    per_seed_columns = [
        "dataset", "experiment", "representation", "seed", "seed_index", "target_epochs",
        "best_epoch", "best_optimizer_step", "in_channels", "trainable_parameters",
        "test_human_box_iou_40ms", "test_miou_40ms", "test_background_iou_40ms", "test_acc_40ms", "test_loss_40ms",
        "test_human_box_iou_10ms", "test_human_box_iou_20ms",
    ]
    save_csv(run_dir / "summary_results_by_seed.csv", rows)
    save_csv(run_dir / "summary_results.csv", rows)
    save_md_table(run_dir / "summary_results_by_seed.md", rows, per_seed_columns)

    # Main table requested by the user: every seed and every time interval in one place.
    flat_rows = seed_window_rows(rows)
    flat_columns = [
        "dataset", "experiment", "representation", "seed", "window_ms",
        "human_box_iou", "mIoU", "background_iou", "accuracy", "test_loss",
        "best_val_human_box_iou", "best_epoch", "best_optimizer_step", "target_epochs", "trainable_parameters",
    ]
    save_csv(run_dir / "all_metrics_by_seed_and_window.csv", flat_rows, fieldnames=flat_columns)
    save_md_table(run_dir / "all_metrics_by_seed_and_window.md", flat_rows, flat_columns)

    window_rows = average_windows(rows)
    save_csv(run_dir / "mean_std_by_window.csv", window_rows)
    save_window_average_md(run_dir / "mean_std_by_window.md", window_rows)
    save_csv(run_dir / "compact_table.csv", window_rows)
    save_compact_md(run_dir / "compact_table.md", window_rows)

    aggregated = average_seeds(rows)
    save_csv(run_dir / "summary_results_mean_std.csv", aggregated)
    save_seed_average_md(run_dir / "summary_results_mean_std.md", aggregated)

    fig_paths: list[Path] = []
    fig_paths.extend(save_iou_plot(run_dir, rows))
    fig_paths.extend(save_iou_curve(run_dir, rows, main_window_ms=40))
    save_figure_list(run_dir, fig_paths)


def make_loaders(representation: str, train_samples: list[Sample], val_samples: list[Sample],
                  args: Settings, generator: Optional[torch.Generator]) -> tuple[DataLoader, DataLoader]:
    train_dataset = PedroDataset(
        train_samples, representation=representation, train=True,
        train_height=args.train_height, train_width=args.train_width, window_us=TRAIN_WINDOW_US, num_bins=NUM_BINS,
    )
    val_dataset = PedroDataset(
        val_samples, representation=representation, train=False,
        train_height=args.train_height, train_width=args.train_width, window_us=TRAIN_WINDOW_US, num_bins=NUM_BINS,
    )
    loader_kwargs = loader_settings(args, generator)
    train_loader = DataLoader(
        train_dataset, batch_size=args.micro_batch_size, shuffle=True, sampler=None,
        drop_last=False, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.micro_batch_size, shuffle=False,
        drop_last=False, **loader_kwargs,
    )
    return train_loader, val_loader


def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, criterion: nn.Module,
                optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR,
                device: torch.device, args: Settings, out_dir: Path, metadata: dict,
                target_optimizer_steps: int, eval_every_steps: int, use_amp: bool) -> dict:
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    metrics_rows: list[dict] = []
    best_human_iou = -1.0
    best_record: dict = {}
    evals_without_improvement = 0
    recent_losses: list[float] = []
    train_iter = iter(train_loader)

    for opt_step in range(1, target_optimizer_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(x)
                loss = segmentation_loss(outputs, y, criterion, aux_weight=0.4)
                scaled_loss = loss / args.grad_accum_steps
            scaler.scale(scaled_loss).backward()
            accum_loss += float(loss.item())

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        recent_losses.append(accum_loss / args.grad_accum_steps)
        if len(recent_losses) > eval_every_steps:
            recent_losses = recent_losses[-eval_every_steps:]

        should_eval = (opt_step % eval_every_steps == 0) or (opt_step == target_optimizer_steps)
        if should_eval:
            val_loss, val_iou, val_miou, val_acc = eval_model(model, val_loader, criterion, device, use_amp=use_amp)
            human_iou = float(val_iou[1])
            row = {
                "optimizer_step": opt_step,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss_recent": float(np.mean(recent_losses)),
                "val_loss": val_loss,
                "val_miou": val_miou,
                "val_human_box_iou": human_iou,
                "val_acc": val_acc,
                "val_background_iou": float(val_iou[0]),
            }
            metrics_rows.append(row)
            save_csv(out_dir / "val_metrics.csv", metrics_rows)
            print(
                f"step={opt_step:05d}/{target_optimizer_steps} "
                f"train_loss={row['train_loss_recent']:.4f} "
                f"val_human_box_iou={human_iou:.4f} val_miou={val_miou:.4f} val_acc={val_acc:.4f}"
            )
            if human_iou > best_human_iou + float(args.early_stopping_min_delta):
                best_human_iou = human_iou
                best_record = {
                    **metadata,
                    "best_optimizer_step": int(opt_step),
                    "best_val_human_box_iou": human_iou,
                    "best_val_miou": float(val_miou),
                    "best_val_acc": float(val_acc),
                    "best_val_iou": val_iou.tolist(),
                    "class_names": CLASS_NAMES,
                    "target_type": "binary box mask from PEDRo bounding boxes",
                    "primary_metric_label": "human_box IoU",
                    "early_stopping_patience_evals": int(args.early_stopping_patience_evals),
                    "early_stopping_min_delta": float(args.early_stopping_min_delta),
                }
                torch.save({"model": model.state_dict(), **best_record}, out_dir / "best.pt")

                evals_without_improvement = 0
            else:
                evals_without_improvement += 1

            if int(args.early_stopping_patience_evals) > 0 and evals_without_improvement >= int(args.early_stopping_patience_evals):
                print(
                    f"early stopping at step={opt_step:05d}: "
                    f"no val_human_box_iou improvement > {args.early_stopping_min_delta} "
                    f"for {args.early_stopping_patience_evals} evaluations"
                )
                break

    if not best_record:
        raise RuntimeError("No val record was produced; reduce eval_every_steps or check val split.")
    return best_record


# VISUALIZATION

def normalize_image(image: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    image = image.astype(np.float32)
    high = np.percentile(image, percentile)
    if high <= 1e-6:
        return image
    return np.clip(image / high, 0.0, 1.0)


def make_preview(x: torch.Tensor, representation: str) -> np.ndarray:
    x_np = x.detach().cpu().numpy()
    if representation == "count":
        image = x_np[0]
    elif representation in {"polarity", "recent", "evsegnet"}:
        image = x_np[0] + x_np[1]
    elif representation == "time_surface":
        image = np.max(x_np, axis=0)
    else:
        image = np.sum(x_np, axis=0)
    return normalize_image(image)


def first_box_index(dataset: PedroDataset) -> int:
    for i, sample in enumerate(dataset.samples):
        if has_box(sample):
            return i
    return 0


@torch.no_grad()
def save_prediction(model: nn.Module, dataset: PedroDataset, index: int,
                                  device: torch.device, out_path: Path, title: str, use_amp: bool) -> None:
    model.eval()
    x, y = dataset[index]
    with torch.autocast(device_type=device.type, enabled=use_amp):
        logits = model(x.unsqueeze(0).to(device))["out"]
    pred = torch.argmax(logits, dim=1)[0].cpu().numpy()
    gt = y.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(make_preview(x, dataset.representation), cmap="gray")
    axes[0].set_title("event input preview")
    axes[1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("box-mask target")
    axes[2].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("prediction")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# EXPERIMENT EXECUTION

def test_windows(model: nn.Module, representation: str, test_samples: list[Sample],
                          criterion: nn.Module, args: Settings, device: torch.device,
                          use_amp: bool) -> list[dict]:
    rows: list[dict] = []
    for window_us in TEST_WINDOWS_US:
        dataset = PedroDataset(
            test_samples, representation=representation, train=False,
            train_height=args.train_height, train_width=args.train_width,
            window_us=window_us, num_bins=NUM_BINS,
        )
        loader = DataLoader(dataset, batch_size=args.micro_batch_size, shuffle=False,
                            **loader_settings(args, None))
        test_loss, test_iou, test_miou, test_acc = eval_model(model, loader, criterion, device, use_amp=use_amp)
        rows.append({
            "window_us": int(window_us),
            "window_ms": float(window_us / 1000.0),
            "test_loss": test_loss,
            "test_miou": test_miou,
            "test_human_box_iou": float(test_iou[1]),
            "test_background_iou": float(test_iou[0]),
            "test_acc": test_acc,
        })
    return rows


def run_one_rep(representation: str, train_samples: list[Sample], val_samples: list[Sample],
                       test_samples: list[Sample], args: Settings, device: torch.device,
                       run_dir: Path, seed: int, seed_index: int) -> dict:
    seed = int(seed)
    set_seed(seed)
    generator = make_generator(seed)

    out_dir = run_dir / representation / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_summary_path = out_dir / "result_summary.json"

    train_counts = count_pixels(train_samples)
    val_counts = count_pixels(val_samples)
    test_counts = count_pixels(test_samples)

    distribution_rows = []
    for split, counts in [("train", train_counts), ("val", val_counts), ("test", test_counts)]:
        total = int(counts.sum())
        for class_id, class_name in enumerate(CLASS_NAMES):
            distribution_rows.append({
                "split": split,
                "class_id": class_id,
                "class_name": class_name,
                "pixels": int(counts[class_id]),
                "ratio": float(counts[class_id] / max(total, 1)),
            })
    save_csv(out_dir / "pixel_distribution.csv", distribution_rows)

    in_channels = get_num_channels(representation, NUM_BINS)
    model = SegceptionLite(in_channels=in_channels, num_classes=2, base=args.model_base, aux=True).to(device)
    trainable_parameters = count_params(model)
    class_weights = make_class_weights(train_counts, max_weight=args.max_class_weight).to(device)
    criterion = CEDiceLoss(class_weights, IGNORE_INDEX, dice_class_ids=(1,), dice_weight=args.dice_weight)
    requested_epochs = int(args.epochs)
    computed_target_steps, steps_per_epoch = count_steps(
        num_samples=len(train_samples),
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        epochs=requested_epochs,
    )
    target_optimizer_steps = computed_target_steps
    eval_every_steps = count_eval_steps(steps_per_epoch, args.eval_every_epochs)
    actual_target_epochs = float(target_optimizer_steps / max(steps_per_epoch, 1))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: poly_lr(step, target_optimizer_steps, POLY_POWER)
    )
    train_loader, val_loader = make_loaders(representation, train_samples, val_samples, args, generator)

    metadata = {
        "dataset": "PEDRo",
        "task": "human binary segmentation using bounding-box pseudo masks",
        "representation": representation,
        "seed": seed,
        "seed_index": seed_index,
        "class_names": CLASS_NAMES,
        "input_height": PEDRO_HEIGHT,
        "input_width": PEDRO_WIDTH,
        "train_height": args.train_height,
        "train_width": args.train_width,
        "train_window_us": TRAIN_WINDOW_US,
        "test_windows_us": TEST_WINDOWS_US,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        "in_channels": in_channels,
        "trainable_parameters": trainable_parameters,
        "micro_batch_size": args.micro_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.micro_batch_size * args.grad_accum_steps,
        "target_epochs": requested_epochs,
        "actual_target_epochs": actual_target_epochs,
        "steps_per_epoch": steps_per_epoch,
        "target_optimizer_steps": target_optimizer_steps,
        "eval_every_steps": eval_every_steps,
        "eval_every_epochs": args.eval_every_epochs,
        "learning_rate": args.learning_rate,
        "model_base": args.model_base,
        "class_weights": class_weights.detach().cpu().tolist(),
        "dice_weight": args.dice_weight,
        "frame_filtering": "none",
        "frame_sampling": "none",
        "early_stopping_patience_evals": args.early_stopping_patience_evals,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "primary_metric_label": "human_box IoU",
    }
    with (out_dir / "run_parameters.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n--- PEDRo human box-mask segmentation | {representation} | seed={seed} ---")
    print(f"train/val/test samples={len(train_samples)}/{len(val_samples)}/{len(test_samples)}; channels={in_channels}; trainable_parameters={trainable_parameters}")
    print(f"class_weights={metadata['class_weights']}")

    use_amp = bool(args.amp and device.type == "cuda")
    best_record = train(model, train_loader, val_loader, criterion, optimizer, scheduler,
                              device, args, out_dir, metadata,
                              target_optimizer_steps=target_optimizer_steps,
                              eval_every_steps=eval_every_steps,
                              use_amp=use_amp)

    checkpoint = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    test_rows = test_windows(model, representation, test_samples, criterion, args, device, use_amp=use_amp)
    save_csv(out_dir / "test_metrics_by_window.csv", test_rows)

    row_40 = next(row for row in test_rows if row["window_us"] == TRAIN_WINDOW_US)
    test_dataset_40 = PedroDataset(
        test_samples, representation=representation, train=False,
        train_height=args.train_height, train_width=args.train_width,
        window_us=TRAIN_WINDOW_US, num_bins=NUM_BINS,
    )
    if len(test_dataset_40) > 0:
        index = first_box_index(test_dataset_40)
        save_prediction(
            model, test_dataset_40, index, device,
            out_dir / "prediction_example_40ms.png",
            title=f"PEDRo human box mask | {representation} | sample {index}",
            use_amp=use_amp,
        )

    result = {
        "dataset": "PEDRo",
        "experiment": "human_box_segmentation",
        "representation": representation,
        "seed": seed,
        "seed_index": seed_index,
        "target_epochs": requested_epochs,
        "actual_target_epochs": actual_target_epochs,
        "steps_per_epoch": steps_per_epoch,
        "in_channels": in_channels,
        "trainable_parameters": trainable_parameters,
        "best_optimizer_step": int(best_record["best_optimizer_step"]),
        "best_epoch": float(best_record["best_optimizer_step"] / max(steps_per_epoch, 1)),
        "best_val_human_box_iou": float(best_record["best_val_human_box_iou"]),
        "best_val_miou": float(best_record["best_val_miou"]),
        "test_loss_40ms": float(row_40["test_loss"]),
        "test_human_box_iou_40ms": float(row_40["test_human_box_iou"]),
        "test_background_iou_40ms": float(row_40["test_background_iou"]),
        "test_miou_40ms": float(row_40["test_miou"]),
        "test_acc_40ms": float(row_40["test_acc"]),
        "checkpoint_path": str(out_dir / "best.pt"),
        "figure_path": str(out_dir / "prediction_example_40ms.png"),
        "primary_metric_label": "human_box IoU",
    }
    for row in test_rows:
        ms = int(row["window_ms"])
        result[f"test_loss_{ms}ms"] = float(row["test_loss"])
        result[f"test_human_box_iou_{ms}ms"] = float(row["test_human_box_iou"])
        result[f"test_background_iou_{ms}ms"] = float(row["test_background_iou"])
        result[f"test_miou_{ms}ms"] = float(row["test_miou"])
        result[f"test_acc_{ms}ms"] = float(row["test_acc"])

    with result_summary_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        f"DONE PEDRo | {representation} | seed={seed}: "
        f"best_step={result['best_optimizer_step']} "
        f"val_human_box_iou={result['best_val_human_box_iou']:.4f} "
        f"test_human_box_iou_40ms={result['test_human_box_iou_40ms']:.4f} "
        f"test_miou_40ms={result['test_miou_40ms']:.4f} "
        f"test_acc_40ms={result['test_acc_40ms']:.4f}"
    )
    return result


# MAIN

def main() -> None:
    args = Settings()
    set_seed(RANDOM_SEED)
    set_reproducible()

    representations = list(args.representations)
    seeds = list(args.seeds)

    check_data_folder(args.pedro_root)
    train_samples = load_sample_list(args.pedro_root, "train")
    val_samples = load_sample_list(args.pedro_root, "val")
    test_samples = load_sample_list(args.pedro_root, "test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    output_root = args.output_root.expanduser().resolve()
    run_dir = make_run_folder(output_root / "runs" / args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "run_name": args.run_name,
        "resolved_run_dir": str(run_dir),
        "pedro_root": str(args.pedro_root),
        "output_root": str(output_root),
        "device": str(device),
        "use_amp": use_amp,
        "representations": representations,
        "seeds": seeds,
        "task": "PEDRo human-only binary segmentation with bounding-box pseudo masks",
        "warning": "PEDRo labels are bounding boxes, not true silhouette masks; human IoU is box-mask IoU.",
        "train_window_us": TRAIN_WINDOW_US,
        "test_windows_us": TEST_WINDOWS_US,
        "train_height": args.train_height,
        "train_width": args.train_width,
        "epochs": args.epochs,
        "micro_batch_size": args.micro_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.micro_batch_size * args.grad_accum_steps,
        "learning_rate": args.learning_rate,
        "model_base": args.model_base,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "eval_every_epochs": args.eval_every_epochs,
        "early_stopping_patience_evals": args.early_stopping_patience_evals,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "default_random_seed": RANDOM_SEED,
        "frame_filtering": "none",
        "frame_sampling": "none",
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "test_samples": len(test_samples),
        **get_env_info(),
    }
    with (run_dir / "pedro_run_parameters.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print("=" * 80)
    print("PEDRo HUMAN-ONLY BOX-MASK SEGMENTATION RUN")
    print("=" * 80)
    print(json.dumps(params, indent=2))

    all_results: list[dict] = []
    for representation in representations:
        for seed_index, seed in enumerate(seeds):
            result = run_one_rep(
                representation, train_samples, val_samples, test_samples,
                args=args, device=device, run_dir=run_dir, seed=seed, seed_index=seed_index,
            )
            all_results.append(result)
            save_outputs(run_dir, all_results)

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    for result in average_seeds(all_results):
        print(
            f"{result['representation']:>12s} | "
            f"n={result['n_seeds']} | seeds={result['seeds']} | "
            f"params={result['trainable_parameters']} | "
            f"human_box_iou_40ms={mean_std_text(result, 'test_human_box_iou_40ms')} | "
            f"miou_40ms={mean_std_text(result, 'test_miou_40ms')} | "
            f"acc_40ms={mean_std_text(result, 'test_acc_40ms')}"
        )
    print(f"\nSaved all outputs under: {run_dir}")
    print(f"Main table: {run_dir / 'all_metrics_by_seed_and_window.csv'}")
    print(f"Mean/std table: {run_dir / 'mean_std_by_window.csv'}")
    print(f"Compact table: {run_dir / 'compact_table.md'}")
    print(f"Figures: {run_dir / 'figures'}")


if __name__ == "__main__":
    main()
