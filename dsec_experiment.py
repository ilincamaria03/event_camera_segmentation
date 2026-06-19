from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import hdf5plugin
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
DEFAULT_DATA_ROOT = PROJECT_DIR / "dsec_seg"
DEFAULT_OUTPUT_ROOT = PROJECT_DIR
RUN_NAME = "dsec_segmentation_comparison"

SEQUENCE_IDS = ["00", "06", "07", "08", "13"]
TRAIN_SEQUENCE_IDS = ["00", "06", "07"]
VAL_SEQUENCE_IDS = ["08"]
TEST_SEQUENCE_IDS = ["13"]

LABEL_HEIGHT = 440
WIDTH = 640
IGNORE_INDEX = 255

TRAIN_WINDOW_US = 50_000
TEST_WINDOWS_US = [10_000, 50_000, 250_000]
NUM_BINS = 5
RANDOM_SEED = 42
DEFAULT_SEEDS = [42, 43, 44]

DEFAULT_REPRESENTATIONS = ["recent", "voxel", "evsegnet"]

DEFAULT_MULTICLASS_EPOCHS = 12
DEFAULT_HUMAN_EPOCHS = 12
DEFAULT_MICRO_BATCH_SIZE = 8
DEFAULT_GRAD_ACCUM_STEPS = 1
DEFAULT_LR = 1e-4
POLY_POWER = 0.9

DEFAULT_EVAL_EVERY_EPOCHS = 1.0
DEFAULT_EARLY_STOPPING_PATIENCE_EVALS = 4
DEFAULT_EARLY_STOPPING_MIN_DELTA = 0.002
DEFAULT_NUM_WORKERS = 4
DEFAULT_PREFETCH_FACTOR = 2


@dataclass
class Params:
    data_root: Path = DEFAULT_DATA_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    cache_root: Optional[Path] = None
    cache_dtype: str = "uint8"
    run_name: str = RUN_NAME

    reps: tuple[str, ...] = tuple(DEFAULT_REPRESENTATIONS)
    seeds: tuple[int, ...] = tuple(DEFAULT_SEEDS)

    epochs: Optional[int] = None
    batch_size: int = DEFAULT_MICRO_BATCH_SIZE
    accum_steps: int = DEFAULT_GRAD_ACCUM_STEPS
    lr: float = DEFAULT_LR
    eval_every: float = DEFAULT_EVAL_EVERY_EPOCHS

    img_h: int = LABEL_HEIGHT
    img_w: int = WIDTH
    model_base: int = 24

    patience: int = DEFAULT_EARLY_STOPPING_PATIENCE_EVALS
    min_delta: float = DEFAULT_EARLY_STOPPING_MIN_DELTA
    workers: int = DEFAULT_NUM_WORKERS
    prefetch: int = DEFAULT_PREFETCH_FACTOR
    amp: bool = True


# REPRODUCIBILITY

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_repeatable() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def make_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def calc_steps(num_samples: int, batch_size: int, accum_steps: int, epochs: int) -> tuple[int, int]:
    if num_samples <= 0:
        raise ValueError("Empty training split.")
    if batch_size <= 0 or accum_steps <= 0 or epochs <= 0:
        raise ValueError("batch_size, accum_steps and epochs must be positive.")
    batches_epoch = math.ceil(num_samples / batch_size)
    steps_epoch = max(1, math.ceil(batches_epoch / accum_steps))
    return int(steps_epoch * epochs), int(steps_epoch)


def calc_eval_steps(steps_epoch: int, eval_every: float) -> int:
    return max(1, int(round(float(steps_epoch) * float(eval_every))))


def loader_opts(args: Params, generator: Optional[torch.Generator]) -> dict:
    kwargs = {
        "num_workers": int(args.workers),
        "pin_memory": torch.cuda.is_available(),
        "generator": generator,
        "worker_init_fn": seed_worker,
    }
    if int(args.workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(args.prefetch)
    return kwargs


def num_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def new_run_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    for suffix in range(1, 10_000):
        candidate = base_dir.with_name(f"{base_dir.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Couldn't find a free run directory based on {base_dir}")


def system_info() -> dict:
    return {
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }

DATA_ROOT = DEFAULT_DATA_ROOT

def set_root(path: str | Path) -> None:
    global DATA_ROOT, SEQUENCE_PATHS
    DATA_ROOT = Path(path).expanduser().resolve()
    SEQUENCE_PATHS = {seq_id: make_paths(seq_id) for seq_id in SEQUENCE_IDS}

@dataclass(frozen=True)
class Paths:
    seq: str
    root_dir: Path
    event_h5: Path
    rectify_map_h5: Path
    semantic_dir: Path
    label_dir: Path
    timestamps_txt: Path


def make_paths(seq: str) -> Paths:
    root_dir = DATA_ROOT / f"zurich_{seq}"
    city_name = f"zurich_city_{seq}_a"
    event_dir = root_dir / f"{city_name}_events_left"
    semantic_outer_dir = root_dir / f"{city_name}_semantic"
    semantic_dir = semantic_outer_dir / city_name
    return Paths(
        seq=seq,
        root_dir=root_dir,
        event_h5=event_dir / "events.h5",
        rectify_map_h5=event_dir / "rectify_map.h5",
        semantic_dir=semantic_dir,
        label_dir=semantic_dir / "19classes",
        timestamps_txt=semantic_dir / f"{city_name}_semantic_timestamps.txt",
    )


SEQUENCE_PATHS = {seq_id: make_paths(seq_id) for seq_id in SEQUENCE_IDS}


# DSEC LABELS AND EXPERIMENTS

DSEC_19_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
]
HUMAN_DSEC_IDS = [11, 12]


@dataclass(frozen=True)
class Labels:
    name: str
    class_names: list[str]
    remap_fn: Callable[[np.ndarray], np.ndarray]
    primary_metric_class_id: Optional[int] = None
    primary_metric_label: str = "mIoU"
    use_dice_loss: bool = False
    dice_class_ids: tuple[int, ...] = ()

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


@dataclass(frozen=True)
class Experiment:
    name: str
    labels: Labels
    default_epochs: int = DEFAULT_MULTICLASS_EPOCHS
    max_class_weight: float = 20.0
    dice_weight: float = 1.0


EVSEGNET_CATEGORY_CLASS_NAMES = [
    "flat",        # road + sidewalk
    "background",  # construction/background + sky
    "object",      # pole + traffic light + traffic sign
    "vegetation",  # vegetation + terrain
    "human",       # person + rider
    "vehicle",     # car/truck/bus/train/motorcycle/bicycle
]


def map_multi(mask19: np.ndarray) -> np.ndarray:
    # DSEC 19 classes -> EV-SegNet-style urban categories, keeping human as its own class.
    out = np.zeros(mask19.shape, dtype=np.uint8)
    out[np.isin(mask19, [0, 1])] = 0                         # flat
    out[np.isin(mask19, [2, 3, 4, 10])] = 1                  # background: construction + sky
    out[np.isin(mask19, [5, 6, 7])] = 2                      # object
    out[np.isin(mask19, [8, 9])] = 3                         # vegetation/nature
    out[np.isin(mask19, HUMAN_DSEC_IDS)] = 4                 # human
    out[np.isin(mask19, [13, 14, 15, 16, 17, 18])] = 5       # vehicle
    out[mask19 == IGNORE_INDEX] = IGNORE_INDEX
    return out


HUMAN_CLASS_NAMES = ["background", "human"]


def map_human(mask19: np.ndarray) -> np.ndarray:
    out = np.zeros(mask19.shape, dtype=np.uint8)
    out[np.isin(mask19, HUMAN_DSEC_IDS)] = 1
    out[mask19 == IGNORE_INDEX] = IGNORE_INDEX
    return out


MULTICLASS_LABEL_SPEC = Labels(
    name="multiclass",
    class_names=EVSEGNET_CATEGORY_CLASS_NAMES,
    remap_fn=map_multi,
    primary_metric_class_id=None,
    primary_metric_label="mIoU",
    use_dice_loss=False,
)

HUMAN_LABEL_SPEC = Labels(
    name="human",
    class_names=HUMAN_CLASS_NAMES,
    remap_fn=map_human,
    primary_metric_class_id=1,
    primary_metric_label="human IoU",
    use_dice_loss=True,
    dice_class_ids=(1,),
)

EXPERIMENTS = [
    Experiment(
        name="multiclass_with_human",
        labels=MULTICLASS_LABEL_SPEC,
        default_epochs=DEFAULT_MULTICLASS_EPOCHS,
        max_class_weight=20.0,
    ),
    Experiment(
        name="human_binary",
        labels=HUMAN_LABEL_SPEC,
        default_epochs=DEFAULT_HUMAN_EPOCHS,
        max_class_weight=20.0,
        dice_weight=1.0,
    ),
]


def label_files(label_dir: str | Path) -> list[Path]:
    paths = sorted(Path(label_dir).glob("*.png"))
    if not paths:
        raise RuntimeError(f"No PNG labels found in {label_dir}")
    return paths


def load_times(path: str | Path) -> np.ndarray:
    timestamps = np.loadtxt(path, dtype=np.int64)
    timestamps = np.atleast_1d(timestamps)
    if timestamps.ndim == 2:
        timestamps = timestamps[:, -1]
    return timestamps.astype(np.int64)


def load_mask(path: str | Path) -> np.ndarray:
    mask = np.array(Image.open(path), dtype=np.int64)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask[:LABEL_HEIGHT, :WIDTH]


def find_rectify_data(h5_obj: h5py.Group) -> Optional[np.ndarray]:
    for key in h5_obj.keys():
        item = h5_obj[key]
        if isinstance(item, h5py.Dataset) and item.ndim == 3 and item.shape[-1] == 2:
            return item[:]
        if isinstance(item, h5py.Group):
            found = find_rectify_data(item)
            if found is not None:
                return found
    return None


def get_rectify_map(path: str | Path) -> np.ndarray:
    path = Path(path)
    with h5py.File(path, "r") as f:
        rectify_map = find_rectify_data(f)
    if rectify_map is None:
        raise RuntimeError(f"Couldn't find a correct rectify map dataset inside {path}")
    return rectify_map.astype(np.float32)


def find_h5_data(h5_obj: h5py.Group, dataset_name: str) -> Optional[h5py.Dataset]:
    for key in h5_obj.keys():
        item = h5_obj[key]
        if isinstance(item, h5py.Dataset) and key == dataset_name:
            return item
        if isinstance(item, h5py.Group):
            found = find_h5_data(item, dataset_name)
            if found is not None:
                return found
    return None


# EVENT READER

class EventSeq:

    def __init__(self, paths: Paths):
        self.paths = paths
        self.h5 = h5py.File(paths.event_h5, "r")
        self.x_ds = self.h5["events/x"]
        self.y_ds = self.h5["events/y"]
        self.t_ds = self.h5["events/t"]
        self.p_ds = self.h5["events/p"]
        self.t_offset = int(self.h5["t_offset"][()]) if "t_offset" in self.h5 else 0
        ms_to_idx_ds = find_h5_data(self.h5, "ms_to_idx")
        self.ms_to_idx = ms_to_idx_ds[:] if ms_to_idx_ds is not None else None
        self.rectify_map = get_rectify_map(paths.rectify_map_h5)

    def close(self) -> None:
        try:
            self.h5.close()
        except Exception:
            pass

    def __del__(self) -> None:
        self.close()

    def index_from_map(self, t_start_rel: int, t_end_rel: int) -> tuple[int, int]:
        assert self.ms_to_idx is not None
        n_events = len(self.t_ds)
        if n_events == 0:
            return 0, 0

        start_ms = max(int(np.floor(t_start_rel / 1000.0)) - 1, 0)
        end_ms = max(int(np.ceil(t_end_rel / 1000.0)) + 2, start_ms + 1)
        end_ms = min(end_ms, len(self.ms_to_idx) - 1)
        i0 = int(np.clip(int(self.ms_to_idx[start_ms]), 0, n_events))
        i1 = int(np.clip(int(self.ms_to_idx[end_ms]), i0, n_events))
        if i1 <= i0:
            i1 = min(i0 + 1, n_events)
        t_local = self.t_ds[i0:i1]
        local_start = int(np.searchsorted(t_local, t_start_rel, side="left"))
        local_end = int(np.searchsorted(t_local, t_end_rel, side="right"))
        return i0 + local_start, i0 + local_end

    def index_from_search(self, t_start_rel: int, t_end_rel: int) -> tuple[int, int]:
        t_all = self.t_ds[:]
        return (
            int(np.searchsorted(t_all, t_start_rel, side="left")),
            int(np.searchsorted(t_all, t_end_rel, side="right")),
        )

    def get_window(self, t_end_abs: int, delta_t: int) -> Events:
        t_start_abs = int(t_end_abs - delta_t)
        t_start_rel = int(t_start_abs - self.t_offset)
        t_end_rel = int(t_end_abs - self.t_offset)
        if t_end_rel < 0:
            return Events(np.empty(0, dtype=np.int16), np.empty(0, dtype=np.int16), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8))
        t_start_rel = max(t_start_rel, 0)
        if self.ms_to_idx is not None:
            start_idx, end_idx = self.index_from_map(t_start_rel, t_end_rel)
        else:
            start_idx, end_idx = self.index_from_search(t_start_rel, t_end_rel)
        if end_idx <= start_idx:
            return Events(np.empty(0, dtype=np.int16), np.empty(0, dtype=np.int16), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int8))
        return Events(
            x=self.x_ds[start_idx:end_idx],
            y=self.y_ds[start_idx:end_idx],
            t=self.t_ds[start_idx:end_idx].astype(np.int64) + self.t_offset,
            p=self.p_ds[start_idx:end_idx],
        )


# SAMPLE BUILDING

@dataclass(frozen=True)
class Sample:
    seq: str
    label_path: Path
    timestamp: int


def check_files(seqs: list[str]) -> None:
    missing: list[Path] = []
    for seq in seqs:
        paths = SEQUENCE_PATHS[seq]
        for p in [paths.event_h5, paths.rectify_map_h5, paths.label_dir, paths.timestamps_txt]:
            if not p.exists():
                missing.append(p)
    if missing:
        raise FileNotFoundError("Missing required DSEC files/folders:\n" + "\n".join(str(p) for p in missing))


def build_seq_samples(seq: str) -> list[Sample]:
    paths = SEQUENCE_PATHS[seq]
    label_paths = label_files(paths.label_dir)
    timestamps = load_times(paths.timestamps_txt)
    if len(label_paths) != len(timestamps):
        raise RuntimeError(f"Mismatch for zurich_{seq}: {len(label_paths)} labels but {len(timestamps)} timestamps.")

    return [Sample(seq=seq, label_path=label_paths[i], timestamp=int(timestamps[i])) for i in range(len(label_paths))]


def build_samples(seqs: list[str]) -> list[Sample]:
    samples: list[Sample] = []
    for seq in seqs:
        samples.extend(build_seq_samples(seq))
    return samples


# REPRESENTATION AND LABEL CACHING

def samples_hash(samples: list[Sample]) -> str:
    hasher = hashlib.sha1()
    for sample in samples:
        token = f"{sample.seq}|{sample.label_path.name}|{int(sample.timestamp)}\n"
        hasher.update(token.encode("utf-8"))
    return hasher.hexdigest()


def cache_type(name: str) -> np.dtype:
    name = str(name).lower().strip()
    if name in {"uint8", "u8", "byte"}:
        return np.dtype(np.uint8)
    if name in {"float16", "fp16", "half"}:
        return np.dtype(np.float16)
    if name in {"float32", "fp32", "single"}:
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported cache dtype: {name}. Use uint8, float16 or float32.")


def pack_rep(rep: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.dtype(dtype) == np.dtype(np.uint8):
        return np.rint(np.clip(rep, 0.0, 1.0) * 255.0).astype(np.uint8, copy=False)
    return rep.astype(dtype, copy=False)


def unpack_rep(rep: np.ndarray) -> np.ndarray:
    if rep.dtype == np.uint8:
        return rep.astype(np.float32) / 255.0
    return rep.astype(np.float32, copy=False)


def cache_ok(meta_path: Path, expected: dict) -> bool:
    if not meta_path.exists():
        return False
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            actual = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    keys = [
        "kind", "sample_signature", "sample_count", "shape", "dtype",
        "representation", "window_us", "height", "width", "num_bins", "label_spec",
    ]
    return all(actual.get(key) == expected.get(key) for key in keys if key in expected)


def resize_rep(rep: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    if rep.shape[-2:] == (int(img_h), int(img_w)):
        return rep.astype(np.float32, copy=False)
    x = torch.from_numpy(rep).float().unsqueeze(0)
    x = F.interpolate(x, size=(int(img_h), int(img_w)), mode="bilinear", align_corners=False)
    return x.squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def resize_label(mask: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    if mask.shape == (int(img_h), int(img_w)):
        return mask.astype(np.uint8, copy=False)
    y_img = Image.fromarray(mask.astype(np.uint8))
    y_img = y_img.resize((int(img_w), int(img_h)), Image.Resampling.NEAREST)
    return np.array(y_img, dtype=np.uint8)


def build_rep_cache(
    samples: list[Sample],
    rep: str,
    window_us: int,
    img_h: int,
    img_w: int,
    num_bins: int,
    cache_root: Path,
    split_name: str,
    dtype_name: str = "float16",
) -> Path:
    cache_root = Path(cache_root)
    dtype = cache_type(dtype_name)
    n = len(samples)
    channels = get_num_channels(rep, num_bins=num_bins)
    sig = samples_hash(samples)
    window_ms = int(round(window_us / 1000.0))
    shape = [n, channels, int(img_h), int(img_w)]
    cache_dir = cache_root / "representations" / f"h{img_h}_w{img_w}_bins{num_bins}_{dtype.name}"
    stem = f"{split_name}_{sig[:12]}_{rep}_{window_ms}ms"
    cache_path = cache_dir / f"{stem}.npy"
    meta_path = cache_dir / f"{stem}.json"
    expected_meta = {
        "kind": "representation",
        "sample_signature": sig,
        "sample_count": n,
        "shape": shape,
        "dtype": dtype.name,
        "representation": rep,
        "window_us": int(window_us),
        "height": int(img_h),
        "width": int(img_w),
        "num_bins": int(num_bins),
        "label_spec": None,
    }
    if cache_path.exists() and cache_ok(meta_path, expected_meta):
        return cache_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_dir / f"{stem}.tmp.npy"
    if tmp_path.exists():
        tmp_path.unlink()
    arr = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=dtype, shape=tuple(shape))
    sequences: dict[str, EventSeq] = {}
    try:
        for idx, sample in enumerate(samples):
            if sample.seq not in sequences:
                sequences[sample.seq] = EventSeq(SEQUENCE_PATHS[sample.seq])
            seq = sequences[sample.seq]
            window_events = seq.get_window(t_end_abs=sample.timestamp, delta_t=window_us)
            rep = make_representation(
                window_events,
                rep=rep,
                t_end=sample.timestamp,
                delta_t=window_us,
                height=LABEL_HEIGHT,
                width=WIDTH,
                num_bins=num_bins,
                rectify_map=seq.rectify_map,
            )
            rep = resize_rep(rep, img_h=img_h, img_w=img_w)
            arr[idx] = pack_rep(rep, dtype)
        arr.flush()
    finally:
        for seq in sequences.values():
            seq.close()
        del arr

    if cache_path.exists():
        cache_path.unlink()
    tmp_path.replace(cache_path)
    expected_meta["created_by"] = Path(__file__).name
    expected_meta["raw_label_height"] = LABEL_HEIGHT
    expected_meta["raw_width"] = WIDTH
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(expected_meta, f, indent=2)
    return cache_path


def build_label_cache(
    samples: list[Sample],
    labels: Labels,
    img_h: int,
    img_w: int,
    cache_root: Path,
    split_name: str,
) -> Path:
    cache_root = Path(cache_root)
    n = len(samples)
    sig = samples_hash(samples)
    shape = [n, int(img_h), int(img_w)]
    cache_dir = cache_root / "labels" / f"h{img_h}_w{img_w}"
    stem = f"{split_name}_{sig[:12]}_{clean_name(labels.name)}"
    cache_path = cache_dir / f"{stem}.npy"
    meta_path = cache_dir / f"{stem}.json"
    expected_meta = {
        "kind": "label",
        "sample_signature": sig,
        "sample_count": n,
        "shape": shape,
        "dtype": "uint8",
        "representation": None,
        "window_us": None,
        "height": int(img_h),
        "width": int(img_w),
        "num_bins": None,
        "label_spec": labels.name,
    }
    if cache_path.exists() and cache_ok(meta_path, expected_meta):
        return cache_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_dir / f"{stem}.tmp.npy"
    if tmp_path.exists():
        tmp_path.unlink()
    arr = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.uint8, shape=tuple(shape))
    for idx, sample in enumerate(samples):
        mask19 = load_mask(sample.label_path)
        mask = labels.remap_fn(mask19)
        arr[idx] = resize_label(mask, img_h=img_h, img_w=img_w)
    arr.flush()
    del arr
    if cache_path.exists():
        cache_path.unlink()
    tmp_path.replace(cache_path)
    expected_meta["created_by"] = Path(__file__).name
    expected_meta["raw_label_height"] = LABEL_HEIGHT
    expected_meta["raw_width"] = WIDTH
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(expected_meta, f, indent=2)
    return cache_path

# AUGMENTATION AND DATASET

def _augment_pair(x: torch.Tensor, y: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> tuple[torch.Tensor, torch.Tensor]:
    y_u8 = y.to(torch.uint8).unsqueeze(0)

    # Random crop + resize approximates the crop augmentation used in EV-SegNet while preserving fixed output size.
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
        angle = random.uniform(-15.0, 15.0)
        x = TF.rotate(x, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        y_u8 = TF.rotate(y_u8, angle, interpolation=TF.InterpolationMode.NEAREST, fill=ignore_index)

    if random.random() < 0.5:
        H, W = x.shape[-2:]
        dx = int(random.uniform(-0.25, 0.25) * W)
        dy = int(random.uniform(-0.25, 0.25) * H)
        x = TF.affine(x, angle=0, translate=[dx, dy], scale=1.0, shear=0,
                      interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        y_u8 = TF.affine(y_u8, angle=0, translate=[dx, dy], scale=1.0, shear=0,
                         interpolation=TF.InterpolationMode.NEAREST, fill=ignore_index)

    return x, y_u8.squeeze(0).long()


class DsecDataset(Dataset):

    def __init__(
        self,
        samples: list[Sample],
        rep: str,
        labels: Labels,
        train: bool,
        img_h: int,
        img_w: int,
        window_us: int,
        num_bins: int,
        cache_root: Path,
        split_name: str,
        cache_dtype: str = "float16",
    ):
        self.samples = samples
        self.rep = rep
        self.labels = labels
        self.train = train
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.window_us = int(window_us)
        self.num_bins = int(num_bins)
        self.cache_root = Path(cache_root)
        self.split_name = split_name
        self.cache_dtype = cache_dtype
        self.rep_cache_path = build_rep_cache(
            samples=samples,
            rep=rep,
            window_us=window_us,
            img_h=img_h,
            img_w=img_w,
            num_bins=num_bins,
            cache_root=self.cache_root,
            split_name=split_name,
            dtype_name=cache_dtype,
        )
        self.label_cache_path = build_label_cache(
            samples=samples,
            labels=labels,
            img_h=img_h,
            img_w=img_w,
            cache_root=self.cache_root,
            split_name=split_name,
        )
        self._rep_cache = None
        self._label_cache = None

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_rep_cache"] = None
        state["_label_cache"] = None
        return state

    def _open_caches(self) -> None:
        if self._rep_cache is None:
            self._rep_cache = np.load(self.rep_cache_path, mmap_mode="r")
        if self._label_cache is None:
            self._label_cache = np.load(self.label_cache_path, mmap_mode="r")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._open_caches()
        x_np = unpack_rep(np.array(self._rep_cache[index], copy=True))
        y_np = np.array(self._label_cache[index], dtype=np.int64, copy=True)
        x = torch.from_numpy(x_np)
        y = torch.from_numpy(y_np).long()
        if self.train:
            x, y = _augment_pair(x, y)
        return x, y


# LOSS, METRICS, SAMPLING

class CEDiceLoss(nn.Module):
    def __init__(self, weights: Optional[torch.Tensor], ignore_index: int, dice_class_ids: tuple[int, ...],
                 dice_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)
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


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int,
                           ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]
    if target.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.float64)
    indices = target * num_classes + pred
    return torch.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes).double()


def calc_iou(cm: torch.Tensor) -> tuple[np.ndarray, float, float]:
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


def main_metric(iou: np.ndarray, miou: float, labels: Labels) -> float:
    # Metric used for model selection: mIoU for multiclass, human IoU for binary human.
    class_id = labels.primary_metric_class_id
    if class_id is None:
        return float(miou)
    if 0 <= int(class_id) < len(iou):
        return float(iou[int(class_id)])
    return float(miou)


def human_iou(iou: np.ndarray, labels: Labels) -> Optional[float]:
    for class_id, name in enumerate(labels.class_names):
        if name.strip().lower() == "human" and class_id < len(iou):
            return float(iou[class_id])
    return None


def clean_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")


def pixel_counts(samples: Iterable[Sample], labels: Labels) -> np.ndarray:
    counts = np.zeros(labels.num_classes, dtype=np.int64)
    for sample in samples:
        mask19 = load_mask(sample.label_path)
        mask = labels.remap_fn(mask19)
        valid = mask != IGNORE_INDEX
        values, c = np.unique(mask[valid], return_counts=True)
        for value, count in zip(values, c):
            if 0 <= value < labels.num_classes:
                counts[int(value)] += int(count)
    return counts


def get_weights(counts: np.ndarray, max_weight: float = 20.0) -> torch.Tensor:
    counts = np.maximum(counts.astype(np.float64), 1.0)
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / weights.mean()
    weights = np.clip(weights, 0.1, max_weight)
    return torch.tensor(weights, dtype=torch.float32)

def get_loss(labels: Labels, weights: torch.Tensor, task: Experiment) -> nn.Module:
    if labels.use_dice_loss:
        return CEDiceLoss(weights, IGNORE_INDEX, labels.dice_class_ids, dice_weight=task.dice_weight)
    return nn.CrossEntropyLoss(weight=weights, ignore_index=IGNORE_INDEX)


# TRAINING AND EVALUATION

def poly_lr(current_step: int, total_steps: int, power: float = POLY_POWER) -> float:
    if total_steps <= 0:
        return 1.0
    return max(0.0, (1.0 - float(current_step) / float(total_steps)) ** power)


@torch.no_grad()
def check_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
             num_classes: int, use_amp: bool) -> tuple[float, np.ndarray, float, float]:
    model.eval()
    total_loss = 0.0
    n = 0
    cm_total = torch.zeros((num_classes, num_classes), dtype=torch.float64)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        if device.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(x)
            logits = outputs["out"]
            loss = criterion(logits, y)
        pred = torch.argmax(logits, dim=1)
        cm_total += confusion_matrix(pred.cpu(), y.cpu(), num_classes=num_classes)
        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)
    iou, miou, acc = calc_iou(cm_total)
    return total_loss / max(n, 1), iou, miou, acc


def save_csv(path: Path, rows: list[dict], fieldnames: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
        for row in rows[1:]:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_md_table(path: Path, rows: list[dict], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No results yet.\n", encoding="utf-8")
        return
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(f"{row.get(col, ''):.4f}" if isinstance(row.get(col, ''), float) else str(row.get(col, '')) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mean_over_seeds(rows: list[dict]) -> list[dict]:
    if not rows:
        return []

    base_metrics = [
        "best_epoch", "best_optimizer_step",
        "best_val_primary_metric", "best_val_miou", "best_val_human_iou",
        "test_primary_metric_10ms", "test_primary_metric_50ms", "test_primary_metric_250ms",
        "test_human_iou_10ms", "test_human_iou_50ms", "test_human_iou_250ms",
        "test_miou_10ms", "test_miou_50ms", "test_miou_250ms",
        "test_acc_10ms", "test_acc_50ms", "test_acc_250ms",
    ]
    dynamic_metrics = sorted({
        key for row in rows for key in row
        if key.startswith("test_iou_") and key.endswith("ms")
    })
    metrics = list(dict.fromkeys(base_metrics + dynamic_metrics))

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row["experiment"]), str(row["representation"]))
        grouped.setdefault(key, []).append(row)

    aggregated: list[dict] = []
    for (experiment, rep), group in sorted(grouped.items()):
        group = sorted(group, key=lambda r: int(r.get("seed", 0)))
        out: dict[str, object] = {
            "experiment": experiment,
            "representation": rep,
            "n_seeds": len(group),
            "seeds": ",".join(str(r.get("seed", "")) for r in group),
            "target_epochs": group[0].get("target_epochs", ""),
            "actual_target_epochs": group[0].get("actual_target_epochs", ""),
            "in_channels": group[0].get("in_channels", ""),
            "trainable_parameters": group[0].get("trainable_parameters", ""),
            "primary_metric_label": group[0].get("primary_metric_label", ""),
        }
        for metric in metrics:
            values = [float(r[metric]) for r in group if metric in r and r[metric] not in ("", None)]
            if not values:
                continue
            arr = np.asarray(values, dtype=np.float64)
            out[f"{metric}_mean"] = float(arr.mean())
            out[f"{metric}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        aggregated.append(out)
    return aggregated


def mean_std(row: dict, metric: str) -> str:
    mean = row.get(f"{metric}_mean", "")
    std = row.get(f"{metric}_std", "")
    if mean == "" or std == "":
        return ""
    return f"{float(mean):.4f} ± {float(std):.4f}"


def save_avg_md(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No aggregate results yet.\n", encoding="utf-8")
        return
    columns = [
        "experiment", "representation", "n", "seeds", "params",
        "mIoU 50ms", "human IoU 50ms", "accuracy 50ms",
        "mIoU 10ms", "mIoU 250ms", "human IoU 10ms", "human IoU 250ms",
    ]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [
            str(row.get("experiment", "")),
            str(row.get("representation", "")),
            str(row.get("n_seeds", "")),
            str(row.get("seeds", "")),
            str(row.get("trainable_parameters", "")),
            mean_std(row, "test_miou_50ms"),
            mean_std(row, "test_human_iou_50ms"),
            mean_std(row, "test_acc_50ms"),
            mean_std(row, "test_miou_10ms"),
            mean_std(row, "test_miou_250ms"),
            mean_std(row, "test_human_iou_10ms"),
            mean_std(row, "test_human_iou_250ms"),
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_summaries(run_dir: Path, rows: list[dict]) -> None:
    per_seed_columns = [
        "experiment", "representation", "seed", "seed_index", "target_epochs", "best_epoch", "best_optimizer_step",
        "in_channels", "trainable_parameters",
        "test_miou_50ms", "test_human_iou_50ms", "test_acc_50ms",
        "test_miou_10ms", "test_miou_250ms", "test_human_iou_10ms", "test_human_iou_250ms",
    ]
    save_csv(run_dir / "summary_results_by_seed.csv", rows)
    save_md_table(run_dir / "summary_results_by_seed.md", rows, per_seed_columns)

    aggregated = mean_over_seeds(rows)
    save_csv(run_dir / "summary_results_mean_std.csv", aggregated)
    save_avg_md(run_dir / "summary_results_mean_std.md", aggregated)


def get_scaler(device: torch.device, use_amp: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=use_amp)
        except TypeError:
            return torch.amp.GradScaler(enabled=use_amp)
    return torch.cuda.amp.GradScaler(enabled=use_amp)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    device: torch.device,
    labels: Labels,
    train_steps: int,
    accum_steps: int,
    eval_steps: int,
    out_dir: Path,
    metadata: dict,
    use_amp: bool,
    patience: int,
    min_delta: float,
) -> dict:
    scaler = get_scaler(device, use_amp)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows: list[dict] = []
    best_val_primary_metric = -1.0
    best_record: dict = {}
    no_improve_evals = 0

    train_iter = iter(train_loader)
    recent_losses: list[float] = []

    for opt_step in range(1, train_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(x)
                loss = segmentation_loss(outputs, y, criterion, aux_weight=0.4)
                scaled_loss = loss / accum_steps
            scaler.scale(scaled_loss).backward()
            accum_loss += float(loss.item())

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        recent_losses.append(accum_loss / accum_steps)
        if len(recent_losses) > eval_steps:
            recent_losses = recent_losses[-eval_steps:]

        should_eval = (opt_step % eval_steps == 0) or (opt_step == train_steps)
        if should_eval:
            val_loss, val_iou, val_miou, val_acc = check_model(
                model, val_loader, criterion, device, labels.num_classes, use_amp=use_amp
            )
            val_primary_metric = main_metric(val_iou, val_miou, labels)
            val_human_iou = human_iou(val_iou, labels)
            row = {
                "optimizer_step": opt_step,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss_recent": float(np.mean(recent_losses)) if recent_losses else 0.0,
                "val_loss": val_loss,
                "val_miou": val_miou,
                "val_primary_metric": val_primary_metric,
                "val_human_iou": val_human_iou if val_human_iou is not None else "",
                "val_acc": val_acc,
            }
            metrics_rows.append(row)
            save_csv(out_dir / "validation_metrics.csv", metrics_rows)

            if val_primary_metric > best_val_primary_metric + min_delta:
                best_val_primary_metric = val_primary_metric
                no_improve_evals = 0
                best_record = {
                    **metadata,
                    "best_optimizer_step": int(opt_step),
                    "best_val_primary_metric": float(val_primary_metric),
                    "best_val_human_iou": float(val_human_iou) if val_human_iou is not None else "",
                    "best_val_miou": float(val_miou),
                    "best_val_acc": float(val_acc),
                    "best_val_iou": val_iou.tolist(),
                    "class_names": labels.class_names,
                    "primary_metric_class_id": labels.primary_metric_class_id,
                    "primary_metric_label": labels.primary_metric_label,
                    "early_stopping_patience_evals": int(patience),
                    "early_stopping_min_delta": float(min_delta),
                }
                torch.save({"model": model.state_dict(), **best_record}, out_dir / "best.pt")
            else:
                no_improve_evals += 1
                if patience > 0 and no_improve_evals >= patience:
                    break

    if not best_record:
        raise RuntimeError("No validation record was produced; reduce eval_every_steps or check the validation loader.")
    return best_record


# EXPERIMENT

def make_loaders(task: Experiment, rep: str, train_data: list[Sample], val_data: list[Sample],
                  args: Params, generator: Optional[torch.Generator], cache_root: Path) -> tuple[DataLoader, DataLoader]:
    train_dataset = DsecDataset(
        samples=train_data, rep=rep, labels=task.labels, train=True,
        img_h=args.img_h, img_w=args.img_w, window_us=TRAIN_WINDOW_US, num_bins=NUM_BINS,
        cache_root=cache_root, split_name="train", cache_dtype=args.cache_dtype,
    )
    val_dataset = DsecDataset(
        samples=val_data, rep=rep, labels=task.labels, train=False,
        img_h=args.img_h, img_w=args.img_w, window_us=TRAIN_WINDOW_US, num_bins=NUM_BINS,
        cache_root=cache_root, split_name="val", cache_dtype=args.cache_dtype,
    )
    loader_kwargs = loader_opts(args, generator)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, sampler=None,
        drop_last=False, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        drop_last=False, **loader_kwargs,
    )
    return train_loader, val_loader


def test_all_windows(model: nn.Module, task: Experiment, rep: str, test_data: list[Sample],
                          criterion: nn.Module, args: Params, device: torch.device, use_amp: bool,
                          cache_root: Path) -> list[dict]:
    rows: list[dict] = []
    for window_us in TEST_WINDOWS_US:
        dataset = DsecDataset(
            samples=test_data, rep=rep, labels=task.labels, train=False,
            img_h=args.img_h, img_w=args.img_w, window_us=window_us, num_bins=NUM_BINS,
            cache_root=cache_root, split_name="test", cache_dtype=args.cache_dtype,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, **loader_opts(args, None))
        test_loss, test_iou, test_miou, test_acc = check_model(model, loader, criterion, device, task.labels.num_classes, use_amp=use_amp)
        human_value = human_iou(test_iou, task.labels)
        row = {
            "window_us": int(window_us),
            "window_ms": float(window_us / 1000.0),
            "test_loss": test_loss,
            "test_miou": test_miou,
            "test_human_iou": human_value if human_value is not None else "",
            "test_primary_metric": main_metric(test_iou, test_miou, task.labels),
            "test_acc": test_acc,
        }
        for class_id, class_name in enumerate(task.labels.class_names):
            row[f"iou_{class_id}_{class_name}"] = float(test_iou[class_id])
        rows.append(row)
    return rows


def run_exp(task: Experiment, rep: str, train_data: list[Sample],
                                      val_data: list[Sample], test_data: list[Sample],
                                      args: Params, device: torch.device, run_dir: Path,
                                      seed: int, seed_index: int, cache_root: Path,
                                      split_class_counts: dict[str, np.ndarray]) -> dict:
    seed = int(seed)
    set_seed(seed)
    generator = make_generator(seed)

    labels = task.labels
    out_dir = run_dir / task.name / rep / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_summary_path = out_dir / "result_summary.json"

    train_counts = split_class_counts["train"]
    val_counts = split_class_counts["val"]
    test_counts = split_class_counts["test"]

    distributions = []
    for split, counts in [("train", train_counts), ("val", val_counts), ("test", test_counts)]:
        total = int(counts.sum())
        for class_id, class_name in enumerate(labels.class_names):
            distributions.append({
                "split": split,
                "class_id": class_id,
                "class_name": class_name,
                "pixels": int(counts[class_id]),
                "ratio": float(counts[class_id] / max(total, 1)),
            })
    save_csv(out_dir / "pixel_distribution.csv", distributions)

    in_channels = get_num_channels(rep, num_bins=NUM_BINS)
    model = SegceptionLite(in_channels=in_channels, num_classes=labels.num_classes, base=args.model_base, aux=True).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    param_count = num_params(model)
    weights = get_weights(train_counts, max_weight=task.max_class_weight).to(device)
    criterion = get_loss(labels, weights, task)
    requested_epochs = int(args.epochs) if args.epochs is not None else int(task.default_epochs)
    computed_target_steps, steps_epoch = calc_steps(
        num_samples=len(train_data),
        batch_size=args.batch_size,
        accum_steps=args.accum_steps,
        epochs=requested_epochs,
    )
    train_steps = computed_target_steps
    eval_steps = calc_eval_steps(steps_epoch, args.eval_every)
    actual_target_epochs = float(train_steps / max(steps_epoch, 1))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: poly_lr(step, train_steps, POLY_POWER)
    )
    train_loader, val_loader = make_loaders(task, rep, train_data, val_data, args, generator=generator, cache_root=cache_root)

    metadata = {
        "experiment": task.name,
        "representation": rep,
        "seed": seed,
        "seed_index": seed_index,
        "in_channels": in_channels,
        "trainable_parameters": param_count,
        "train_sequences": TRAIN_SEQUENCE_IDS,
        "val_sequences": VAL_SEQUENCE_IDS,
        "test_sequences": TEST_SEQUENCE_IDS,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "test_samples": len(test_data),
        "train_window_us": TRAIN_WINDOW_US,
        "test_windows_us": TEST_WINDOWS_US,
        "train_height": args.img_h,
        "train_width": args.img_w,
        "micro_batch_size": args.batch_size,
        "grad_accum_steps": args.accum_steps,
        "effective_batch_size": args.batch_size * args.accum_steps,
        "target_epochs": requested_epochs,
        "actual_target_epochs": actual_target_epochs,
        "steps_per_epoch": steps_epoch,
        "target_optimizer_steps": train_steps,
        "eval_every_steps": eval_steps,
        "eval_every_epochs": args.eval_every,
        "early_stopping_patience_evals": args.patience,
        "early_stopping_min_delta": args.min_delta,
        "learning_rate": args.lr,
        "poly_lr_power": POLY_POWER,
        "model_base": args.model_base,
        "class_weights": weights.detach().cpu().tolist(),
        "frame_filtering": "none",
        "primary_metric_label": labels.primary_metric_label,
        "primary_metric_class_id": labels.primary_metric_class_id,
    }
    with (out_dir / "run_parameters.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    use_amp = bool(args.amp and device.type == "cuda")
    best_record = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        labels=labels,
        train_steps=train_steps,
        accum_steps=args.accum_steps,
        eval_steps=eval_steps,
        out_dir=out_dir,
        metadata=metadata,
        use_amp=use_amp,
        patience=args.patience,
        min_delta=args.min_delta,
    )

    checkpoint = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])

    test_rows = test_all_windows(model, task, rep, test_data, criterion, args, device, use_amp=use_amp, cache_root=cache_root)
    save_csv(out_dir / "test_metrics_by_window.csv", test_rows)

    row_50 = next(row for row in test_rows if row["window_us"] == TRAIN_WINDOW_US)
    per_class_rows = []
    for class_id, class_name in enumerate(labels.class_names):
        per_class_rows.append({"class_id": class_id, "class_name": class_name, "test_iou_50ms": row_50[f"iou_{class_id}_{class_name}"]})
    save_csv(out_dir / "test_per_class_iou_50ms.csv", per_class_rows)

    result = {
        "experiment": task.name,
        "representation": rep,
        "seed": seed,
        "seed_index": seed_index,
        "target_epochs": requested_epochs,
        "actual_target_epochs": actual_target_epochs,
        "steps_per_epoch": steps_epoch,
        "in_channels": in_channels,
        "trainable_parameters": param_count,
        "best_optimizer_step": int(best_record["best_optimizer_step"]),
        "best_epoch": float(best_record["best_optimizer_step"] / max(steps_epoch, 1)),
        "best_val_primary_metric": float(best_record["best_val_primary_metric"]),
        "best_val_human_iou": best_record.get("best_val_human_iou", ""),
        "best_val_miou": float(best_record["best_val_miou"]),
        "test_primary_metric_50ms": float(row_50["test_primary_metric"]),
        "test_human_iou_50ms": row_50.get("test_human_iou", ""),
        "test_miou_50ms": float(row_50["test_miou"]),
        "test_acc_50ms": float(row_50["test_acc"]),
        "checkpoint_path": str(out_dir / "best.pt"),
        "primary_metric_label": labels.primary_metric_label,
    }
    for row in test_rows:
        ms = int(row["window_ms"])
        result[f"test_miou_{ms}ms"] = float(row["test_miou"])
        result[f"test_human_iou_{ms}ms"] = row.get("test_human_iou", "")
        result[f"test_primary_metric_{ms}ms"] = float(row["test_primary_metric"])
        result[f"test_acc_{ms}ms"] = float(row["test_acc"])
        for class_id, class_name in enumerate(labels.class_names):
            safe_name = clean_name(class_name)
            result[f"test_iou_{safe_name}_{ms}ms"] = float(row[f"iou_{class_id}_{class_name}"])

    with result_summary_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


# MAIN

def main() -> None:
    args = Params()
    set_seed(RANDOM_SEED)
    set_repeatable()

    reps = list(args.reps)
    seeds = list(args.seeds)
    set_root(args.data_root)

    output_root = args.output_root.expanduser().resolve()
    cache_root = args.cache_root.expanduser().resolve() if args.cache_root else (output_root / "representation_cache")
    cache_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")
    run_dir = new_run_dir(output_root / "runs" / args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "device": str(device),
        "use_amp": use_amp,
        "data_root": str(DATA_ROOT),
        "output_root": str(output_root),
        "cache_root": str(cache_root),
        "cache_dtype": args.cache_dtype,
        "representations": reps,
        "seeds": seeds,
        "experiments": [cfg.name for cfg in EXPERIMENTS],
        "train_sequences": TRAIN_SEQUENCE_IDS,
        "val_sequences": VAL_SEQUENCE_IDS,
        "test_sequences": TEST_SEQUENCE_IDS,
        "train_window_us": TRAIN_WINDOW_US,
        "test_windows_us": TEST_WINDOWS_US,
        "epochs": args.epochs,
        "default_multiclass_epochs": DEFAULT_MULTICLASS_EPOCHS,
        "default_human_epochs": DEFAULT_HUMAN_EPOCHS,
        "early_stopping_patience_evals": args.patience,
        "early_stopping_min_delta": args.min_delta,
        "micro_batch_size": args.batch_size,
        "grad_accum_steps": args.accum_steps,
        "effective_batch_size": args.batch_size * args.accum_steps,
        "learning_rate": args.lr,
        "poly_lr_power": POLY_POWER,
        "train_height": args.img_h,
        "train_width": args.img_w,
        "model_base": args.model_base,
        "num_workers": args.workers,
        "prefetch_factor": args.prefetch,
        "eval_every_epochs": args.eval_every,
        "default_random_seed": RANDOM_SEED,
        **system_info(),
    }
    with (run_dir / "run_parameters.json").open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    check_files(SEQUENCE_IDS)
    base_train_samples = build_samples(TRAIN_SEQUENCE_IDS)
    base_val_samples = build_samples(VAL_SEQUENCE_IDS)
    base_test_samples = build_samples(TEST_SEQUENCE_IDS)

    all_results: list[dict] = []
    for task in EXPERIMENTS:
        train_data, val_data, test_data = base_train_samples, base_val_samples, base_test_samples

        split_class_counts = {
            "train": pixel_counts(train_data, task.labels),
            "val": pixel_counts(val_data, task.labels),
            "test": pixel_counts(test_data, task.labels),
        }

        for rep in reps:
            for seed_index, seed in enumerate(seeds):
                result = run_exp(
                    task=task,
                    rep=rep,
                    train_data=train_data,
                    val_data=val_data,
                    test_data=test_data,
                    args=args,
                    device=device,
                    run_dir=run_dir,
                    seed=seed,
                    seed_index=seed_index,
                    cache_root=cache_root,
                    split_class_counts=split_class_counts,
                )
                all_results.append(result)
                save_summaries(run_dir, all_results)


    save_summaries(run_dir, all_results)


if __name__ == "__main__":
    main()
