from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_HEIGHT = 480
DEFAULT_WIDTH = 640
SUPPORTED_REPRESENTATIONS = ("recent", "voxel", "evsegnet")


@dataclass
class Events:
    x: np.ndarray
    y: np.ndarray
    t: np.ndarray
    p: np.ndarray

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x)
        self.y = np.asarray(self.y)
        self.t = np.asarray(self.t)
        self.p = np.asarray(self.p)

    def __len__(self) -> int:
        return len(self.x)


def time_window(events: Events, t_end: int | float, delta_t: int | float) -> Events:
    t_start = t_end - delta_t
    i0 = np.searchsorted(events.t, t_start, side="left")
    i1 = np.searchsorted(events.t, t_end, side="right")
    return Events(
        x=events.x[i0:i1],
        y=events.y[i0:i1],
        t=events.t[i0:i1],
        p=events.p[i0:i1],
    )


def norm_p(p: np.ndarray) -> np.ndarray:
    return np.where(p > 0, 1, -1).astype(np.int8)


def valid_events(events: Events, height: int, width: int) -> np.ndarray:
    return (events.x >= 0) & (events.x < width) & (events.y >= 0) & (events.y < height)


def max_norm(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    img = img.astype(np.float32)
    m = float(img.max())
    if m < eps:
        return np.zeros_like(img, dtype=np.float32)
    return img / m


def percentile_norm(img: np.ndarray, percentile: float = 99.0, eps: float = 1e-6) -> np.ndarray:
    img = img.astype(np.float32)
    scale = float(np.percentile(img, percentile))
    if scale < eps:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip(img / scale, 0.0, 1.0).astype(np.float32)


def norm_channels(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32).copy()
    for c in range(x.shape[0]):
        x[c] = percentile_norm(x[c])
    return x


def flat_index(y: np.ndarray, x: np.ndarray, width: int) -> np.ndarray:
    return y.astype(np.int64, copy=False) * int(width) + x.astype(np.int64, copy=False)


def count_image(index: np.ndarray, size: int, weights: Optional[np.ndarray] = None) -> np.ndarray:
    if index.size == 0:
        return np.zeros(size, dtype=np.float32)
    values = np.bincount(index, weights=weights, minlength=size)
    if values.shape[0] > size:
        values = values[:size]
    return values.astype(np.float32, copy=False)


def rectify_events(events: Events, rectify_map: np.ndarray, height: int, width: int) -> Events:
    if len(events) == 0:
        return events

    map_h, map_w = rectify_map.shape[:2]
    ok = (events.x >= 0) & (events.x < map_w) & (events.y >= 0) & (events.y < map_h)
    if not np.any(ok):
        return Events(
            x=np.empty(0, dtype=np.int64),
            y=np.empty(0, dtype=np.int64),
            t=np.empty(0, dtype=events.t.dtype),
            p=np.empty(0, dtype=events.p.dtype),
        )

    x_raw = events.x[ok].astype(np.int64)
    y_raw = events.y[ok].astype(np.int64)
    t = events.t[ok]
    p = events.p[ok]

    rectified = rectify_map[y_raw, x_raw]
    x_new = np.rint(rectified[:, 0]).astype(np.int64)
    y_new = np.rint(rectified[:, 1]).astype(np.int64)
    ok_new = (x_new >= 0) & (x_new < width) & (y_new >= 0) & (y_new < height)

    return Events(x=x_new[ok_new], y=y_new[ok_new], t=t[ok_new], p=p[ok_new])


def recent_rep(events: Events, t_start: int | float, t_end: int | float,
               height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH) -> np.ndarray:
    out = np.zeros((4, height, width), dtype=np.float32)
    if len(events) == 0:
        return out

    ok = valid_events(events, height, width)
    if not np.any(ok):
        return out

    size = int(height) * int(width)
    x = events.x[ok].astype(np.int64, copy=False)
    y = events.y[ok].astype(np.int64, copy=False)
    t = events.t[ok].astype(np.float32, copy=False)
    p = norm_p(events.p[ok])

    duration = max(float(t_end) - float(t_start), 1.0)
    t_norm = np.clip((t - float(t_start)) / duration, 0.0, 1.0).astype(np.float32)
    index = flat_index(y, x, width)

    for polarity, count_ch, time_ch in [(1, 0, 2), (-1, 1, 3)]:
        chosen = p == polarity
        if not np.any(chosen):
            continue
        out[count_ch] = count_image(index[chosen], size).reshape(height, width)
        np.maximum.at(out[time_ch].reshape(-1), index[chosen], t_norm[chosen])

    out[0] = max_norm(out[0])
    out[1] = max_norm(out[1])
    return out.astype(np.float32, copy=False)


def voxel_rep(events: Events, t_start: int | float, t_end: int | float,
              num_bins: int = 5, height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH) -> np.ndarray:
    channels = 2 * int(num_bins)
    size = int(height) * int(width)
    out = np.zeros((channels, height, width), dtype=np.float32)

    ok = valid_events(events, height, width)
    if not np.any(ok):
        return out

    x = events.x[ok].astype(np.int64, copy=False)
    y = events.y[ok].astype(np.int64, copy=False)
    t = events.t[ok].astype(np.float64, copy=False)
    p = norm_p(events.p[ok])

    duration = max(float(t_end) - float(t_start), 1.0)
    bins = ((t - float(t_start)) / duration * int(num_bins)).astype(np.int64)
    bins = np.clip(bins, 0, int(num_bins) - 1)

    ch = np.where(p > 0, bins, int(num_bins) + bins)
    pix = flat_index(y, x, width)
    index = ch.astype(np.int64, copy=False) * size + pix
    out = count_image(index, channels * size).reshape(channels, height, width)
    return norm_channels(out)


def evsegnet_rep(events: Events, t_start: int | float, t_end: int | float,
                 height: int = DEFAULT_HEIGHT, width: int = DEFAULT_WIDTH) -> np.ndarray:
    out = np.zeros((6, height, width), dtype=np.float32)
    if len(events) == 0:
        return out

    ok = valid_events(events, height, width)
    if not np.any(ok):
        return out

    size = int(height) * int(width)
    x_all = events.x[ok].astype(np.int64, copy=False)
    y_all = events.y[ok].astype(np.int64, copy=False)
    t_all = events.t[ok].astype(np.float32, copy=False)
    p_all = norm_p(events.p[ok])

    duration = max(float(t_end) - float(t_start), 1.0)
    t_all = np.clip((t_all - float(t_start)) / duration, 0.0, 1.0).astype(np.float32)
    index_all = flat_index(y_all, x_all, width)

    for polarity, count_ch, mean_ch, std_ch in [(-1, 0, 2, 4), (1, 1, 3, 5)]:
        chosen = p_all == polarity
        if not np.any(chosen):
            continue

        index = index_all[chosen]
        t = t_all[chosen]

        counts = count_image(index, size)
        sums = count_image(index, size, weights=t)
        sums2 = count_image(index, size, weights=t * t)

        mean = np.zeros(size, dtype=np.float32)
        std = np.zeros(size, dtype=np.float32)

        has_events = counts > 0
        mean[has_events] = sums[has_events] / counts[has_events]

        has_many = counts > 1
        var = np.zeros(size, dtype=np.float32)
        var[has_many] = (sums2[has_many] - (sums[has_many] ** 2) / counts[has_many]) / (counts[has_many] - 1.0)
        std[has_many] = np.sqrt(np.clip(var[has_many], 0.0, None))

        out[count_ch] = counts.reshape(height, width)
        out[mean_ch] = mean.reshape(height, width)
        out[std_ch] = std.reshape(height, width)

    out[0] = max_norm(out[0])
    out[1] = max_norm(out[1])
    out[2:] = np.clip(out[2:], 0.0, 1.0)
    return out.astype(np.float32, copy=False)


def get_num_channels(representation: str, num_bins: int = 5) -> int:
    representation = representation.strip().lower()
    if representation == "recent":
        return 4
    if representation == "voxel":
        return 2 * int(num_bins)
    if representation == "evsegnet":
        return 6
    raise ValueError(f"Unsupported representation: {representation}. Supported: {SUPPORTED_REPRESENTATIONS}")


def make_representation(
    events: Events,
    representation: str | None = None,
    *,
    rep: str | None = None,
    t_end: int | float,
    delta_t: int | float,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_bins: int = 5,
    rectify_map: Optional[np.ndarray] = None,
) -> np.ndarray:
    name = representation if representation is not None else rep
    if name is None:
        raise ValueError("Give a representation name.")

    name = name.strip().lower()
    t_start = t_end - delta_t
    window = time_window(events, t_end=t_end, delta_t=delta_t)

    if rectify_map is not None:
        window = rectify_events(window, rectify_map=rectify_map, height=height, width=width)

    if name == "recent":
        return recent_rep(window, t_start=t_start, t_end=t_end, height=height, width=width)

    if name == "voxel":
        return voxel_rep(window, t_start=t_start, t_end=t_end, num_bins=num_bins, height=height, width=width)

    if name == "evsegnet":
        return evsegnet_rep(window, t_start=t_start, t_end=t_end, height=height, width=width)

    raise ValueError(f"Unsupported representation: {name}. Supported: {SUPPORTED_REPRESENTATIONS}")
