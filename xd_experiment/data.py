"""XD official RGB feature I/O, verified frame GT, and original window semantics.

No filename labels enter pseudo-label targets, sampling, or training. Five genuine
spatial crops are averaged at the feature level, matching both branch loaders'
existing input semantics. This differs from XDVioDet's crop-probability mean.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ASSETS = Path(__file__).resolve().parent / "assets"
PATTERN = re.compile(r"^(.*)__([0-4])\.npy$")
ASSET_HASHES = {
    "official_test_ids.json": "2cfb5a6ad6ba4230f57feebe5f08bc59302a1b32b719c526c704344105040aea",
    "official_train_ids.json": "e98bc8c912d7e71ff9f5cdbf7d9ad36d5cac05c2eafefb5f5a103511cc969bc0",
    "official_annotations.txt": "27b583ba06fe2e096d7e3d281845365c3b24de7e30797ae5c1ec771667e04c2f",
}
OFFICIAL_GT_HASH = "55e3017b39c8429517d296f398f7ce34af1e77903fb2848f23b6c99ba1621b13"


def file_hash(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _protocol():
    for name, expected in ASSET_HASHES.items():
        if file_hash(ASSETS / name) != expected:
            raise ValueError(f"Official metadata asset changed: {name}")
    value = json.loads((ASSETS / "protocol.json").read_text("utf-8"))
    for key, default in {"dimension": 1024, "crops": 5, "frames_per_snippet": 16,
                         "train_videos": 3954, "test_videos": 800}.items():
        value.setdefault(key, default)
    if not value.get("expected_gt_uint8_sha256"):
        raise ValueError("Official GT numeric hash is required in assets/protocol.json")
    if (value["dimension"], value["crops"], value["frames_per_snippet"],
        value["train_videos"], value["test_videos"], value.get("expected_gt_frames")) != (1024, 5, 16, 3954, 800, 2330384):
        raise ValueError("Unexpected official XD feature protocol")
    if value["expected_gt_uint8_sha256"] != OFFICIAL_GT_HASH:
        raise ValueError("Unexpected official GT reference hash")
    return value


def discover(directory, dimension=1024):
    """Return video -> ordered five paths, checking every NPY header and size."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Missing feature directory: {directory}")
    groups = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".npy":
            continue
        match = PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unexpected feature filename: {path}")
        video, crop = match.group(1), int(match.group(2))
        groups.setdefault(video, {})[crop] = path
    if not groups:
        raise ValueError(f"No feature files: {directory}")
    result = {}
    for video, crops in groups.items():
        if set(crops) != set(range(5)):
            raise ValueError(f"Missing real crop files: {video}: {sorted(crops)}")
        paths = [crops[crop] for crop in range(5)]
        shapes = []
        for path in paths:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype != np.float32 or array.ndim != 2 or array.shape[0] < 1 or array.shape[1] != dimension:
                raise ValueError(f"Invalid feature header: {path}: {array.shape}/{array.dtype}")
            if path.stat().st_size != array.offset + array.nbytes:
                raise ValueError(f"Unexpected feature payload size: {path}")
            shapes.append(array.shape)
            del array
        if len(set(shapes)) != 1:
            raise ValueError(f"Crop temporal/dimension mismatch: {video}: {shapes}")
        result[video] = paths
    return result


def read_video_crops(files):
    """Read five actual crop files, require finite TD values, mean feature axis.

    The result retains each original temporal row and original feature dimension.
    Accumulation is float64 followed by float32 output to avoid order-dependent
    large rounding errors; no temporal interpolation, spatial duplication, or D
    padding occurs.
    """
    paths = [Path(path) for path in files]
    if len(paths) != 5:
        raise ValueError("Exactly five genuine crop files are required")
    matches = [PATTERN.fullmatch(path.name) for path in paths]
    if any(match is None for match in matches):
        raise ValueError("Invalid crop filename")
    if len({match.group(1) for match in matches}) != 1 or {int(match.group(2)) for match in matches} != set(range(5)):
        raise ValueError("Crop files must be one video, each real crop exactly once")
    total = None
    shape = None
    for path in paths:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.dtype != np.float32 or value.ndim != 2 or value.shape[0] < 1:
            raise ValueError(f"Invalid feature shape/dtype: {path}")
        if not np.isfinite(value).all():
            raise ValueError(f"Non-finite feature: {path}")
        if total is None:
            shape = value.shape
            total = np.zeros(shape, dtype=np.float64)
        elif value.shape != shape:
            raise ValueError(f"Mismatched crop shape: {path}")
        total += value
        del value
    result = (total / 5).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite averaged feature")
    return result


def _annotations(path):
    annotations = {}
    for number, line in enumerate(Path(path).read_text("utf-8-sig").splitlines(), 1):
        tokens = line.split()
        if not tokens:
            continue
        # The pinned official annotations have 14 IDs with an .mp4 suffix;
        # feature IDs omit it. Strip only this terminal container extension.
        if tokens[0].endswith(".mp4"):
            tokens[0] = tokens[0][:-4]
        if tokens[0] in annotations or (len(tokens) - 1) % 2:
            raise ValueError(f"Invalid annotation row {number}")
        values = list(map(int, tokens[1:]))
        bounds = list(zip(values[::2], values[1::2]))
        if any(start < 0 or end < start for start, end in bounds):
            raise ValueError(f"Invalid annotation interval at row {number}")
        annotations[tokens[0]] = bounds
    return annotations


def build_gt(ids, index, annotations_path, protocol):
    """Reproduce official 0-based [start:end) GT over the published T*16 horizon.

    Annotation ends beyond the published feature horizon clip naturally as in
    original NumPy slicing followed by official tail removal. Every abnormal ID
    must have an annotation row; every annotation row must map to an official ID.
    Full numeric equality to the independently published GT is mandatory.
    """
    annotations = _annotations(annotations_path)
    if set(annotations) - set(ids):
        raise ValueError("Annotation IDs missing from official test split")
    frames = int(index[-1, 1]) * 16
    gt = np.zeros(frames, dtype=np.uint8)
    for video, (a, b) in zip(ids, index):
        start_frame, end_frame = int(a) * 16, int(b) * 16
        if video in annotations:
            if "_label_A" in video:
                raise ValueError(f"Normal ID unexpectedly annotated: {video}")
            local = gt[start_frame:end_frame]
            for left, right in annotations[video]:
                local[left:right] = 1
        elif "_label_A" not in video:
            raise ValueError(f"Missing anomaly annotation: {video}")
    if frames != protocol.get("expected_gt_frames", 2330384):
        raise ValueError(f"Official GT frame count mismatch: {frames}")
    numeric_hash = hashlib.sha256(gt.tobytes()).hexdigest()
    if numeric_hash != protocol["expected_gt_uint8_sha256"]:
        raise ValueError("Reconstructed GT differs from independently published official GT")
    return gt


def _index(paths_by_id, ids):
    rows = []
    cursor = 0
    for video in ids:
        array = np.load(paths_by_id[video][0], mmap_mode="r", allow_pickle=False)
        length = len(array)
        del array
        rows.append((cursor, cursor + length))
        cursor += length
    return np.asarray(rows, dtype=np.int64)


def prepare(feature_root, cache_dir):
    """Build a new RGB cache; never overwrite existing data or partial results.

    This stage reads all source payloads and checks finiteness. It is intended
    for the school PC (about 4.8 GB output), not for local GPU-free smoke tests.
    """
    protocol = _protocol()
    supplied_root = Path(feature_root).expanduser().resolve()
    root = supplied_root
    if (root / "i3d-features").is_dir() and not (root / "RGB").is_dir():
        root = root / "i3d-features"
    destination = Path(cache_dir).expanduser().resolve()
    for source in {supplied_root, root}:
        if destination == source or source in destination.parents or destination in source.parents:
            raise ValueError("Cache directory must be separate from the original feature tree")
    train = discover(root / "RGB", protocol["dimension"])
    test = discover(root / "RGBTest", protocol["dimension"])
    test_ids = json.loads((ASSETS / "official_test_ids.json").read_text("utf-8"))
    if len(test_ids) != len(set(test_ids)) or set(test_ids) != set(test):
        raise ValueError("Local test membership differs from official test IDs")
    if len(train) != protocol["train_videos"] or len(test) != protocol["test_videos"]:
        raise ValueError(f"Unexpected official split sizes: train={len(train)}, test={len(test)}")
    if set(train) & set(test):
        raise ValueError("Train/test identity overlap")
    train_ids = sorted(train)
    expected_train = json.loads((ASSETS / "official_train_ids.json").read_text("utf-8"))
    if len(expected_train) != len(set(expected_train)) or set(expected_train) != set(train):
        raise ValueError("Local train membership differs from official IDs")
    train_ids = expected_train
    indices = {"train": _index(train, train_ids), "test": _index(test, test_ids)}
    # Verify annotation alignment before writing several GB of averaged data.
    gt = build_gt(test_ids, indices["test"], ASSETS / "official_annotations.txt", protocol)
    destination.mkdir(parents=True, exist_ok=False)
    incomplete = destination / "INCOMPLETE"
    incomplete.write_text("Preparation did not finish. Do not use this directory.\n", encoding="utf-8")
    ids = {"train": train_ids, "test": test_ids}
    inventory = []
    for split, groups in (("train", train), ("test", test)):
        out = np.lib.format.open_memmap(destination / f"{split}.npy", mode="w+", dtype=np.float32,
                                        shape=(int(indices[split][-1, 1]), protocol["dimension"]))
        for count, (video, (a, b)) in enumerate(zip(ids[split], indices[split]), 1):
            paths = groups[video]
            before = [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
            out[int(a):int(b)] = read_video_crops(paths)
            for path, prior in zip(paths, before):
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) != prior:
                    raise ValueError(f"Source feature changed during preparation: {path}")
                inventory.append({"split": split, "video": video, "file": str(path.relative_to(root)),
                                  "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
            if count % 100 == 0 or count == len(ids[split]):
                print(f"Prepared {split} {count}/{len(ids[split])}", flush=True)
        out.flush()
        del out
        np.save(destination / f"{split}_index.npy", indices[split], allow_pickle=False)
    np.save(destination / "gt.npy", gt, allow_pickle=False)
    write_json(destination / "ids.json", ids)
    write_json(destination / "source_inventory.json", inventory)
    files = ["train.npy", "test.npy", "train_index.npy", "test_index.npy", "gt.npy", "ids.json", "source_inventory.json"]
    manifest = {
        "schema_version": 1, "dataset": "XD-Violence", "modality": "RGB-only",
        "source_root": str(root), "dimension": protocol["dimension"], "real_crops": 5,
        "crop_policy": "arithmetic mean of five actual feature crops; float64 accumulation, float32 output",
        "temporal_policy": "natural published snippet order, each score repeated 16 times; official feature-aligned frame domain",
        "gt_indexing": "0-based end-exclusive annotation intervals; official numeric GT equality checked",
        "gt_uint8_sha256": hashlib.sha256(gt.tobytes()).hexdigest(),
        "train_videos": len(train_ids), "test_videos": len(test_ids), "test_frames": len(gt),
        "train_snippets": int(indices["train"][-1, 1]), "test_snippets": int(indices["test"][-1, 1]),
        "all_source_payloads_finite": True,
        "source_payload_hashes_recorded": False,
        "protocol": protocol,
        "asset_hashes": {name: file_hash(ASSETS / name) for name in
                         ("protocol.json", "official_train_ids.json", "official_test_ids.json", "official_annotations.txt")},
        "files": {name: {"bytes": (destination / name).stat().st_size,
                         "sha256": file_hash(destination / name)} for name in files},
    }
    write_json(destination / "manifest.json", manifest)
    incomplete.unlink()
    return manifest


def _validate_index(index, count, ids):
    if index.dtype.kind not in "iu" or index.shape != (len(ids), 2) or len(ids) != len(set(ids)):
        raise ValueError("Invalid video index/IDs")
    if not len(index) or index[0, 0] != 0 or index[-1, 1] != count:
        raise ValueError("Video index does not cover feature array")
    if np.any(index[:, 1] <= index[:, 0]) or np.any(index[1:, 0] != index[:-1, 1]):
        raise ValueError("Video index gap/overlap/empty video")


def load_cache(cache_dir, training=False, verify_hashes=True):
    """Load only the selected split. Training never reads GT or test arrays."""
    root = Path(cache_dir).expanduser().resolve()
    if (root / "INCOMPLETE").exists():
        raise ValueError("Cache preparation is incomplete")
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("dataset") != "XD-Violence":
        raise ValueError("Unexpected cache schema/dataset")
    protocol = _protocol()
    if manifest.get("protocol") != protocol:
        raise ValueError("Cache protocol differs from current pinned protocol")
    for name in ("protocol.json", "official_train_ids.json", "official_test_ids.json", "official_annotations.txt"):
        if file_hash(ASSETS / name) != manifest["asset_hashes"].get(name):
            raise ValueError(f"Cache metadata asset mismatch: {name}")
    if manifest.get("gt_uint8_sha256") != protocol["expected_gt_uint8_sha256"]:
        raise ValueError("Cache GT reference differs from pinned official GT")
    split = "train" if training else "test"
    files = [f"{split}.npy", f"{split}_index.npy", "ids.json"]
    if not training:
        files.append("gt.npy")
    for name in files:
        expected = manifest["files"][name]
        if (root / name).stat().st_size != expected["bytes"]:
            raise ValueError(f"Cache byte-size mismatch: {name}")
        if verify_hashes and file_hash(root / name) != expected["sha256"]:
            raise ValueError(f"Cache hash mismatch: {name}")
    features = np.load(root / f"{split}.npy", mmap_mode="r", allow_pickle=False)
    index = np.load(root / f"{split}_index.npy", allow_pickle=False)
    ids = json.loads((root / "ids.json").read_text("utf-8"))[split]
    expected_ids = json.loads((ASSETS / f"official_{split}_ids.json").read_text("utf-8"))
    if ids != expected_ids or len(ids) != protocol[f"{split}_videos"]:
        raise ValueError(f"Cache {split} IDs/order differ from official split")
    if features.dtype != np.float32 or features.ndim != 2 or features.shape[1] != manifest["dimension"]:
        raise ValueError("Invalid cached feature shape/dtype")
    _validate_index(index, len(features), ids)
    value = {"features": features, "index": index, "ids": ids, "manifest": manifest}
    if not training:
        gt = np.load(root / "gt.npy", mmap_mode="r", allow_pickle=False)
        if gt.dtype != np.uint8 or gt.ndim != 1 or len(gt) != len(features) * 16 or not np.isin(gt, [0, 1]).all():
            raise ValueError("Invalid cached frame GT")
        if len(gt) != protocol["expected_gt_frames"]:
            raise ValueError("Cached GT length differs from official frame horizon")
        if hashlib.sha256(gt.tobytes()).hexdigest() != protocol["expected_gt_uint8_sha256"]:
            raise ValueError("Cached numeric GT hash mismatch")
        value["gt"] = gt
    return value


class WindowDataset(Dataset):
    """Original branch window/stride/end-window semantics over crop-mean TD."""
    def __init__(self, features, index, pseudo, confidence, window, stride):
        if window < 1 or stride < 1 or stride > window:
            raise ValueError("Require 1 <= stride <= window")
        if features.ndim != 2 or features.dtype != np.float32:
            raise ValueError("Expected float32 TD feature cache")
        self._path = str(features.filename) if isinstance(features, np.memmap) else None
        self._features = features
        self._pid = os.getpid()
        self.pseudo = np.asarray(pseudo, dtype=np.float32)
        self.confidence = np.asarray(confidence, dtype=np.float32)
        for name, value in (("pseudo", self.pseudo), ("confidence", self.confidence)):
            if value.shape != (len(features),) or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
                raise ValueError(f"Invalid {name} targets")
        _validate_index(np.asarray(index), len(features), list(map(str, range(len(index)))))
        self.windows = []
        for video, (a, b) in enumerate(index):
            a, b = int(a), int(b)
            length = b - a
            if length <= window:
                self.windows.append((video, a, b))
            else:
                for local in range(0, length - window + 1, stride):
                    self.windows.append((video, a + local, a + local + window))
                if (length - window) % stride:
                    last = (video, b - window, b)
                    if last not in self.windows:
                        self.windows.append(last)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self._path:
            state["_features"] = None
        return state

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, item):
        if self._path and (self._features is None or self._pid != os.getpid()):
            self._features = np.load(self._path, mmap_mode="r", allow_pickle=False)
            self._pid = os.getpid()
        _, a, b = self.windows[item]
        return (torch.from_numpy(np.array(self._features[a:b], copy=True)),
                torch.from_numpy(self.pseudo[a:b].copy()),
                torch.from_numpy(self.confidence[a:b].copy()), b - a)


def collate(batch):
    """Original masked temporal batching; zero pad only within a batch."""
    features, labels, confidence, lengths = zip(*batch)
    maximum, dimension = max(lengths), features[0].shape[1]
    x = torch.zeros(len(batch), maximum, dimension, dtype=torch.float32)
    y = torch.zeros(len(batch), maximum, dtype=torch.float32)
    c = torch.zeros_like(y)
    mask = torch.zeros_like(y)
    for row, (feature, label, conf, length) in enumerate(batch):
        if feature.shape != (length, dimension):
            raise ValueError("Mismatched batch feature dimensions")
        x[row, :length], y[row, :length], c[row, :length] = feature, label, conf
        mask[row, :length] = 1
    return x, y, c, mask, torch.tensor(lengths, dtype=torch.long)
