"""XD input bridge to the unchanged model/UCFSHT implementations.

The two branches deliberately keep separate bridges and original model/loss/
pseudo/ITA functions. Only inputs, CLI isolation and output locations are new.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BRANCH = "model/UCFSHT"
COMMIT = "b68d216d4ee659afb7e087947f1b8e56fa86b330"
SOURCE_BLOBS = {
    "model.py": "83d7d26cf06f635a044ac0e337c7fc3f5126a24b",
    "train.py": "7a2eb74244a9a1ccc15c24455b1618d96ca034c8",
    "adapter.py": "6c582da70375f313e8c22d09433f45e67641f99e",
    "pseudo_oc.py": "fb004dd0eb97c463291e68bcfd1f469f61c7181d",
    "test.py": "4426d8245e5ebc3534968da88005d86b0f15241b",
}
PSEUDO_SETTINGS = {
    "feature_normalization": "standard", "threshold_method": "p_value",
    "score_normalization": "zscore", "prototype_method": "none",
    "use_attraction": True, "attraction_strength": 0.8,
    "attraction_iterations": 3, "repulsion_strength": 0.7,
    "remove_isolated_abn": True, "isolated_abn_min_length": 2,
    "fill_isolated_norm": True, "isolated_norm_max_gap": 2,
    "use_prototype_swap": False, "swap_threshold": 0.8,
}


def _blob_hash(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest()


def verify_source() -> dict:
    """Reject algorithm edits; tolerate Git's CRLF checkout conversion only."""
    files = {}
    for filename, expected in SOURCE_BLOBS.items():
        raw = (ROOT / filename).read_bytes().replace(b"\r\n", b"\n")
        actual = _blob_hash(raw)
        if actual != expected:
            raise RuntimeError(f"Preserved {BRANCH} source changed: {filename}")
        files[filename] = {"git_blob": expected}
    return {"branch": BRANCH, "commit": COMMIT, "files": files}


@lru_cache(maxsize=None)
def _module(filename: str) -> ModuleType:
    # Compile source directly rather than writing __pycache__ beside preserved files.
    verify_source()
    if filename == "pseudo_oc.py":
        import matplotlib
        matplotlib.use("Agg")
    path = ROOT / filename
    module = ModuleType("xd_ucfsht_original_" + path.stem)
    module.__file__ = str(path)
    exec(compile(path.read_text(encoding="utf-8-sig"), str(path), "exec"), module.__dict__)
    return module


def _dimension(dimension: int) -> int:
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("Feature dimension must be a positive integer")
    return dimension


def make_model(dimension: int) -> torch.nn.Module:
    verify_source()
    return _module("model.py").Model_V2_AllCNN(_dimension(dimension))


def make_adapter(dimension: int) -> torch.nn.Module:
    verify_source()
    # The original class accepts arbitrary d despite its historical class name.
    return _module("adapter.py").ResidualAdapter2048(d=_dimension(dimension), use_ln=True)


def train_epoch(loader, model, optimizer, epoch, device, accumulation_steps=1) -> float:
    verify_source()
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    return float(_module("train.py").concatenated_train_variable_length(
        loader, model, optimizer, epoch, device,
        accumulation_steps=accumulation_steps))


@contextmanager
def _output_cwd(directory: Path):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def generate_pseudo(featuresTD, indexNx2, outputdir) -> np.ndarray:
    """Run original OC main settings on already-averaged real crop features.

    features[:,None,:] is a zero-copy singleton-axis view, not a crop replica.
    The original generator's mean(axis=1) therefore preserves exactly (T,D).
    """
    provenance = verify_source()
    if featuresTD.ndim != 2 or featuresTD.dtype != np.dtype("float32"):
        raise ValueError("Expected float32 crop-mean features (total_T,D)")
    index = np.asarray(indexNx2)
    if index.ndim != 2 or index.shape[1] != 2 or not len(index) or not np.issubdtype(index.dtype, np.integer):
        raise ValueError("Expected integer (N,2) video intervals")
    if (index[0, 0] != 0 or index[-1, 1] != len(featuresTD)
            or np.any(index[:, 1] <= index[:, 0])
            or np.any(index[1:, 0] != index[:-1, 1])):
        raise ValueError("Video intervals must partition the feature timeline")
    if np.all(index[:, 1] - index[:, 0] < 8):
        raise ValueError("Original OC rank statistics require at least one video with >=8 snippets")
    output = Path(outputdir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _output_cwd(output):
        labels_list = _module("pseudo_oc.py").generate_improved_pseudo_labels(
            featuresTD[:, None, :], index, **PSEUDO_SETTINGS)
    if len(labels_list) != len(index):
        raise RuntimeError("Original pseudo generator changed video count")
    for labels, (start, end) in zip(labels_list, index):
        if len(labels) != int(end - start):
            raise RuntimeError("Original pseudo labels do not match video length")
    labels = np.concatenate(labels_list).astype(np.float32)
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise RuntimeError("Pseudo targets must be finite binary values")
    metadata = {"source": provenance, "settings": PSEUDO_SETTINGS,
                "p_value_inside_original_function": 0.1,
                "input": "mean of five real RGB crop features, (T,1024)",
                "singleton_axis": "zero-copy identity view for original mean(axis=1)",
                "confidence_refinement_in_original_loss": "inactive/commented",
                "positive_fraction": float(labels.mean()), "snippets": len(labels)}
    (output / "ucfsht_pseudo_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return labels


def generate_confidences(featuresTD, index, pseudo, outputdir) -> np.ndarray:
    """Loader compatibility only: original UCFSHT loss ignores confidences."""
    verify_source()
    targets = np.asarray(pseudo)
    if targets.shape != (len(featuresTD),):
        raise ValueError("Pseudo targets and feature timeline disagree")
    metadata = {"branch": BRANCH, "confidence_refinement": "inactive/commented in original train.py",
                "values": "ones; placeholder has no effect on original loss",
                "snippets": len(featuresTD)}
    directory = Path(outputdir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ucfsht_confidence_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return np.ones(len(featuresTD), dtype=np.float32)


def _original_tta(settings):
    path = ROOT / "test.py"
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    names = {"_split_prefix_suffix_video", "_tta_update_one_video"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != names:
        raise RuntimeError("Missing original ITA definitions")
    scope = {"torch": torch, "F": torch.nn.functional,
             "args": SimpleNamespace(warmup_segments=settings["k"], tta_q=settings["q"],
                 tta_min_keep=settings["min_keep"], tta_lr=settings["lr"],
                 tta_steps_per_video=settings["steps"])}
    # Compile the original bodies/signatures verbatim, omitting CLI parse_args.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), scope)
    return scope["_tta_update_one_video"]


def adapt_one_video(model, adapter, featuresTD_tensor, settings):
    verify_source()
    if featuresTD_tensor.ndim != 2 or featuresTD_tensor.shape[0] < 1:
        raise ValueError("ITA input must be a nonempty (T,D) tensor")
    if (settings["k"] < 1 or not 0 <= settings["q"] <= 1
            or settings["min_keep"] < 1 or settings["lr"] <= 0 or settings["steps"] < 0):
        raise ValueError("Invalid ITA settings")
    updated, debug = _original_tta(settings)(
        x_video=featuresTD_tensor, adapter_episode=adapter, model=model,
        q=settings["q"], min_keep=settings["min_keep"], tta_lr=settings["lr"],
        tta_steps_per_video=settings["steps"], adapt_prefix_only=True,
        warmup_segments=settings["k"])
    updated._xd_last_adaptation_debug = debug
    updated.xd_debug = debug
    return updated
