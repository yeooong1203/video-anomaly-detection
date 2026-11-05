"""XD I/O bridge to the unchanged model/APWCS implementation.

No implementation is imported from another checkout. Root model/train/pseudo/
confidence/adapter/test source files are loaded by absolute path after checksum
verification. Only two test function definitions are compiled to avoid the root
module's import-time CLI parsing. Their bodies/signatures remain unchanged.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
BRANCH_NAME = "model/APWCS"
SOURCE_COMMIT = "2fff83ab856417cc805fad44b55e68dc820fd03e"
BRANCH = BRANCH_NAME
COMMIT = SOURCE_COMMIT
SOURCE_SHA256 = {
    "model.py": "801c8ac2907bbe64d8954c19558516a6074c2f1b14a0b1d56762fee99612215a",
    "train.py": "447d67e2ea82a2d9ce2ca912044a1f44c5f08109cafe1d55e2518723b180ce82",
    "adapter.py": "619bf0f5ffd6d4432c65bdd476f3f0d0e2e95fc0fb4c41efce3a16ffb5896a1e",
    "test.py": "69ac57136f34b6683d7bcab15842e6a21d4e327a6761ae078575d7fbb2729b97",
    "confidence.py": "2366e71d6e5d21f8d90dc004dffd7b0a4e245c591091a8b8599c830b144c4018",
    "pseudo_proto_dist_swap.py": "63ffadd535b8224e59bd1b1b7ae91af14f87d3619ab66487714c9e37c6057695",
}
PSEUDO_SETTINGS = {
    "feature_normalization": "standard",
    "threshold_method": "p_value",
    "score_normalization": "zscore",
    "prototype_method": "none",
    "use_attraction": True,
    "attraction_strength": 0.8,
    "attraction_iterations": 3,
    "repulsion_strength": 0.7,
    "remove_isolated_abn": True,
    "isolated_abn_min_length": 2,
    "fill_isolated_norm": True,
    "isolated_norm_max_gap": 2,
    "use_prototype_swap": True,
    "swap_threshold": 0.8,
}
CONFIDENCE_ACTIVE = True


def provenance() -> dict:
    """Verify source integrity and return serializable branch-specific settings."""
    for name, expected in SOURCE_SHA256.items():
        canonical = (ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        actual = hashlib.sha256(canonical).hexdigest()
        if actual != expected:
            raise RuntimeError(f"{BRANCH_NAME} original source changed: {name}")
    return {
        "branch": BRANCH_NAME,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": dict(SOURCE_SHA256),
        "source_hash_policy": "SHA256 bytes after CRLF-to-LF only; all other bytes retained",
        "model": "model.py:Model_V2_AllCNN",
        "loss": "train.py:adaptive_hybrid_loss; BCE / 0.5*MSE, threshold .3",
        "confidence_active": CONFIDENCE_ACTIVE,
        "confidence_source": "confidence.py:generate_confidence_scores",
        "pseudo_source": "pseudo_proto_dist_swap.py:generate_improved_pseudo_labels",
        "pseudo_settings": dict(PSEUDO_SETTINGS),
        "pseudo_gaussian_tail_probability": 0.05,
        "pseudo_selection_reason": "train.sh *_ng.npy; generator header explicitly names that artifact",
        "pseudo_history_verified": False,
        "ita": "test.py:_tta_update_one_video; original function body",
        "legacy_ita_shape_and_broadcast_preserved": True,
    }


def verify_source() -> dict:
    """Shared runner entry point for branch/source provenance validation."""
    return provenance()


@functools.lru_cache(maxsize=None)
def _module(filename: str):
    provenance()
    name = "_xd_apwcs_" + Path(filename).stem
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _AveragedFeatureView:
    """Expose already crop-averaged TD to original crop-mean routines.

    The singleton axis denotes one already reduced feature, not synthetic crops:
    original mean(axis=1) is identity. No feature values are duplicated/padded.
    """

    def __init__(self, features):
        self.features = features
        self.shape = (features.shape[0], 1, features.shape[1])

    def __getitem__(self, item):
        return np.asarray(self.features[item])[:, None, :]


def _validate_inputs(features, index):
    if getattr(features, "ndim", None) != 2 or features.shape[1] <= 0:
        raise ValueError("Expected actual crop-averaged features (T,D)")
    boundaries = np.asarray(index)
    if boundaries.ndim != 2 or boundaries.shape[1] != 2 or len(boundaries) == 0:
        raise ValueError("Expected nonempty video boundaries (N,2)")
    if not np.issubdtype(boundaries.dtype, np.integer):
        raise ValueError("Video boundaries must be integers")
    if (boundaries[0, 0] != 0 or boundaries[-1, 1] != len(features)
            or np.any(boundaries[:, 1] <= boundaries[:, 0])
            or np.any(boundaries[1:, 0] != boundaries[:-1, 1])):
        raise ValueError("Video boundaries must cover features contiguously")
    return boundaries


def make_model(dimension: int) -> torch.nn.Module:
    provenance()
    return _module("model.py").Model_V2_AllCNN(int(dimension), kernel_size=5)


def train_epoch(loader, model, optimizer, epoch, device, accumulation_steps=1) -> float:
    provenance()
    return float(_module("train.py").concatenated_train_variable_length(
        loader, model, optimizer, epoch, device,
        accumulation_steps=accumulation_steps,
    ))


def generate_pseudo(features, index, outputdir):
    """Use the APWCS frozen *_ng generator and its own main() settings."""
    provenance()
    boundaries = _validate_inputs(features, index)
    output = Path(outputdir)
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg", force=True)  # original diagnostic export, no GUI/Tk dependency
    pseudo_module = _module("pseudo_proto_dist_swap.py")
    # Preserve the original diagnostic while keeping its output inside this run.
    original_visualize = pseudo_module.visualize_score_distribution
    pseudo_module.visualize_score_distribution = functools.partial(
        original_visualize, save_path=str(output / "apwcs_pseudo_score_distribution.png")
    )
    try:
        parts = pseudo_module.generate_improved_pseudo_labels(
            _AveragedFeatureView(features), boundaries, **PSEUDO_SETTINGS
        )
    finally:
        pseudo_module.visualize_score_distribution = original_visualize
    labels = np.concatenate(parts).astype(np.float32)
    if labels.shape != (len(features),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("APWCS pseudo generator produced invalid targets")
    (output / "apwcs_branch_provenance.json").write_text(
        json.dumps(provenance(), indent=2) + "\n", encoding="utf-8"
    )
    return labels


def generate_confidences(features, index, pseudo_labels, outputdir=None):
    """Keep APWCS confidence refinement active, using only train pseudo labels."""
    provenance()
    boundaries = _validate_inputs(features, index)
    labels = np.asarray(pseudo_labels)
    if labels.shape != (len(features),) or not np.isin(labels, [0, 1]).all():
        raise ValueError("APWCS confidence requires one binary pseudo label per clip")
    parts = _module("confidence.py").generate_confidence_scores(
        _AveragedFeatureView(features), boundaries, labels
    )
    confidence = np.concatenate(parts).astype(np.float32)
    if confidence.shape != labels.shape or not np.isfinite(confidence).all():
        raise ValueError("Invalid APWCS confidence scores")
    if np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("APWCS confidence outside [0,1]")
    if outputdir is not None:
        output = Path(outputdir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "apwcs_confidence_provenance.json").write_text(
            json.dumps({"source": "confidence.py:generate_confidence_scores",
                        "active_in_training": True,
                        "sha256": SOURCE_SHA256["confidence.py"]}, indent=2) + "\n",
            encoding="utf-8",
        )
    return confidence


def make_adapter(dimension: int) -> torch.nn.Module:
    provenance()
    return _module("adapter.py").ResidualAdapter2048(d=int(dimension), use_ln=True)


@functools.lru_cache(maxsize=1)
def _ita_function():
    provenance()
    path = ROOT / "test.py"
    tree = ast.parse(path.read_text("utf-8"), filename=str(path))
    wanted = {"_split_prefix_suffix_video", "_tta_update_one_video"}
    definitions = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in definitions} != wanted:
        raise RuntimeError("APWCS original ITA definitions missing")
    namespace = {
        "torch": torch, "F": F, "np": np,
        "args": SimpleNamespace(tta_q=1.0, tta_min_keep=3, tta_lr=0.01,
                                tta_steps_per_video=30, warmup_segments=5),
    }
    # Compile original nodes without changing any signature, expression or body.
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["_tta_update_one_video"]


def adapt_one_video(model, adapter, features, settings):
    """Caller creates/reset adapter per video; legacy ITA arithmetic is retained."""
    provenance()
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("ITA expects a nonempty (T,D) feature tensor")

    def value(name, default):
        return settings.get(name, default) if isinstance(settings, dict) else getattr(settings, name, default)

    k = int(value("k", 5))
    if k < 1:
        raise ValueError("APWCS legacy prefix ITA requires k >= 1")
    updated, debug = _ita_function()(
        x_video=features,
        adapter_episode=adapter,
        model=model,
        q=float(value("q", 1.0)),
        min_keep=int(value("min_keep", 3)),
        tta_lr=float(value("lr", 0.01)),
        tta_steps_per_video=int(value("steps", 30)),
        adapt_prefix_only=True,
        warmup_segments=k,
    )
    updated._xd_ita_debug = debug
    updated.eval()
    return updated
