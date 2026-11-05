"""Independent XD experiment entry point. Never imports a sibling checkout."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (auc, average_precision_score, precision_recall_curve,
                             roc_auc_score)
from torch.utils.data import DataLoader

from . import branch, data


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def environment():
    import scipy
    import sklearn
    return dict(python=platform.python_version(), platform=platform.platform(),
                torch=str(torch.__version__), numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)


def experiment_config(args):
    return {key: getattr(args, key) for key in (
        'seed', 'epochs', 'lr', 'batch_size', 'window', 'stride', 'accumulation_steps',
        'ita', 'k', 'q', 'min_keep', 'ita_lr', 'ita_steps')}


def identity(args):
    """Reject mixing datasets, algorithms or settings in a single run directory."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    code = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    assets = {p.name: sha256(p) for p in sorted((Path(__file__).parent / 'assets').glob('*'))
              if p.is_file()}
    record = dict(branch=branch.verify_source(), config=experiment_config(args),
                  implementation_sha256=code, assets_sha256=assets,
                  cache_manifest_sha256=sha256(Path(args.cache) / 'manifest.json'))
    target = output / 'identity.json'
    if target.exists():
        if json.loads(target.read_text('utf-8')) != record:
            raise ValueError('Run identity differs; choose a new --output directory')
    else:
        write_json(target, record)
        write_json(output / 'environment.json', environment())
    return output


def validate_labels(labels, total, name, binary=False):
    labels = np.asarray(labels, dtype=np.float32)
    if labels.shape != (total,) or not np.isfinite(labels).all():
        raise ValueError(f'{name}: invalid shape or nonfinite values')
    if binary and not np.isin(labels, [0, 1]).all():
        raise ValueError(f'{name}: nonbinary pseudo labels')
    if np.any(labels < 0) or np.any(labels > 1):
        raise ValueError(f'{name}: values outside [0,1]')
    return labels


def pseudo_stage(args):
    cache = data.load_cache(args.cache, training=True)
    output = identity(args)
    if (output / 'pseudo.npy').exists() or (output / 'confidence.npy').exists():
        raise FileExistsError('Pseudo outputs exist; choose a fresh run directory')
    labels = validate_labels(branch.generate_pseudo(cache['features'], cache['index'], output),
                             len(cache['features']), 'pseudo', binary=True)
    confidence = validate_labels(branch.generate_confidences(
        cache['features'], cache['index'], labels, output), len(labels), 'confidence')
    np.save(output / 'pseudo.npy', labels, allow_pickle=False)
    np.save(output / 'confidence.npy', confidence, allow_pickle=False)
    write_json(output / 'pseudo_metadata.json', dict(
        source='XD train features only; no ground-truth training labels',
        pseudo_sha256=sha256(output / 'pseudo.npy'),
        confidence_sha256=sha256(output / 'confidence.npy'),
        positive_fraction=float(labels.mean()), confidence_min=float(confidence.min()),
        confidence_max=float(confidence.max()), per_video_positive_fraction=[
            float(labels[a:b].mean()) for a, b in cache['index']]))


def train_stage(args):
    cache = data.load_cache(args.cache, training=True)
    output = identity(args)
    if (output / 'final.pt').exists() or (output / 'train.csv').exists():
        raise FileExistsError('Training artifacts exist; no implicit overwrite/resume')
    pseudo_meta = json.loads((output / 'pseudo_metadata.json').read_text('utf-8'))
    for filename, key in [('pseudo.npy', 'pseudo_sha256'), ('confidence.npy', 'confidence_sha256')]:
        if sha256(output / filename) != pseudo_meta[key]:
            raise ValueError(f'Changed {filename}')
    labels = validate_labels(np.load(output / 'pseudo.npy', allow_pickle=False),
                             len(cache['features']), 'pseudo', True)
    confidence = validate_labels(np.load(output / 'confidence.npy', allow_pickle=False),
                                 len(labels), 'confidence')
    dataset = data.WindowDataset(cache['features'], cache['index'], labels, confidence,
                                 args.window, args.stride)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=data.collate, drop_last=False)
    model = branch.make_model(cache['features'].shape[1]).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4,
                                momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.1)
    with (output / 'train.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['epoch', 'lr', 'loss'])
        for epoch in range(1, args.epochs + 1):
            lr = optimizer.param_groups[0]['lr']
            loss = branch.train_epoch(loader, model, optimizer, epoch, torch.device(args.device),
                                      accumulation_steps=args.accumulation_steps)
            if not np.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            writer.writerow([epoch, lr, loss])
            stream.flush()
            scheduler.step()
    torch.save(model.state_dict(), output / 'final.pt')
    write_json(output / 'checkpoint.json', dict(
        sha256=sha256(output / 'final.pt'), dimension=cache['features'].shape[1],
        epochs=args.epochs, seed=args.seed, selection='predeclared fixed final epoch',
        train_video_count=len(cache['ids']), train_windows=len(dataset),
        test_features_or_gt_read_during_training=False, environment=environment()))


def frame_metrics(gt, scores):
    gt, scores = np.asarray(gt), np.asarray(scores)
    if gt.ndim != 1 or gt.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError('Frame GT/score length or finiteness mismatch')
    if not np.isin(gt, [0, 1]).all() or len(np.unique(gt)) != 2:
        raise ValueError('Pooled metric requires binary GT containing both classes')
    precision, recall, _ = precision_recall_curve(gt, scores)
    values = dict(ap=float(average_precision_score(gt, scores)),
                  official_pr_auc=float(auc(recall, precision)),
                  roc_auc=float(roc_auc_score(gt, scores)))
    return dict(**values, **{key + '_percent': val * 100 for key, val in values.items()},
                frames=len(gt), anomalous_frames=int(np.count_nonzero(gt)))


def predict(model, features, adapter=None):
    model.eval()
    if adapter is not None:
        adapter.eval()
    with torch.inference_mode():
        x = features.unsqueeze(0)
        if adapter is not None:
            x = adapter(x)
        scores = model(x).reshape(-1).cpu().numpy()
    if len(scores) != len(features) or not np.isfinite(scores).all():
        raise ValueError('Detector returned invalid clip scores')
    return scores


def evaluate_stage(args):
    cache = data.load_cache(args.cache, training=False)
    output = identity(args)
    score_dir = output / 'scores'
    if score_dir.exists() or (output / 'metrics.json').exists():
        raise FileExistsError('Evaluation artifacts exist; no implicit overwrite')
    checkpoint = json.loads((output / 'checkpoint.json').read_text('utf-8'))
    if sha256(output / 'final.pt') != checkpoint['sha256']:
        raise ValueError('Checkpoint hash mismatch')
    dimension = cache['features'].shape[1]
    if checkpoint['dimension'] != dimension:
        raise ValueError('Training/test feature dimension differs')
    model = branch.make_model(dimension).to(args.device)
    model.load_state_dict(torch.load(output / 'final.pt', map_location=args.device, weights_only=True), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adapter_initial = branch.make_adapter(dimension).to(args.device).eval() if args.ita else None
    score_dir.mkdir()
    raw_parts, ln_full_parts, ln_parts, ita_parts, suffix_gt_parts, coverage = [], [], [], [], [], []
    settings = dict(k=args.k, q=args.q, min_keep=args.min_keep, lr=args.ita_lr, steps=args.ita_steps)
    for position, (video_id, (start, end)) in enumerate(zip(cache['ids'], cache['index'])):
        start, end = int(start), int(end)
        features = torch.from_numpy(np.array(cache['features'][start:end], copy=True)).to(args.device)
        gt = np.asarray(cache['gt'][start * 16:end * 16])
        if len(gt) != len(features) * 16:
            raise ValueError(f'Per-video GT mapping mismatch: {video_id}')
        raw = predict(model, features)
        raw_parts.append(np.repeat(raw, 16))
        saved = dict(raw_clip=raw, gt_frame=gt, frames_per_clip=np.array(16),
                     video_id=np.array(video_id))
        row = dict(video_id=video_id, clips=len(features), full_frames=len(gt))
        if args.ita:
            adapter = copy.deepcopy(adapter_initial)
            ln = predict(model, features, adapter)
            ln_full_parts.append(np.repeat(ln, 16))
            cut = min(args.k, len(features))
            if len(features) > cut:
                adapter = branch.adapt_one_video(model, adapter, features, settings)
                ita = predict(model, features, adapter)
                row['adapted'] = True
            else:
                ita = ln.copy()
                row['adapted'] = False
                row['reason'] = 'No suffix after fixed k; retained in full raw metrics'
            ln_parts.append(np.repeat(ln[cut:], 16))
            ita_parts.append(np.repeat(ita[cut:], 16))
            suffix_gt_parts.append(gt[cut * 16:])
            row.update(prefix_clips=cut, excluded_frames=cut * 16,
                       suffix_frames=len(gt) - cut * 16)
            saved.update(ln_clip=ln, ita_clip=ita, suffix_start_clip=np.array(cut))
        coverage.append(row)
        # Numeric file names avoid Windows reserved/long names; IDs are inside NPZ and manifest.
        np.savez_compressed(score_dir / f'{position:04d}.npz', **saved)
    metrics = dict(
        primary_metric='frame AP = sklearn.average_precision_score; pooled continuous scores',
        companion_metric='official_pr_auc = auc(recall,precision), NOT sklearn AP',
        units='fractions [0,1] and explicitly suffixed _percent fields',
        frame_domain='official feature-aligned T*16 frames; no raw tail extrapolation',
        crop_policy='mean of five actual RGB crop FEATURES before detector',
        raw_full=frame_metrics(cache['gt'], np.concatenate(raw_parts)),
        branch=branch.BRANCH, source_commit=branch.COMMIT,
        official_test_videos=len(cache['ids']),
        test_gt_used_for_model_selection=False,
    )
    if args.ita:
        gt_suffix = np.concatenate(suffix_gt_parts)
        metrics.update(ln_full=frame_metrics(cache['gt'], np.concatenate(ln_full_parts)),
                       ln_suffix=frame_metrics(gt_suffix, np.concatenate(ln_parts)),
                       ita_suffix=frame_metrics(gt_suffix, np.concatenate(ita_parts)),
                       legacy_ita_preserved=True,
                       suffix_videos=sum(row['suffix_frames'] > 0 for row in coverage),
                       excluded_frames=sum(row['excluded_frames'] for row in coverage))
        metrics['ita_minus_ln_suffix_ap'] = metrics['ita_suffix']['ap'] - metrics['ln_suffix']['ap']
    write_json(output / 'coverage.json', coverage)
    write_json(output / 'metrics.json', metrics)
    print(json.dumps(metrics, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['prepare', 'pseudo', 'train', 'evaluate', 'all'])
    parser.add_argument('--features', help='Directory containing RGB and RGBTest; no extraction')
    parser.add_argument('--cache', required=True)
    parser.add_argument('--output', default='xd_runs/seed42')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--window', type=int, default=500)
    parser.add_argument('--stride', type=int, default=500)
    parser.add_argument('--accumulation-steps', type=int, default=1)
    parser.add_argument('--ita', action='store_true', help='Also evaluate original per-video prefix ITA')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--q', type=float, default=1.0)
    parser.add_argument('--min-keep', type=int, default=3)
    parser.add_argument('--ita-lr', type=float, default=0.01)
    parser.add_argument('--ita-steps', type=int, default=30)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.window, args.stride, args.accumulation_steps,
           args.k, args.min_keep, args.ita_steps) < 1 or args.lr <= 0 or args.ita_lr <= 0:
        parser.error('Lengths/counts and learning rates must be positive')
    if not 0 < args.q <= 1:
        parser.error('--q must be in (0,1]')
    if args.stride > args.window:
        parser.error('--stride must not exceed --window')
    if args.stage in ['prepare', 'all']:
        if not args.features:
            parser.error('--features is required to prepare the feature cache')
        if args.stage == 'prepare' or not (Path(args.cache) / 'manifest.json').exists():
            data.prepare(args.features, args.cache)
        if args.stage == 'prepare':
            return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    if args.device == 'cpu':
        torch.set_num_threads(2)
    seed_all(args.seed)
    branch.verify_source()
    if args.stage in ['pseudo', 'all']:
        pseudo_stage(args)
    if args.stage in ['train', 'all']:
        train_stage(args)
    if args.stage in ['evaluate', 'all']:
        evaluate_stage(args)


if __name__ == '__main__':
    main()
