"""CPU integration/parity checks; all generated scores are synthetic diagnostics.

The same test file runs independently in each checkout against its own branch.
Cache loading is mocked only for tiny runtime fixtures; strict real cache/GT
validation is covered separately by test_data.py.
"""
from __future__ import annotations

import ast
import copy
import gc
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from xd_experiment import branch, data, run


class XDCommonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        parent = Path(__file__).resolve().parents[1] / '.xd-test-temp'
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temp.name)
        run.seed_all(42)

    def tearDown(self):
        gc.collect()
        self.temp.cleanup()

    def runtime_fixture(self):
        cache = self.root / 'cache'
        cache.mkdir()
        rng = np.random.default_rng(42)
        np.save(cache / 'train.npy', rng.normal(size=(22, 8)).astype(np.float32))
        np.save(cache / 'test.npy', rng.normal(size=(25, 8)).astype(np.float32))
        gt = np.zeros(25 * 16, dtype=np.uint8)
        gt[1:3] = 1
        gt[96:128] = 1
        gt[308:338] = 1
        np.save(cache / 'gt.npy', gt)
        (cache / 'manifest.json').write_text(json.dumps({'synthetic_fixture': True}))
        args = SimpleNamespace(cache=str(cache), output=str(self.root / 'run'),
            seed=42, epochs=1, lr=.001, batch_size=2, window=8, stride=6,
            accumulation_steps=1, ita=True, k=5, q=1., min_keep=3,
            ita_lr=.01, ita_steps=1, device='cpu')

        def load_cache(location, training=False):
            self.assertEqual(Path(location), cache)
            if training:
                # No test payload/GT is read or even stat'ed along this path.
                return dict(features=np.load(cache / 'train.npy', allow_pickle=False),
                    index=np.array([[0, 2], [2, 10], [10, 22]], dtype=np.int64),
                    ids=['synthetic_train_short', 'synthetic_train_medium', 'synthetic_train_long'])
            return dict(features=np.load(cache / 'test.npy', allow_pickle=False),
                index=np.array([[0, 5], [5, 13], [13, 25]], dtype=np.int64),
                ids=['synthetic_short_T5', 'synthetic_medium_T8', 'synthetic_long_T12'],
                gt=np.load(cache / 'gt.npy', allow_pickle=False))
        return args, cache, load_cache

    def test_runtime_real_branch_pseudo_train_evaluate_and_reset(self):
        args, cache, load_cache = self.runtime_fixture()
        provenance = branch.verify_source()
        self.assertTrue(provenance)
        with patch.object(data, 'load_cache', side_effect=load_cache):
            run.pseudo_stage(args)
            labels = np.load(Path(args.output) / 'pseudo.npy', allow_pickle=False)
            self.assertEqual(labels.shape, (22,))
            self.assertTrue(np.isin(labels, [0, 1]).all())
            # Physical absence proves training has no GT/test payload dependency.
            (cache / 'gt.npy').rename(cache / 'gt.unavailable')
            (cache / 'test.npy').rename(cache / 'test.unavailable')
            try:
                run.train_stage(args)
            finally:
                (cache / 'gt.unavailable').rename(cache / 'gt.npy')
                (cache / 'test.unavailable').rename(cache / 'test.npy')
            checkpoint = json.loads((Path(args.output) / 'checkpoint.json').read_text('utf-8'))
            self.assertFalse(checkpoint['test_features_or_gt_read_during_training'])
            self.assertEqual(checkpoint['epochs'], 1)
            self.assertEqual(checkpoint['dimension'], 8)

            original_adapt = branch.adapt_one_video
            initial_states = []

            def checked_adapt(model, adapter, features, settings):
                initial_states.append(copy.deepcopy(adapter.state_dict()))
                detector_before = copy.deepcopy(model.state_dict())
                delta_before = adapter.delta.weight.detach().clone()
                result = original_adapt(model, adapter, features, settings)
                for key, value in detector_before.items():
                    self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
                self.assertTrue(torch.equal(delta_before, result.delta.weight))
                self.assertTrue(result.ln.weight.requires_grad)
                self.assertFalse(result.delta.weight.requires_grad)
                self.assertTrue(all(torch.isfinite(v).all() for v in result.state_dict().values()))
                return result

            with patch.object(branch, 'adapt_one_video', side_effect=checked_adapt):
                run.evaluate_stage(args)
            self.assertEqual(len(initial_states), 2)  # T5 has no suffix and is not adapted.
            for key in initial_states[0]:
                self.assertTrue(torch.equal(initial_states[0][key], initial_states[1][key]), key)

        metrics = json.loads((Path(args.output) / 'metrics.json').read_text('utf-8'))
        coverage = json.loads((Path(args.output) / 'coverage.json').read_text('utf-8'))
        self.assertEqual(metrics['raw_full']['frames'], 400)
        self.assertEqual(metrics['ln_full']['frames'], 400)
        self.assertEqual(metrics['ln_suffix']['frames'], 160)
        self.assertEqual(metrics['ita_suffix']['frames'], 160)
        self.assertEqual(metrics['excluded_frames'], 240)
        self.assertEqual(metrics['suffix_videos'], 2)
        self.assertFalse(coverage[0]['adapted'])
        self.assertEqual(coverage[0]['full_frames'], 80)
        self.assertEqual(coverage[0]['suffix_frames'], 0)
        pooled_gt, pooled_scores = [], []
        for row, path in zip(coverage, sorted((Path(args.output) / 'scores').glob('*.npz'))):
            with np.load(path, allow_pickle=False) as saved:
                self.assertEqual(len(saved['gt_frame']), len(saved['raw_clip']) * 16)
                cut = int(saved['suffix_start_clip'])
                self.assertEqual(len(saved['gt_frame'][cut * 16:]), row['suffix_frames'])
                pooled_gt.append(saved['gt_frame'].copy())
                pooled_scores.append(np.repeat(saved['raw_clip'], 16))
        repeated_metric = run.frame_metrics(np.concatenate(pooled_gt), np.concatenate(pooled_scores))
        self.assertEqual(metrics['raw_full'], repeated_metric)
        self.assertEqual(provenance, branch.verify_source())

    def test_ap_definition_percent_and_strict_frame_checks(self):
        actual = run.frame_metrics(np.array([0, 1, 1, 0]), np.array([.8, .6, .4, .2]))
        self.assertAlmostEqual(actual['ap'], 7 / 12)
        self.assertAlmostEqual(actual['official_pr_auc'], 5 / 12)
        self.assertAlmostEqual(actual['ap_percent'], actual['ap'] * 100)
        self.assertNotEqual(actual['ap'], actual['official_pr_auc'])
        for gt, scores in [([0, 1], [.5]), ([0, 1], [np.nan, .5]),
                           ([0, 2], [.4, .5]), ([1, 1], [.4, .5])]:
            with self.assertRaises(ValueError):
                run.frame_metrics(np.asarray(gt), np.asarray(scores))

    def test_predict_is_stable_eval_and_rejects_wrong_time_length(self):
        model = branch.make_model(8).train()
        x = torch.randn(12, 8)
        first, second = run.predict(model, x), run.predict(model, x)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(model.training)
        class WrongLength(torch.nn.Module):
            def forward(self, value):
                return torch.zeros(1, 1, 1)
        with self.assertRaisesRegex(ValueError, 'invalid clip scores'):
            run.predict(WrongLength(), x)

    def test_window_and_collate_match_original_branch_loader(self):
        # Synthetic ten genuine, distinct crop arrays exercise the OLD loader.
        # They are not used to manufacture crops for real XD experiment input.
        crops = np.random.default_rng(7).normal(size=(19, 10, 8)).astype(np.float32)
        index = np.array([[0, 2], [2, 19]], dtype=np.int64)
        labels = np.arange(19).astype(np.float32) % 2
        confidence = np.linspace(.1, .9, 19).astype(np.float32)
        for name, value in [('index', index), ('labels', labels), ('confidence', confidence)]:
            np.save(self.root / f'{name}.npy', value)
        feature_path = self.root / 'original_features.npy'
        if branch.BRANCH == 'model/UCFSHT':
            feature_path.write_bytes(crops.tobytes())
        else:
            np.save(feature_path, crops)
        path = branch.ROOT / 'dataset.py'
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        wanted = {'UCFTrainVideoDataset_Stratified', 'collate_fn_variable_length'}
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in wanted]
        self.assertEqual({n.name for n in nodes}, wanted)
        scope = dict(np=np, torch=torch, data=torch.utils.data, args=SimpleNamespace(feature_size=8))
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
        original = scope['UCFTrainVideoDataset_Stratified'](
            str(feature_path), str(self.root / 'labels.npy'), str(self.root / 'index.npy'),
            str(self.root / 'confidence.npy'), window_size=6, stride=4)
        current = data.WindowDataset(crops.mean(axis=1), index, labels, confidence, 6, 4)
        self.assertEqual(original.windows, current.windows)
        for i in range(len(current)):
            a, b = original[i], current[i]
            self.assertEqual(a[-1], b[-1])
            for old_tensor, new_tensor in zip(a[:-1], b[:-1]):
                self.assertTrue(torch.equal(old_tensor, new_tensor))
        old_batch = scope['collate_fn_variable_length']([original[0], original[-1]])
        new_batch = data.collate([current[0], current[-1]])
        for a, b in zip(old_batch, new_batch):
            self.assertTrue(torch.equal(a, b))
        original.con_all._mmap.close()
        del original, current


if __name__ == '__main__':
    unittest.main()
