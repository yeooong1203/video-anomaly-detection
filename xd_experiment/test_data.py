"""Meaningful CPU data checks; synthetic values are never benchmark results."""
import hashlib
import json
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from xd_experiment import data


class XDDataTests(unittest.TestCase):
    def setUp(self):
        # Keep temporary writes in this independent workspace on restricted PCs.
        parent = Path(__file__).resolve().parents[1] / ".xd-test-temp"
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def crops(self, directory, video, length=3, dimension=4):
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for crop in range(5):
            path = directory / f"{video}__{crop}.npy"
            np.save(path, np.full((length, dimension), crop, dtype=np.float32))
            paths.append(path)
        return paths

    def fixture(self):
        feature_root = self.root / "features"
        self.crops(feature_root / "RGB", "train_label_A", length=2)
        self.crops(feature_root / "RGBTest", "abnormal_label_B", length=3)
        self.crops(feature_root / "RGBTest", "normal_label_A", length=2)
        assets = self.root / "assets"
        assets.mkdir()
        ids = ["abnormal_label_B", "normal_label_A"]
        (assets / "official_test_ids.json").write_text(json.dumps(ids))
        (assets / "official_train_ids.json").write_text('["train_label_A"]')
        # Two intervals touch the first frame and extend beyond the covered tail.
        (assets / "official_annotations.txt").write_text("abnormal_label_B.mp4 0 2 45 53\n")
        expected = np.zeros(80, dtype=np.uint8)
        expected[:2] = 1
        expected[45:48] = 1
        protocol = {"dimension": 4, "crops": 5, "frames_per_snippet": 16,
                    "train_videos": 1, "test_videos": 2, "expected_gt_frames": 80,
                    "expected_gt_uint8_sha256": hashlib.sha256(expected.tobytes()).hexdigest()}
        (assets / "protocol.json").write_text(json.dumps(protocol))
        return feature_root, assets, protocol, expected

    def load_fixture_cache(self, cache, assets, protocol, **kwargs):
        with patch.object(data, "ASSETS", assets), patch.object(data, "_protocol", return_value=protocol):
            return data.load_cache(cache, **kwargs)

    def test_actual_crop_mean_preserves_temporal_rows(self):
        paths = self.crops(self.root / "RGB", "example", length=2)
        result = data.read_video_crops(paths)
        self.assertEqual(result.shape, (2, 4))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, np.full((2, 4), 2, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "Exactly five"):
            data.read_video_crops(paths[:4])
        with self.assertRaisesRegex(ValueError, "each real crop"):
            data.read_video_crops(paths[:4] + paths[:1])

    def test_rejects_inconsistent_crop_and_nonfinite(self):
        paths = self.crops(self.root / "RGB", "example")
        np.save(paths[-1], np.zeros((4, 4), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "Crop temporal"):
            data.discover(self.root / "RGB", dimension=4)
        np.save(paths[-1], np.full((3, 4), np.nan, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            data.read_video_crops(paths)

    def test_prepare_roundtrip_gt_hash_and_no_test_dependency_in_train(self):
        root, assets, protocol, expected_gt = self.fixture()
        cache = self.root / "cache"
        with patch.object(data, "ASSETS", assets), patch.object(data, "_protocol", return_value=protocol):
            manifest = data.prepare(root, cache)
        self.assertEqual(manifest["test_frames"], 80)
        test = self.load_fixture_cache(cache, assets, protocol)
        np.testing.assert_array_equal(test["gt"], expected_gt)
        np.testing.assert_array_equal(test["features"], np.full((5, 4), 2, dtype=np.float32))
        del test
        # A valid train stage must not require or read GT/test payloads.
        (cache / "gt.npy").rename(cache / "gt.unavailable")
        (cache / "test.npy").rename(cache / "test.unavailable")
        train = self.load_fixture_cache(cache, assets, protocol, training=True)
        self.assertEqual(train["features"].shape, (2, 4))
        del train
        # Changed cached payload is rejected before use.
        changed = np.lib.format.open_memmap(cache / "train.npy", mode="r+")
        changed[0, 0] += 1
        changed.flush()
        del changed
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.load_fixture_cache(cache, assets, protocol, training=True)

    def test_rejects_wrong_split_and_gt_alignment(self):
        root, assets, protocol, _ = self.fixture()
        with patch.object(data, "ASSETS", assets), patch.object(data, "_protocol", return_value=protocol):
            altered = protocol.copy()
            altered["expected_gt_uint8_sha256"] = "0" * 64
            with patch.object(data, "_protocol", return_value=altered):
                with self.assertRaisesRegex(ValueError, "GT differs"):
                    data.prepare(root, self.root / "wrong_gt")
            (assets / "official_test_ids.json").write_text('["missing_label_A"]')
            with self.assertRaisesRegex(ValueError, "membership"):
                data.prepare(root, self.root / "wrong_split")

    def test_original_feature_tree_is_never_a_cache_target(self):
        root, assets, protocol, _ = self.fixture()
        with patch.object(data, "ASSETS", assets), patch.object(data, "_protocol", return_value=protocol):
            with self.assertRaisesRegex(ValueError, "separate"):
                data.prepare(root, root / "cache")
            self.assertFalse((root / "cache").exists())

    def test_runtime_rejects_changed_metadata(self):
        root, assets, protocol, _ = self.fixture()
        cache = self.root / "cache"
        with patch.object(data, "ASSETS", assets), patch.object(data, "_protocol", return_value=protocol):
            data.prepare(root, cache)
            (assets / "official_annotations.txt").write_text("abnormal_label_B.mp4 0 1\n")
            with self.assertRaisesRegex(ValueError, "metadata asset"):
                data.load_cache(cache, training=True)

    def test_original_end_window_and_masked_short_video(self):
        features = np.arange(15 * 4, dtype=np.float32).reshape(15, 4)
        index = np.array([[0, 2], [2, 15]], dtype=np.int64)
        dataset = data.WindowDataset(features, index, np.zeros(15), np.ones(15), window=5, stride=4)
        self.assertEqual(dataset.windows, [(0, 0, 2), (1, 2, 7), (1, 6, 11), (1, 10, 15)])
        batch = data.collate([dataset[0], dataset[1]])
        self.assertEqual(tuple(batch[0].shape), (2, 5, 4))
        self.assertEqual(batch[3][0].tolist(), [1, 1, 0, 0, 0])
        np.testing.assert_array_equal(batch[0][0, :2], features[:2])

    def test_mmap_dataset_reopens_after_pickling(self):
        path = self.root / "features.npy"
        np.save(path, np.ones((2, 4), dtype=np.float32))
        features = np.load(path, mmap_mode="r")
        dataset = data.WindowDataset(features, np.array([[0, 2]]), np.zeros(2), np.ones(2), 5, 5)
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertIsNone(restored._features)
        self.assertTrue(torch.equal(restored[0][0], torch.ones(2, 4)))


if __name__ == "__main__":
    unittest.main()
