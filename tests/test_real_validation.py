import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from configs import Config
from data_mini import MiniCharucoDataset
from prepare_real_val import prepare_real_validation
from real_validation import decode_heatmap_offset
from transformations import Transformation


class TestMiniRealValidation(unittest.TestCase):
    def _make_config(self, input_size=(32, 24)):
        return Config(
            board_name="DICT_4X4_50",
            row_count=6,
            col_count=9,
            square_len=0.085,
            marker_len=0.045,
            input_size=list(input_size),
            num_workers=0,
            bs_train=1,
            bs_train_rn=1,
            bs_val=1,
            bs_val_rn=1,
            train_labels="",
            val_labels="",
            train_images="",
            val_images="",
            use_real_val=True,
            real_val_dir="",
            real_val_batch_size=1,
            real_val_num_workers=0,
            real_val_every=1,
            val_every=1,
        )

    def _write_image(self, path, size=(32, 24)):
        image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        image[:] = (128, 128, 128)
        cv2.circle(image, (10, 10), 3, (255, 0, 0), -1)
        cv2.imwrite(str(path), image)

    def test_prepare_manifest_and_real_dataset_loading(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            raw_images = temp_dir / "raw_images"
            raw_csvs = temp_dir / "raw_csvs"
            raw_images.mkdir()
            raw_csvs.mkdir()

            image_path = raw_images / "sample1.png"
            self._write_image(image_path, size=(64, 48))

            csv_path = raw_csvs / "sample1.csv"
            with csv_path.open("w", encoding="utf-8") as f:
                f.write("corner_id,x,y,valid\n")
                f.write("0,10,10,1\n")
                f.write("1,20,20,1\n")

            output_root = temp_dir / "prepared"
            prepare_real_validation(
                images_dir=raw_images,
                csv_dir=raw_csvs,
                output_root=output_root,
                expected_count=1,
                mode="copy",
            )

            manifest_path = output_root / "manifest.csv"
            self.assertTrue(manifest_path.exists())
            self.assertTrue((output_root / "images" / "sample1.png").exists())
            self.assertTrue((output_root / "corners" / "sample1.csv").exists())

            config = self._make_config(input_size=(32, 24))
            config.real_val_dir = str(output_root)
            dataset = MiniCharucoDataset(
                config,
                str(manifest_path),
                str(output_root),
                validation=True,
                visualize=False,
                real_val=True,
                return_raw=True,
            )

            item = dataset[0]
            self.assertIn("image", item)
            self.assertIn("keypoints", item)
            self.assertIn("ids", item)
            self.assertIn("valid_mask", item)
            self.assertIn("image_id", item)
            self.assertIn("orig_size", item)
            self.assertEqual(item["image"].shape[1:], (24, 32))
            self.assertEqual(item["image_id"], "sample1")
            self.assertEqual(item["orig_size"], (48, 64))
            self.assertEqual(item["valid_mask"].sum(), 2)
            self.assertEqual(item["keypoints"].shape[1], 2)

    def test_real_csv_formats_with_id_and_visible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            raw_images = temp_dir / "raw_images"
            raw_csvs = temp_dir / "raw_csvs"
            raw_images.mkdir()
            raw_csvs.mkdir()

            image_path = raw_images / "sample2.png"
            self._write_image(image_path, size=(64, 48))

            csv_path = raw_csvs / "sample2.csv"
            with csv_path.open("w", encoding="utf-8") as f:
                f.write("id,x,y,visible\n")
                f.write("0,15,15,1\n")
                f.write("1,30,10,0\n")
                f.write("2,25,20,1\n")

            output_root = temp_dir / "prepared2"
            prepare_real_validation(
                images_dir=raw_images,
                csv_dir=raw_csvs,
                output_root=output_root,
                expected_count=1,
                mode="copy",
            )

            config = self._make_config(input_size=(32, 24))
            config.real_val_dir = str(output_root)
            dataset = MiniCharucoDataset(
                config,
                str(output_root / "manifest.csv"),
                str(output_root),
                validation=True,
                visualize=False,
                real_val=True,
                return_raw=True,
            )

            item = dataset[0]
            self.assertEqual(int(item["valid_mask"].sum()), 2)
            self.assertEqual(item["ids"].tolist(), [0, 2])

    def test_heatmap_and_offset_share_floor_cell(self):
        config = self._make_config(input_size=(32, 24))
        dataset = MiniCharucoDataset.__new__(MiniCharucoDataset)
        dataset.stride = int(config.mini_stride)
        dataset.sigma = float(config.mini_heatmap_sigma)
        dataset.n_corners = int(config.n_ids)

        keypoints = np.asarray([[10.25, 7.75], [27.9, 18.1]], dtype=np.float32)
        keypoint_ids = np.asarray([0, 39], dtype=np.int64)
        heatmap, offset, offset_mask = dataset._build_targets(
            image_shape=(24, 32),
            keypoints=keypoints,
            keypoint_ids=keypoint_ids,
            isnegative=False,
        )

        expected_cells = np.floor(keypoints / dataset.stride).astype(np.int64)
        for corner_id, (expected_x, expected_y) in zip(keypoint_ids, expected_cells):
            peak_y, peak_x = np.unravel_index(
                np.argmax(heatmap[corner_id]), heatmap[corner_id].shape
            )
            self.assertEqual((peak_x, peak_y), (expected_x, expected_y))
            self.assertEqual(float(heatmap[corner_id, peak_y, peak_x]), 1.0)
            self.assertEqual(float(offset_mask[0, peak_y, peak_x]), 1.0)

        decoded = decode_heatmap_offset(
            torch.from_numpy(heatmap).unsqueeze(0),
            torch.from_numpy(offset).unsqueeze(0),
            stride=dataset.stride,
        )[0].numpy()
        np.testing.assert_allclose(decoded[keypoint_ids], keypoints, atol=1e-5)

    def test_training_board_uses_square_cells(self):
        config = self._make_config(input_size=(320, 240))
        transformation = Transformation(config)

        self.assertEqual(transformation.board_img.shape[:2], (213, 320))

        corner_grid = transformation.corners.reshape(
            config.row_count - 1,
            config.col_count - 1,
            2,
        )
        horizontal_spacing = np.diff(corner_grid, axis=1)[..., 0].mean()
        vertical_spacing = np.diff(corner_grid, axis=0)[..., 1].mean()
        self.assertAlmostEqual(horizontal_spacing, vertical_spacing, delta=0.2)

        rotation_group = transformation._transf_board.transforms[1]
        self.assertEqual(rotation_group.p, 1.0)
        self.assertEqual(rotation_group.transforms_ps, [0.8, 0.15, 0.05])

        expected_rotations = [(-15, 15), (-45, 45), (-180, 180)]
        for affine, expected_rotation in zip(
            rotation_group.transforms, expected_rotations
        ):
            self.assertEqual(affine.scale["x"], (0.3, 0.5))
            self.assertEqual(affine.scale["y"], (0.3, 0.5))
            self.assertEqual(affine.translate_percent["x"], (-0.34, 0.34))
            self.assertEqual(affine.translate_percent["y"], (-0.34, 0.34))
            self.assertEqual(affine.rotate, expected_rotation)
            self.assertEqual(affine.shear["x"], (-10, 10))
            self.assertEqual(affine.shear["y"], (-10, 10))

        dropout_group = transformation._transf_board.transforms[3]
        self.assertEqual(dropout_group.p, 0.1)
        for dropout in dropout_group.transforms:
            self.assertEqual(dropout.max_holes, 3)
            self.assertEqual(dropout.max_height, 32)
            self.assertEqual(dropout.max_width, 32)
            self.assertEqual(dropout.min_height, 8)
            self.assertEqual(dropout.min_width, 8)

        exposure_group = transformation._transf_joint.transforms[1]
        contrast = transformation._transf_joint.transforms[2]
        blur_group = transformation._transf_joint.transforms[3]
        noise_group = transformation._transf_joint.transforms[4]
        self.assertEqual(exposure_group.p, 0.75)
        self.assertEqual(contrast.contrast_limit, (-0.65, -0.35))
        self.assertEqual(contrast.p, 0.9)
        self.assertEqual(blur_group.p, 0.8)
        self.assertEqual(noise_group.p, 0.4)


if __name__ == "__main__":
    unittest.main()
