import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from augmentation_validation import validate_board_sample
from configs import load_configuration
from transformations import Transformation


class TestAugmentationValidation(unittest.TestCase):
    def setUp(self):
        self.row_count = 6
        self.col_count = 9
        self.ids = np.arange(40, dtype=np.int64)
        self.points = np.asarray(
            [
                (80.0 + col * 12.0, 60.0 + row * 12.0)
                for row in range(5)
                for col in range(8)
            ],
            dtype=np.float32,
        )

    def validate(self, points=None, ids=None, **kwargs):
        return validate_board_sample(
            keypoints=self.points if points is None else points,
            ids=self.ids if ids is None else ids,
            image_shape=(240, 320),
            stride=4,
            row_count=self.row_count,
            col_count=self.col_count,
            **kwargs,
        )

    def test_valid_grid_passes(self):
        result = self.validate()
        self.assertTrue(result.is_valid)
        self.assertEqual(result.same_cell_collisions, 0)
        self.assertEqual(result.min_adjacent_distance, 12.0)
        self.assertEqual(result.max_spacing_ratio, 1.0)

    def test_same_cell_collision_fails(self):
        points = self.points.copy()
        points[1] = points[0] + (1.0, 1.0)
        result = self.validate(points=points)
        self.assertFalse(result.is_valid)
        self.assertIn("same_cell_collision", result.reasons)

    def test_non_finite_and_duplicate_ids_fail(self):
        points = self.points.copy()
        points[0, 0] = np.nan
        ids = self.ids.copy()
        ids[1] = ids[0]
        result = self.validate(points=points, ids=ids)
        self.assertFalse(result.is_valid)
        self.assertIn("non_finite_keypoint", result.reasons)
        self.assertIn("duplicate_corner_id", result.reasons)

    def test_too_few_visible_corners_fail(self):
        result = self.validate(points=self.points[:10], ids=self.ids[:10])
        self.assertFalse(result.is_valid)
        self.assertIn("too_few_visible_corners", result.reasons)

    def test_out_of_bounds_keypoint_fails(self):
        points = self.points.copy()
        points[0] = (-1.0, 60.0)
        result = self.validate(points=points)
        self.assertFalse(result.is_valid)
        self.assertIn("out_of_bounds_keypoint", result.reasons)

    def test_excessive_spacing_ratio_fails(self):
        points = self.points.copy().reshape(5, 8, 2)
        x_positions = np.asarray([80, 92, 104, 116, 128, 140, 152, 200])
        points[:, :, 0] = x_positions
        result = self.validate(points=points.reshape(-1, 2))
        self.assertFalse(result.is_valid)
        self.assertIn("perspective_too_strong", result.reasons)

    def test_folded_grid_fails(self):
        points = self.points.copy()
        points[9] = points[0] + (3.0, 3.0)
        result = self.validate(points=points)
        self.assertFalse(result.is_valid)
        self.assertIn("grid_folded", result.reasons)

    def test_invalid_geometry_uses_safe_fallback(self):
        config = load_configuration(str(SRC_DIR / "config.yaml"))
        transformation = Transformation(config)
        invalid_result = {
            "image": transformation.board_img,
            "mask": transformation.board_mask,
            "keypoints": np.full((40, 2), np.nan, dtype=np.float32),
            "ids": transformation.ids,
        }
        transformation.max_geometry_attempts = 2
        transformation._transf_board_geometry = Mock(return_value=invalid_result)

        result = transformation._transform_board()

        self.assertEqual(transformation._transf_board_geometry.call_count, 2)
        self.assertTrue(transformation._last_geometry_fallback)
        self.assertTrue(transformation._last_board_validation.is_valid)
        self.assertGreater(len(result["ids"]), 0)


if __name__ == "__main__":
    unittest.main()
