from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BoardSampleValidation:
    is_valid: bool
    reasons: tuple[str, ...]
    visible_count: int
    hull_area_ratio: float
    min_adjacent_distance: float
    max_spacing_ratio: float
    same_cell_collisions: int


def _quad_is_valid(points: list[np.ndarray], eps: float = 1e-6) -> bool:
    crosses = []
    for index in range(4):
        current = points[index]
        next_point = points[(index + 1) % 4]
        following = points[(index + 2) % 4]
        first_edge = next_point - current
        second_edge = following - next_point
        crosses.append(
            first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]
        )

    crosses = np.asarray(crosses, dtype=np.float32)
    return bool(np.all(crosses > eps) or np.all(crosses < -eps))


def validate_board_sample(
    keypoints,
    ids,
    image_shape,
    stride: int,
    row_count: int,
    col_count: int,
    min_visible_corners: int = 20,
    min_hull_area_ratio: float = 0.02,
    max_hull_area_ratio: float = 0.40,
    min_adjacent_distance_px: float = 6.0,
    max_spacing_ratio: float = 2.5,
) -> BoardSampleValidation:
    points = np.asarray(keypoints, dtype=np.float32)
    corner_ids = np.asarray(ids, dtype=np.int64)
    reasons = []
    visible_count = len(corner_ids)

    if points.size == 0:
        points = np.empty((0, 2), dtype=np.float32)
    else:
        points = points.reshape((-1, 2))

    if len(points) != visible_count:
        reasons.append("keypoint_id_length_mismatch")

    if not np.isfinite(points).all():
        reasons.append("non_finite_keypoint")

    grid_rows = row_count - 1
    grid_cols = col_count - 1
    num_corners = grid_rows * grid_cols
    if np.any(corner_ids < 0) or np.any(corner_ids >= num_corners):
        reasons.append("invalid_corner_id")
    if len(np.unique(corner_ids)) != visible_count:
        reasons.append("duplicate_corner_id")
    if visible_count < min_visible_corners:
        reasons.append("too_few_visible_corners")

    height, width = image_shape
    if points.size and np.isfinite(points).all():
        in_bounds = (
            (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height)
        )
        if not np.all(in_bounds):
            reasons.append("out_of_bounds_keypoint")

    hull_area_ratio = 0.0
    if len(points) >= 3 and np.isfinite(points).all():
        hull = cv2.convexHull(points)
        hull_area_ratio = float(cv2.contourArea(hull) / (height * width))
    if hull_area_ratio < min_hull_area_ratio:
        reasons.append("board_area_too_small")
    if hull_area_ratio > max_hull_area_ratio:
        reasons.append("board_area_too_large")

    same_cell_collisions = 0
    if points.size and np.isfinite(points).all():
        cells = np.floor(points / stride).astype(np.int64)
        same_cell_collisions = visible_count - len(np.unique(cells, axis=0))
        if same_cell_collisions:
            reasons.append("same_cell_collision")

    point_by_id = {
        int(corner_id): point
        for corner_id, point in zip(corner_ids, points)
        if 0 <= corner_id < num_corners and np.isfinite(point).all()
    }
    adjacent_distances = []
    for corner_id, point in point_by_id.items():
        row, col = divmod(corner_id, grid_cols)
        right_id = corner_id + 1
        down_id = corner_id + grid_cols
        if col + 1 < grid_cols and right_id in point_by_id:
            adjacent_distances.append(np.linalg.norm(point_by_id[right_id] - point))
        if row + 1 < grid_rows and down_id in point_by_id:
            adjacent_distances.append(np.linalg.norm(point_by_id[down_id] - point))

    min_adjacent_distance = 0.0
    spacing_ratio = float("inf")
    if adjacent_distances:
        adjacent_distances = np.asarray(adjacent_distances, dtype=np.float32)
        min_adjacent_distance = float(adjacent_distances.min())
        spacing_ratio = float(adjacent_distances.max() / max(min_adjacent_distance, 1e-6))
    if min_adjacent_distance < min_adjacent_distance_px:
        reasons.append("adjacent_corners_too_close")
    if spacing_ratio > max_spacing_ratio:
        reasons.append("perspective_too_strong")

    grid_folded = False
    for row in range(grid_rows - 1):
        for col in range(grid_cols - 1):
            top_left = row * grid_cols + col
            quad_ids = [
                top_left,
                top_left + 1,
                top_left + 1 + grid_cols,
                top_left + grid_cols,
            ]
            if all(corner_id in point_by_id for corner_id in quad_ids):
                if not _quad_is_valid([point_by_id[corner_id] for corner_id in quad_ids]):
                    grid_folded = True
                    break
        if grid_folded:
            break
    if grid_folded:
        reasons.append("grid_folded")

    unique_reasons = tuple(dict.fromkeys(reasons))
    return BoardSampleValidation(
        is_valid=not unique_reasons,
        reasons=unique_reasons,
        visible_count=visible_count,
        hull_area_ratio=hull_area_ratio,
        min_adjacent_distance=min_adjacent_distance,
        max_spacing_ratio=spacing_ratio,
        same_cell_collisions=same_cell_collisions,
    )
