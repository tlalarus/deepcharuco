import argparse
import random
import sys

import numpy as np

import configs
from configs import load_configuration
from transformations import Transformation


def percentile_summary(values):
    return np.percentile(values, [1, 50, 95, 99]).round(4).tolist()


def main():
    parser = argparse.ArgumentParser(description="Analyze mini board augmentation geometry.")
    parser.add_argument("--config", default=configs.CONFIG_PATH)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    config = load_configuration(args.config)
    transformation = Transformation(config, negative_p=0.0, refinenet=False)

    visible_counts = []
    attempts = []
    fallback_count = 0
    hull_area_ratios = []
    min_adjacent_distances = []
    spacing_ratios = []
    collisions = 0

    for _ in range(args.samples):
        result = transformation._transform_board()
        validation = transformation._last_board_validation
        visible_counts.append(len(result["ids"]))
        attempts.append(transformation._last_geometry_attempts)
        fallback_count += int(transformation._last_geometry_fallback)
        hull_area_ratios.append(validation.hull_area_ratio)
        min_adjacent_distances.append(validation.min_adjacent_distance)
        spacing_ratios.append(validation.max_spacing_ratio)
        collisions += validation.same_cell_collisions

    visible_counts = np.asarray(visible_counts)
    attempts = np.asarray(attempts)
    print(f"samples: {args.samples}")
    print(f"valid_first_try: {np.mean(attempts == 1):.4f}")
    print(f"resampled: {np.mean(attempts > 1):.4f}")
    print(f"fallback: {fallback_count / args.samples:.4f}")
    print(f"full_board: {np.mean(visible_counts == config.n_ids):.4f}")
    print(f"partial_board: {np.mean((visible_counts > 0) & (visible_counts < config.n_ids)):.4f}")
    print(f"empty_board: {np.mean(visible_counts == 0):.4f}")
    print(f"same_cell_collisions: {collisions}")
    print(f"hull_area_ratio p01/p50/p95/p99: {percentile_summary(hull_area_ratios)}")
    print(
        "min_adjacent_distance p01/p50/p95/p99: "
        f"{percentile_summary(min_adjacent_distances)}"
    )
    print(f"spacing_ratio p01/p50/p95/p99: {percentile_summary(spacing_ratios)}")


if __name__ == "__main__":
    sys.exit(main())
