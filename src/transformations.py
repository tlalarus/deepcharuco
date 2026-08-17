import imgaug
import albumentations as A
import numpy as np
import cv2
import random
from augmentation_validation import validate_board_sample
from aruco_utils import board_image, get_board
from custom_aug.custom_aug import PasteBoard


# Monkey patching Albumentations 1.3.0 CoarseDropout bug :)
# https://github.com/albumentations-team/albumentations/pull/1330
def apply_to_keypoints(self, keypoints, holes, **params):
    result = set(keypoints)
    for hole in holes:
        for kp in keypoints:
            if self._keypoint_in_hole(kp, hole):
                result.discard(kp)
    return list(result)
A.CoarseDropout.apply_to_keypoints = apply_to_keypoints  # noqa: E305


def board_geometry_transformations(refinenet, input_size, use_perspective=True):
    transl = (0, 0) if refinenet else (-0.30, 0.30)
    scale = (0.3, 0.75) if refinenet else (0.3, 0.5)
    if refinenet:
        geometry_transform = A.Affine(
            scale=scale,
            rotate=(-360, 360),
            shear=(-35, 35),
            translate_percent=transl,
            keep_ratio=True,
            fit_output=False,
            always_apply=True,
        )
    else:
        affine_kwargs = {
            "scale": scale,
            "shear": (-10, 10),
            "translate_percent": transl,
            "keep_ratio": True,
            "fit_output": False,
        }
        geometry_transform = A.OneOf(
            [
                A.Affine(rotate=(-15, 15), p=0.80, **affine_kwargs),
                A.Affine(rotate=(-45, 45), p=0.15, **affine_kwargs),
                A.Affine(rotate=(-180, 180), p=0.05, **affine_kwargs),
            ],
            p=1.0,
        )

    transf = [
        A.PadIfNeeded(
            min_height=input_size[1],
            min_width=input_size[0],
            always_apply=True,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
        ),
        geometry_transform,
    ]
    if use_perspective and not refinenet:
        transf.append(
            A.Perspective(
                scale=(0.02, 0.08),
                keep_size=True,
                pad_mode=cv2.BORDER_CONSTANT,
                pad_val=0,
                mask_pad_val=0,
                fit_output=False,
                interpolation=cv2.INTER_LINEAR,
                p=0.6,
            )
        )
    transf.append(
        A.Resize(height=input_size[1], width=input_size[0], always_apply=True)
    )
    return A.Compose(transf, keypoint_params=A.KeypointParams(format='xy',
                                                              label_fields=['ids'],
                                                              remove_invisible=True))


def board_occlusion_transformations(refinenet):
    dropout_probability = 0 if refinenet else 0.1
    transforms = [
        A.CoarseDropout(
            max_holes=3,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            mask_fill_value=0,
        ),
        *[
            A.CoarseDropout(
                max_holes=3,
                max_height=32,
                max_width=32,
                min_holes=1,
                min_height=8,
                min_width=8,
                fill_value=fill_value,
                mask_fill_value=255,
            )
            for fill_value in (0, 128, 255)
        ],
    ]
    return A.Compose(
        [A.OneOf(transforms, p=dropout_probability)],
        keypoint_params=A.KeypointParams(
            format='xy', label_fields=['ids'], remove_invisible=True
        ),
    )


def safe_board_geometry_transformations(input_size):
    return A.Compose(
        [
            A.PadIfNeeded(
                min_height=input_size[1],
                min_width=input_size[0],
                always_apply=True,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
            A.Affine(
                scale=0.4,
                rotate=0,
                shear=0,
                translate_percent=0,
                keep_ratio=True,
                fit_output=False,
                always_apply=True,
            ),
            A.Resize(height=input_size[1], width=input_size[0], always_apply=True),
        ],
        keypoint_params=A.KeypointParams(
            format='xy', label_fields=['ids'], remove_invisible=True
        ),
    )


def _fit_board_resolution(input_size, row_count, col_count):
    """Fit a square-cell board inside the model input canvas."""
    input_width, input_height = input_size
    cell_size = min(input_width / col_count, input_height / row_count)
    return (
        int(round(cell_size * col_count)),
        int(round(cell_size * row_count)),
    )


class Transformation:
    """
    Class to apply augmentation on COCO dataset to train deepcharuco.
    Steps:
    0) Choose if is a negative sample, in that case just return an augmented coco
    1) Augment board image (+ mask + corners)
    2) ~ Histogram matching of board image given coco image
    3) Paste image on coco image
    4) Augment coco + board image
    5) Profit!
    """
    def __init__(self, configs, negative_p=0.05, refinenet=False, seed=None):
        self.seed = seed
        self.negative_p = negative_p
        if seed is not None:
            random.seed(seed)
            imgaug.random.seed(seed)

        self.refinenet = refinenet
        self.input_size = tuple(configs.input_size)
        self.stride = int(configs.mini_stride)
        self.row_count = int(configs.row_count)
        self.col_count = int(configs.col_count)
        self.max_geometry_attempts = 10
        self._last_geometry_attempts = 0
        self._last_geometry_fallback = False
        self._last_board_validation = None

        board = get_board(configs)
        board_resolution = _fit_board_resolution(
            configs.input_size,
            configs.row_count,
            configs.col_count,
        )
        board_img, corners = board_image(board, board_resolution,
                                         configs.row_count, configs.col_count)

        self.board_img = board_img
        self.corners = corners
        self.ids = np.arange(self.corners.shape[0])
        self.board_mask = np.full((board_img.shape[0], board_img.shape[1]),
                                  dtype=np.uint8, fill_value=255)

        # 1) Create transformation for board image
        self._transf_board_geometry = board_geometry_transformations(
            self.refinenet, configs.input_size, use_perspective=True
        )
        self._transf_board_occlusion = board_occlusion_transformations(self.refinenet)
        self._transf_board_fallback = safe_board_geometry_transformations(configs.input_size)

        # 1bis) COCO transformation
        self._transf_coco = A.Compose([
            A.Flip(p=0.5),
            A.Rotate(limit=(-180, 180), crop_border=True, p=0.5),
            A.PadIfNeeded(min_height=configs.input_size[1],
                          min_width=configs.input_size[0], always_apply=True,
                          border_mode=cv2.BORDER_CONSTANT, value=0,
                          mask_value=0),
            A.RandomCrop(height=configs.input_size[1],
                         width=configs.input_size[0], always_apply=True),
        ])

        # 2 + 3) Apply histogram matching then Paste transformation
        self._transf_joint = A.Compose([
            PasteBoard(always_apply=True),
            A.OneOf([
                A.RandomGamma(gamma_limit=(110, 180), p=1.0),
                A.MultiplicativeNoise(
                    multiplier=(0.45, 0.85),
                    per_channel=False,
                    elementwise=False,
                    p=1.0,
                ),
            ], p=0.75),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.10, 0.05),
                contrast_limit=(-0.65, -0.35),
                brightness_by_max=True,
                p=0.9,
            ),
            A.OneOf([
                A.GaussianBlur(
                    blur_limit=(3, 5),
                    sigma_limit=(0.5, 1.8),
                    p=1.0,
                ),
                A.MotionBlur(blur_limit=(3, 7), allow_shifted=True, p=1.0),
                A.Downscale(
                    scale_min=0.5,
                    scale_max=0.8,
                    interpolation=cv2.INTER_AREA,
                    p=1.0,
                ),
            ], p=0.8),
            A.OneOf([
                A.GaussNoise(
                    var_limit=(5.0, 30.0),
                    mean=0,
                    per_channel=False,
                    p=1.0,
                ),
                A.ISONoise(
                    color_shift=(0.0, 0.01),
                    intensity=(0.1, 0.4),
                    p=1.0,
                ),
            ], p=0.4),
        ], keypoint_params=A.KeypointParams(format='xy', label_fields=['ids'],
                                            remove_invisible=True)
        )

    def _transform_board(self):
        transform_args = {
            "image": self.board_img,
            "mask": self.board_mask,
            "keypoints": self.corners,
            "ids": self.ids,
        }
        self._last_geometry_fallback = False

        for attempt in range(1, self.max_geometry_attempts + 1):
            result = self._transf_board_geometry(**transform_args)
            validation = self._validate_board_geometry(result)
            if validation.is_valid:
                self._last_geometry_attempts = attempt
                self._last_board_validation = validation
                return self._transf_board_occlusion(**result)

        result = self._transf_board_fallback(**transform_args)
        validation = self._validate_board_geometry(result)
        if not validation.is_valid:
            raise RuntimeError(
                "Safe board geometry fallback produced an invalid sample: "
                + ", ".join(validation.reasons)
            )

        self._last_geometry_attempts = self.max_geometry_attempts
        self._last_geometry_fallback = True
        self._last_board_validation = validation
        return self._transf_board_occlusion(**result)

    def _validate_board_geometry(self, result):
        return validate_board_sample(
            keypoints=result["keypoints"],
            ids=result["ids"],
            image_shape=result["image"].shape[:2],
            stride=self.stride,
            row_count=self.row_count,
            col_count=self.col_count,
        )

    def __call__(self, coco_img):
        return self.transform(coco_img)

    def transform(self, coco_img):
        res = self._transform_board()  # Generate board image

        # Adapt coco image to input_size
        coco_img = self._transf_coco(image=coco_img)['image']

        # We also generate negative instances without board (if not refinenet)
        isnegative = False if self.refinenet else (random.random() < self.negative_p)

        # Apply joint pipeline
        res = self._transf_joint(**res, target=coco_img, isnegative=isnegative)
        return {'image': res['image'], 'keypoints': res['keypoints'],
                'ids': res['ids'], 'isnegative': isnegative}
