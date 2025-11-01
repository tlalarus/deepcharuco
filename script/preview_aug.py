import os
import cv2
from configs import load_configuration
from transformations import Transformation

cfs = load_configuration("src/demo_config.yaml")
transform = Transformation(cfg, negative_p=0.05, refinenet=False, seed=42)

raw = cv2.imread("data/coco_sample.jpg") # COCO original image

# 보드 전처리 
board_step = transform._transform_board()
board_img = board_step["image"]
board_mask = board_step["mask"]

# 배경 전처리 
coco_step = transform._transf_coco(image=raw)
coco_img = coco_step["image"]

# 합성 및 최종 전처리
joint_input = {**board_step, "target": coco_img, "isnegative": False}
final_step = transform._transf_joint(**joint_input)

os.makedirs("debug_aug", exist_ok=True)
cv2.imwrite("debug_aug/01_board.png", board_img)
cv2.imwrite("debug_aug/02_mask.png", board_mask)
cv2.imwrite("debug_aug/03_coco.png", coco_img)
cv2.imwrite("debug_aug/04_final.png", final_step["image"])



