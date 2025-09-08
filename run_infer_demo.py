import cv2
import torch

import sys
sys.path.append("src")
from inference import infer_image, load_models

# Set path
deepc_path = 'src/reference/longrun-epoch=99-step=369700.ckpt'
refinenet_path = 'src/reference/second-refinenet-epoch-100-step=373k.ckpt'
image_path = 'src/data_demo/tv_side_01.bmp'

# Set model
# n_ids = 16 # num of corners
n_ids = 54 # num of corners
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
deepc, refinenet = load_models(deepc_path, refinenet_path, n_ids, device=device)

# Read image
img = cv2.imread(image_path)

# Inference
keypoints, out_img = infer_image(img, n_ids, deepc, refinenet, draw_pred=True, device=device)

# Point keypoints
print("\n📍 Inferred Keypoints:")
for (x, y, corner_id) in keypoints:
    print(f"  ID {int(corner_id):2d}: x={x:.2f}, y={y:.2f}")


# Display result image
cv2.imshow("prediction", out_img)
cv2.waitKey(0)
cv2.destroyAllWindows()