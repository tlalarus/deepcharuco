# Mini DeepCharuco Implementation Specification
## 1. Objective
The goal of this project is to implement **Mini DeepCharuco**, a simplified deep learning model that detects **only chessboard corner points from a Charuco board image**.
Unlike the original DeepCharuco model, which predicts both:
- corner locations
- corner IDs

Mini DeepCharuco **predicts only corner locations**.

### Input
Charuco board image
### Output
K chessboard corner coordinates
### Pipeline:
image -> neural network -> corner coordinates

---
# 2. Original DeepCharuco Architecture
* Original DeepCharuco outputs:
  - loc:(65, H/8, W/8)
    - `loc` -> corner localization
  - ids:(17, H/8, W/8)
    - `ids` -> corner identity prediction
* For Mini DeepCharuco se **remove the ID head** entirely.

---

# 4. Mini DeepCharuco Architecture
* Model structure:
```
input image
│
▼
backbone network
(ResNet / HRNet / UNet)
│
▼
feature map
│
├─ heatmap head (K channels)
│
└─ offset head (2 channels)
```

* Outputs:
  * heatmap: K x H' x W'
  * offset: 2 x H' x W'
---
# 4. K-Channel Keypoint Representation
Each channel corresponds to a **specific corner index**.
`heatmap[k]` represents the probability map for the **k-th corner location**.
The peak value in each channel indicates the predicted position.
This is the standard **heatmap regression approach used in keypoint detection models**.
# 5. CenterNet-style Keypoint Detection
Mini DeepCharuco follow a **CenterNet-styele architecture**.
Prediction components:
```
heatmap -> keypoint location probability
offset -> subpixel correction
```

# 6. Corner Definition
K equals the **number of internal chessboard corners**.
Example:
```
board squares = 8 x 6
internal corners = 7 x 5
K = 35
```
Each corner index corresponds to a known **3D world coordinate**.
---
# 7. Output Resolution
- input resolution: H x W
- stride: s
- output resolution: H/x x W/s
- Recommended:
  - stride = 4
  - Example:
    - input: 1024 x 1024
    - output: 256 x 256

# 8. Ground Truth Generation
### 8.1 Heatmap
A Gaussian is placed at each corner location.
* Feature map coordinates:
  * xh = x / stride
  * yh = y / stride
  ```
  Gaussian:
    exp(-((i-xh)^2 + (j-yh)^2)/(2σ²))

  Recommended:
    σ = 1.5 ~ 2
  ```
### 8.2 Offset
Offsets correct quantization error caused by stride.
```
xh = x / stride
yh = y / stride

xi = floor(xh)
yi = floor(yh)

offset_x = xh - xi
offset_y = yh - yi

stored as:
    offset_gt[:,yi, xi] = [offset_x, offset_y]
```
---
# 9. Loss Function
Total loss:
```
L = L_heatmap + λ L_offset
```
## Heatmap Loss
* Recommended: Local Loss
* Reason:
  * extemely sparse positive pixels
  * large number of negative pixels
## Offset Loss
```
L1 loss
```
offset loss is calcuated **only at corner locations**.
---
# 10. Inference

Corner extraction pipeline.

### Step 1: Heatmap Peak


(yh,xh) = argmax(heatmap[k])


### Step 2: Offset


dx = offset[0,yh,xh]
dy = offset[1,yh,xh]


### Step 3: Coordinate Reconstruction


x = (xh + dx) * stride
y = (yh + dy) * stride


### Step 4: Corner List


corners[k] = (x,y)


---

# 11. Backbone Network

Initial implementation:


ResNet18


Future experiments:


HRNet
UNet


---

# 12. Input Resolution

Camera resolution:


2656 × 2304


Training resolution:


1024
or
1280


Recommended:


1280


---

# 13. Dataset Generation

A synthetic dataset generator is required.

Randomized factors:


perspective transform
blur
motion blur
noise
lighting
exposure
partial occlusion
background texture


Synthetic data quality will have **major impact on model performance**.

---

# 14. Implementation Task List

## Model

File:


model/mini_deepcharuco.py


Components:


backbone
heatmap_head
offset_head


---

## Dataset


dataset/charuco_dataset.py


Responsibilities:


image loading
corner ground truth loading
heatmap generation
offset generation


---

## Loss


loss/heatmap_loss.py
loss/offset_loss.py


---

## Training


train.py


Responsibilities:


dataloader
model initialization
optimizer
loss computation
checkpoint saving


---

## Inference


infer.py


Output:


corner coordinates list


---

# 15. Project Structure


mini_deepcharuco/

models/
mini_deepcharuco.py

dataset/
charuco_dataset.py

loss/
heatmap_loss.py
offset_loss.py

train.py
infer.py
config.yaml


---

# 16. Final Goal

The final model should perform:

```

charuco image
│
▼
mini deepcharuco
│
▼
K corner coordinates
```


These coordinates will be used for:


camera calibration
PnP
3D reconstruction