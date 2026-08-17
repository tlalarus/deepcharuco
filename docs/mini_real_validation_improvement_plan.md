# Mini DeepCharuco Real Validation Improvement Plan

## 목적

이 문서는 synthetic validation에서는 낮은 loss를 기록하지만 real validation의
`real_val_mean_error_px`가 30px 이상인 원인과 개선 순서를 정리합니다. 이후 Git
merge request의 문제 설명, 변경 내용, 검증 결과로 재사용하는 것을 목적으로 합니다.

Metric의 픽셀 단위는 모델 입력 해상도인 320x240 기준입니다. 원본 real validation
이미지는 640x480이므로 입력 기준 1px은 원본 기준 약 2px입니다. 원본 기준 1px 이하가
목표라면 현재 metric은 0.5px 이하여야 합니다.

## 분석 결과 요약

- best real validation mean error: 32.1737px
- real validation median error: 약 14.22px
- real validation PCK@5: 약 15.5%
- synthetic validation coordinate median error: 약 1.35px
- synthetic validation PCK@5: 약 92.3%
- real heatmap confidence median: 0.140
- synthetic heatmap confidence median: 0.749
- real prediction의 nearest GT가 올바른 ID인 비율: 약 45.5%

기존 `val_loss`는 decoded coordinate pixel error가 아닙니다. Heatmap focal loss와
masked offset L1 loss의 합이므로 `val_loss < 1.0`과 real coordinate error를 직접
비교할 수 없습니다.

## P0: Heatmap과 offset target 일관성

### 원인

기존 target 생성은 Gaussian을 feature-map의 소수 좌표 `(xh, yh)` 중심에 그렸지만
offset은 `(floor(xh), floor(yh))` 셀에 기록했습니다.

이 방식에는 두 가지 문제가 있습니다.

1. 소수 중심 Gaussian은 정수 grid에서 peak가 정확히 1.0이 되지 않는 경우가 대부분입니다.
   `HeatmapFocalLoss`는 target이 1.0인 위치만 positive로 처리하므로 positive supervision이
   누락됩니다. 측정 당시 real GT 600개에는 focal loss가 인식하는 positive가 없었습니다.
2. Heatmap argmax는 소수 중심에 가장 가까운 정수 셀에 형성되지만 offset은 floor 셀에
   있습니다. Decoder가 heatmap argmax 위치에서 offset을 조회하므로 서로 다른 셀을 사용할
   수 있습니다. 수정 전 GT encode/decode 왕복 오차는 평균 1.118px, 최대 5.010px였습니다.

### 해결 방법

Heatmap peak와 offset supervision 위치를 동일한 floor 정수 셀로 통일합니다.

```python
xh = x / stride
yh = y / stride
xi = floor(xh)
yi = floor(yh)

draw_gaussian(heatmap[k], xi, yi, sigma)
offset[0, yi, xi] = xh - xi
offset[1, yi, xi] = yh - yi
```

Gaussian 중심이 정수 셀이므로 peak target은 정확히 1.0이 됩니다. Decoder는 동일한
heatmap peak 셀에서 `[0, 1)` 범위의 offset을 읽어 원래 좌표를 복원합니다.

### 검증 기준

- 각 visible corner channel에 정확한 positive peak가 존재해야 합니다.
- Heatmap argmax가 `floor(keypoint / stride)`와 일치해야 합니다.
- GT target encode/decode 왕복 오차가 부동소수점 허용 오차 이하여야 합니다.
- 수정된 target 의미가 기존 학습과 다르므로 기존 optimizer state에서 resume하지 않고
  새로운 checkpoint 디렉터리에서 처음부터 학습합니다.

### 알려진 제한

현재 offset은 모든 corner가 공유하는 2채널입니다. 서로 다른 corner가 같은 spatial cell에
들어오면 offset target이 충돌합니다. Perspective augmentation 도입 후 collision 빈도를
측정하고, 실제 충돌이 발생하면 corner별 `2 * num_corners` offset head와 channel별 mask로
변경합니다. 사람이 구분하기 어려울 정도로 보드가 축소되거나 찌그러진 sample은 최소 인접
corner 거리 조건으로 제외합니다.

### 상태

- 구현 완료: heatmap Gaussian 중심과 offset supervision을 floor 셀로 통일
- 회귀 테스트 추가: positive peak, floor cell, encode/decode 복원 검증

## P0: Synthetic board 종횡비 일관성

### 원인

`gridboard.png`는 9x6 board 비율과 일치하는 480x320 해상도로 생성되어 정사각형 cell을
가집니다. 반면 학습 augmentation은 `min(input_size)`를 가로와 세로에 모두 사용해
9x6 board를 240x240으로 생성했습니다. 이때 합성 board의 가로 cell 간격은 약 26.7px,
세로 cell 간격은 40px이 되어 실제 정사각형 cell과 다른 geometry를 학습합니다.

합성 이미지와 GT corner가 같은 방식으로 변형되므로 synthetic loss는 낮아질 수 있지만,
실제 board로 일반화할 때 systematic domain gap이 발생합니다.

### 해결 방법

모델 입력 canvas 안에 정사각형 cell을 유지하는 최대 board 해상도를 계산합니다.

```python
cell_size = min(input_width / col_count, input_height / row_count)
board_width = round(cell_size * col_count)
board_height = round(cell_size * row_count)
```

현재 320x240 입력과 9x6 board에서는 약 320x213 board를 생성한 뒤 기존
`PadIfNeeded`로 320x240 canvas에 배치합니다. 이후 resize가 입력 크기와 동일하므로
board 비율은 추가로 왜곡되지 않습니다.

### 검증 기준

- 320x240 입력에서 학습 원본 board 해상도가 320x213이어야 합니다.
- 생성된 내부 corner의 평균 가로/세로 간격이 허용 오차 안에서 같아야 합니다.
- augmentation 이후 scale, perspective, visible corner 분포를 다시 측정해야 합니다.
- 기존 checkpoint는 잘못된 board geometry로 학습됐으므로 새 experiment에서 재학습합니다.

### 상태

- 구현 완료: 입력 canvas와 board row/column 수로 올바른 board 해상도 계산
- 회귀 테스트 추가: board 해상도와 정사각형 cell 간격 검증

## P1: Synthetic-to-real domain gap

### 원인

현재 synthetic augmentation은 affine, rotation, shear 중심입니다. Real validation에는
강한 projective perspective, 작은 marker, 저조도/IR 영상, 센서 노이즈, 낮은 contrast,
defocus와 실제 종이 반사가 포함됩니다.

15장 기준 분석에서는 보드 셀이 작을수록 오차가 커지는 상관이 -0.617, 원근 왜곡이
강할수록 오차가 커지는 상관이 +0.733이었습니다. 가장 강한 원근 왜곡을 가진
`rs_capture_05`의 평균 오차는 약 81.2px였습니다.

### 해결 방향

- 카메라 pose 또는 homography 기반 perspective augmentation 추가
- 실제 입력에서 관측된 10-16px cell 크기를 중심으로 scale distribution 조정
- gamma, exposure, low contrast, shot/read noise, defocus, motion blur 추가
- hard-edge binary paste 대신 실제 센서의 board/background 밝기 분포 반영
- 변환 후 인접 corner 최소 거리와 visible area를 검사해 비현실적 sample 제외
- camera, 거리, 각도, 조명별 real train 데이터를 확보하고 별도 validation/test와 분리

## P2: Corner localization과 ID 인식

### 원인

Real prediction을 가장 가까운 아무 GT와 비교한 median error는 약 7.56px이지만 ID를
고려하면 약 14.22px입니다. Nearest GT의 ID가 올바른 비율도 약 45.5%이므로 coarse
localization과 corner ID 인식이 모두 부족합니다.

현재 backbone은 pretrained weight 없이 ResNet18의 `layer1`까지만 사용합니다. 작은 실사
marker의 pattern과 board 문맥을 구분하기에는 feature depth와 receptive field가 제한적입니다.

### 해결 방향

- deeper backbone과 stride-4 FPN 조합 검토
- pretrained RGB weight를 grayscale conv에 맞춰 초기화
- corner localization과 ID classification을 분리하는 head 검토
- board geometry와 homography/RANSAC으로 ID inconsistency와 outlier 제거
- confidence threshold별 precision, recall, coverage를 함께 기록

## P3: Sub-pixel refinement와 평가 체계

### 원인

현재 mean error는 낮은 confidence의 큰 outlier에 민감합니다. 또한 synthetic `val_loss`와
real coordinate metric의 정의가 달라 학습 진행 상황을 직접 비교하기 어렵습니다. Real
validation 15장만으로는 camera와 scene별 일반화 성능을 충분히 판단할 수 없습니다.

### 해결 방향

- synthetic validation에도 decoded mean, median, P95, PCK@2, PCK@5 추가
- real validation에 confidence별 coverage와 per-camera/per-condition metric 추가
- scene와 camera 단위로 real train/validation/test 분리
- coarse detector가 충분한 PCK@5를 확보한 후 `cornerSubPix`, refinement head 또는
  RefineNet으로 0.5px 이하 정밀화

## Merge Request 검증 체크리스트

1. 변경된 target으로 unit test와 inference smoke test를 Docker에서 실행합니다.
2. 기존 checkpoint 디렉터리를 재사용하지 않고 새로운 experiment 이름으로 학습합니다.
3. Synthetic decoded coordinate metric과 real metric을 동일한 픽셀 단위로 보고합니다.
4. Mean뿐 아니라 median, P95, PCK@2, PCK@5를 이전 baseline과 비교합니다.
5. Perspective augmentation 이후 same-cell offset collision 빈도를 기록합니다.
6. Real validation 15장은 학습 데이터에 포함하지 않습니다.
7. 학습 sample의 board cell 가로/세로 비율과 real 분포를 비교합니다.
