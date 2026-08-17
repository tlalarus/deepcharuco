# Mini DeepCharuco Real Validation

## 목적

real validation 데이터는 synthetic train/validation과 절대 섞이지 않고, 매 epoch 종료 후 실제 이미지로만 평가합니다. 이 기능은 `mini_deepcharuco` 모델에서 사용됩니다.

## 데이터 준비

real validation 데이터는 아래 구조로 준비합니다:

```text
data/mini_deepcharuco/val_real/
  images/
  corners/
  manifest.csv
```

각 이미지와 CSV는 동일한 basename을 가져야 합니다.

- 이미지: `*.png`, `*.jpg`, `*.jpeg`, `*.bmp`, `*.tif`, `*.tiff`
- CSV: corner 좌표를 담고 있어야 하며, `x`, `y` 컬럼이 필수입니다.
- 선택 컬럼: `corner_id` 또는 `id`, `visible` 또는 `valid`

CSV 예시:

```csv
,Area,Mean,Min,Max,X,Y,XM,YM,BX,BY,Width,Height
1,0,41,41,41,183.500,205.875,183.500,205.875,184,206,1,1
2,0,54,54,54,209.969,204.938,209.969,204.938,210,205,1,1
```

`X`, `Y` 컬럼이 아닌 경우 header 이름이 `x`, `y`로 반드시 존재해야 합니다.

## 준비 스크립트

다음 커맨드를 사용해 real validation 데이터를 준비합니다:

```bash
cd /home/minkyung/Projects/tlalarus/deepcharuco/src
python prepare_real_val.py \
  --images-dir /data/dataset/real_val/image \
  --csv-dir /data/dataset/real_val/csv \
  --output-root ../data/mini_deepcharuco/val_real \
  --expected-count 15
```

필요에 따라 `--mode copy` 또는 `--mode symlink`를 선택할 수 있습니다.

## 설정

`src/config.yaml` 또는 사용 중인 YAML 설정에 다음 필드를 추가합니다:

```yaml
use_real_val: true
real_val_dir: 'data/mini_deepcharuco/val_real'
real_val_batch_size: 1
real_val_num_workers: 0
real_val_every: 1
val_every: 1
```

`real_val_batch_size`는 `1`이어야 하며, `real_val_num_workers`를 `0`으로 유지하면 가장 안전합니다.

## 학습 실행

real validation 활성화 후 학습을 실행합니다:

```bash
cd /home/minkyung/Projects/tlalarus/deepcharuco/src
python train_mini.py
```

또는 일반 훈련 스크립트에서:

```bash
python train.py
```

## 기록되는 Metric

real validation 결과는 다음 이름으로 로그에 기록됩니다:

- `real_val_mean_error_px`
- `real_val_median_error_px`
- `real_val_pck_2px`
- `real_val_pck_5px`
- `real_val_num_images`
- `real_val_num_points`
- `real_val/mean_error_px`
- `real_val/median_error_px`
- `real_val/pck_2px`
- `real_val/pck_5px`
- `real_val/num_images`
- `real_val/num_points`

## Best checkpoint 기준

`real_val_mean_error_px`가 최소인 모델을 `tb_logs/ckpts_mini_deepcharuco/`에 저장합니다. 체크포인트에는 `real_val_metrics` 메타데이터가 포함됩니다.

## 확인 체크리스트

실제 이미지 15장과 CSV 15개를 추가한 뒤 다음을 확인하세요:

1. `data/mini_deepcharuco/val_real/images`에 15개 이미지가 있는가
2. `data/mini_deepcharuco/val_real/corners`에 15개 CSV가 있는가
3. `data/mini_deepcharuco/val_real/manifest.csv`가 생성되었는가
4. `src/config.yaml`에 `use_real_val: true` 및 `real_val_dir`가 올바로 설정되었는가
5. `trainer.fit` 실행 시 `real_val_mean_error_px` 로그가 기록되는가
6. `tb_logs/ckpts_mini_deepcharuco/`에 real validation 기준 체크포인트가 저장되는가
