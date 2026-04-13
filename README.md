# Rossmann Store Sales Forecasting

Project dự báo doanh thu chuỗi cửa hàng Rossmann dựa trên bộ dữ liệu Kaggle `Rossmann Store Sales`. Repo triển khai pipeline đầy đủ gồm tiền xử lý dữ liệu, feature engineering, huấn luyện mô hình XGBoost, LightGBM và cross-ensemble giữa hai framework.

## Người đóng góp

- Huỳnh Minh Triết
- Nguyễn Tấn Phúc

## Cấu trúc thư mục

```text
rossman-sale-main/
├── data/
│   └── processed/
│       ├── train_featured.csv
│       ├── holdout_featured.csv
│       └── test_featured.csv
├── rossmann-store-sales/
│   ├── train.csv
│   ├── test.csv
│   ├── store.csv
│   └── sample_submission.csv
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── train_lgbm.py
│   ├── train_ensemble.py
│   ├── model.py
│   ├── model_lgbm.py
│   ├── features_recent.py
│   ├── features_temporal.py
│   └── features_trend.py
├── eda-dataset.ipynb
└── requirements.txt
```

## Môi trường đề xuất

- Python: repo hiện đang có `.venv` với `Python 3.13.3`
- Cài package bằng `requirements.txt`
- Thư viện chính:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `xgboost`
  - `lightgbm`
  - `joblib`

Khuyến nghị tạo virtual environment riêng rồi cài dependency:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Trong repo hiện tại, lệnh `python` không nằm sẵn trong `PATH`, vì vậy nên dùng trực tiếp:

```bash
.venv/bin/python -m ...
```

## Dữ liệu đầu vào

Thư mục dữ liệu thô mặc định:

```text
rossmann-store-sales/
├── train.csv
├── test.csv
└── store.csv
```

Các file feature đã được sinh sẵn hiện nằm ở:

```text
data/processed/
├── train_featured.csv
├── holdout_featured.csv
└── test_featured.csv
```

Nếu muốn tái tạo toàn bộ pipeline từ dữ liệu gốc, hãy chạy lại bước preprocessing.

## Quy trình chạy

### 1. Tiền xử lý và tạo feature

Lệnh:

```bash
.venv/bin/python -m src.preprocess
```

Hoặc chỉ định thư mục input/output:

```bash
.venv/bin/python -m src.preprocess \
  --input_dir rossmann-store-sales \
  --output_dir data/processed
```

Script này sẽ:

- đọc `train.csv`, `test.csv`, `store.csv`
- merge dữ liệu cửa hàng
- tách `train` và `holdout` theo thời gian
- tạo feature theo 3 nhóm:
  - recent features
  - temporal/event features
  - trend features
- sinh ra:
  - `data/processed/train_featured.csv`
  - `data/processed/holdout_featured.csv`
  - `data/processed/test_featured.csv`

### 2. Train pipeline XGBoost

Lệnh:

```bash
.venv/bin/python -m src.train
```

Hoặc:

```bash
.venv/bin/python -m src.train \
  --input_dir data/processed \
  --output_dir .
```

Kết quả:

- `submission.csv`
- `data/models/xgb_models.pkl`
- `data/models/xgb_holdout_rmspe.txt`

Pipeline này gồm:

- random feature selection với 500 probe models
- tìm cặp ensemble tốt nhất bằng harmonic mean
- combined model
- season model
- month-ahead model
- final ensemble để predict test set

### 3. Train pipeline LightGBM

Lệnh:

```bash
.venv/bin/python -m src.train_lgbm --mode lgbm
```

Kết quả:

- `submission_lgbm.csv`
- `data/models/lgbm_models.pkl`

### 4. Cross-ensemble XGBoost + LightGBM

Cách 1: chạy sau khi đã có model XGBoost từ `src.train`

```bash
.venv/bin/python -m src.train_lgbm --mode ensemble --xgb_weight 0.5
```

Kết quả:

- `submission_cross_ensemble.csv`

Cách 2: chạy cả 2 pipeline trong một lệnh

```bash
.venv/bin/python -m src.train_ensemble --xgb_weight 0.5
```

Kết quả:

- `submission_xgb.csv`
- `submission_lgbm.csv`
- `submission_cross_ensemble.csv`

## Ý nghĩa các file mã nguồn chính

- `src/preprocess.py`: đọc dữ liệu thô, tách holdout, tạo feature và lưu CSV
- `src/features_recent.py`: rolling statistics theo nhiều khung thời gian
- `src/features_temporal.py`: feature lịch, promo, holiday, competition, refurbishment
- `src/features_trend.py`: trend features bằng Ridge Regression theo store/tháng
- `src/model.py`: toàn bộ logic train và ensemble cho XGBoost
- `src/model_lgbm.py`: logic tương đương cho LightGBM
- `src/train.py`: entrypoint train XGBoost và xuất `submission.csv`
- `src/train_lgbm.py`: entrypoint train LightGBM hoặc cross-ensemble
- `src/train_ensemble.py`: chạy XGBoost + LightGBM trong cùng một lệnh
- `eda-dataset.ipynb`: notebook phục vụ EDA

## Ghi chú quan trọng

- Cần chạy `src.preprocess` trước khi chạy các script train nếu chưa có `*_featured.csv`.
- Mã nguồn hiện cấu hình:
  - XGBoost với `device='cuda'`
  - LightGBM với `device='gpu'`
- Vì vậy môi trường chạy nên có GPU tương thích. Nếu chạy CPU, cần sửa tham số trong `src/model.py` và `src/model_lgbm.py`.
- Metric đánh giá nội bộ là `RMSPE`, chỉ tính trên các dòng có `Sales > 0`.
- Ở bước dự đoán cuối, các cửa hàng đóng cửa (`Open == 0`) sẽ bị ép `Sales = 0`.
- Thư mục `data/processed` hiện khá lớn, khoảng hơn 2 GB.
- Repo hiện không phải working tree Git đầy đủ trong môi trường này, nên README được viết dựa trên cấu trúc file thực tế và nội dung mã nguồn hiện có.

## Tham số CLI

Xem nhanh trợ giúp của từng script:

```bash
.venv/bin/python -m src.preprocess --help
.venv/bin/python -m src.train --help
.venv/bin/python -m src.train_lgbm --help
.venv/bin/python -m src.train_ensemble --help
```

## Kết quả đầu ra thường gặp

- Feature files:
  - `data/processed/train_featured.csv`
  - `data/processed/holdout_featured.csv`
  - `data/processed/test_featured.csv`
- Submission files:
  - `submission.csv`
  - `submission_lgbm.csv`
  - `submission_xgb.csv`
  - `submission_cross_ensemble.csv`
- Saved models:
  - `data/models/xgb_models.pkl`
  - `data/models/lgbm_models.pkl`

## Gợi ý chạy nhanh

Nếu chỉ cần chạy lại toàn bộ pipeline cơ bản:



Nếu muốn xuất luôn submission ensemble giữa XGBoost và LightGBM:


