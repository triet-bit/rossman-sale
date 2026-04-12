"""
BƯỚC 0–4 & 6: Preprocessing – Feature Engineering → Lưu CSV
=============================================================
Script này đọc dữ liệu thô (train.csv, test.csv, store.csv),
thực hiện toàn bộ Feature Engineering *không bị data leakage*,
rồi lưu 3 file CSV để script train.py tiêu thụ:

  data/processed/
  ├── train_featured.csv    ← Dùng để train model
  ├── holdout_featured.csv  ← Dùng để đánh giá RMSPE nội bộ
  └── test_featured.csv     ← Dùng để predict & nộp Kaggle

Cách chạy:
    cd /home/minhtriet/Documents/rossman
    python -m src.preprocess
    # hoặc
    python src/preprocess.py

Tham số tuỳ chỉnh:
    HOLDOUT_WEEKS   : số tuần cuối dùng làm holdout (mặc định 6)
    HISTORY_DAYS    : số ngày lịch sử cần gắn vào holdout/test (mặc định 500)
    RAW_DIR         : thư mục chứa train.csv, test.csv, store.csv
    OUT_DIR         : thư mục xuất file featured
"""
import sys, os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.features_recent   import add_recent_features
from src.features_temporal import add_temporal_features
from src.features_trend    import add_trend_features_optimized

# ──── Cấu hình ───────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(ROOT_DIR,"rossmann-store-sales")                             
OUT_DIR       = os.path.join(ROOT_DIR, 'data', 'processed')
HOLDOUT_WEEKS = 6     
HISTORY_DAYS  = 500   


# ============================================================
# HELPER
# ============================================================

def _run_fe(df):
    """Chạy lần lượt 3 bước Feature Engineering."""
    df = add_recent_features(df)           # Bước 2: rolling stats
    df = add_temporal_features(df)         # Bước 3: temporal/event
    df = add_trend_features_optimized(df)  # Bước 4: Ridge trend
    return df


def _reset_if_datetime(df):
    """Đảm bảo 'Date' là cột thường (không phải index)."""
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def _concat_with_context(context_df, target_df):
    """
    Gắn `context_df` (lịch sử) vào đầu `target_df`, sort theo [Store, Date],
    chạy FE, rồi cắt trả lại phần `target_df`.

    Cơ chế này đảm bảo rolling features của target_df có đủ lịch sử
    mà không bị leakage (context không xuất hiện trong output).
    """
    combined = pd.concat([context_df, target_df], ignore_index=True)
    combined['Date'] = pd.to_datetime(combined['Date'])
    combined = combined.sort_values(['Store', 'Date']).reset_index(drop=True)

    featured = _run_fe(combined)
    featured = _reset_if_datetime(featured)

    # Giữ lại đúng các dòng thuộc target (dựa vào index gốc)
    n_context = len(context_df)
    # Cách an toàn: lọc theo Date >= min_date của target
    min_target_date = pd.to_datetime(target_df['Date']).min()
    result = featured[featured['Date'] >= min_target_date].copy()
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── BƯỚC 0: Đọc dữ liệu thô ─────────────────────────────
    print("=" * 60)
    print("BƯỚC 0: Đọc dữ liệu thô")
    print("=" * 60)

    train_raw = pd.read_csv(os.path.join(RAW_DIR, 'train.csv'), low_memory=False)
    test_raw  = pd.read_csv(os.path.join(RAW_DIR, 'test.csv'))
    store_raw = pd.read_csv(os.path.join(RAW_DIR, 'store.csv'))

    train_raw['Date'] = pd.to_datetime(train_raw['Date'])
    train_store       = train_raw.merge(store_raw, on='Store', how='left')

    print(f"  Train  : {len(train_raw):,} rows")
    print(f"  Test   : {len(test_raw):,} rows")
    print(f"  Store  : {len(store_raw):,} stores")
    print(f"  Range  : {train_raw['Date'].min().date()} → {train_raw['Date'].max().date()}")

    # ── BƯỚC 1: Tách Train / Holdout TRƯỚC feature engineering ─
    print("\n" + "=" * 60)
    print("BƯỚC 1: Tách Train / Holdout (trước Feature Engineering)")
    print("=" * 60)

    max_date      = train_store['Date'].max()
    holdout_start = max_date - pd.Timedelta(weeks=HOLDOUT_WEEKS)

    train_split   = train_store[train_store['Date'] <  holdout_start].copy()
    holdout_split = train_store[train_store['Date'] >= holdout_start].copy()

    print(f"  Train raw  : {len(train_split):,} rows  (đến {holdout_start.date()})")
    print(f"  Holdout raw: {len(holdout_split):,} rows (từ {holdout_start.date()})")

    # ── BƯỚC 2–4: Feature Engineering cho TRAIN ──────────────
    print("\n" + "=" * 60)
    print("BƯỚC 2–4: Feature Engineering cho Train set")
    print("=" * 60)

    train_featured = _run_fe(train_split)
    train_featured = _reset_if_datetime(train_featured)

    out_path = os.path.join(OUT_DIR, 'train_featured.csv')
    train_featured.to_csv(out_path, index=False)
    print(f"  ✅ Đã lưu: {out_path}  ({train_featured.shape})")

    # ── BƯỚC 2–4: Feature Engineering cho HOLDOUT ────────────
    print("\n" + "=" * 60)
    print("BƯỚC 2–4: Feature Engineering cho Holdout set")
    print("=" * 60)

    context_end   = train_split['Date'].max()
    context_start = context_end - pd.Timedelta(days=HISTORY_DAYS)
    context_train = train_split[train_split['Date'] >= context_start].copy()

    holdout_featured = _concat_with_context(context_train, holdout_split)

    out_path = os.path.join(OUT_DIR, 'holdout_featured.csv')
    holdout_featured.to_csv(out_path, index=False)
    print(f"  ✅ Đã lưu: {out_path}  ({holdout_featured.shape})")

    # ── BƯỚC 6: Feature Engineering cho TEST ─────────────────
    print("\n" + "=" * 60)
    print("BƯỚC 6: Feature Engineering cho Test set")
    print("=" * 60)

    test_store        = test_raw.merge(store_raw, on='Store', how='left')
    test_store['Open'] = test_store['Open'].fillna(1)  # giả định mở nếu NaN
    test_store['Date'] = pd.to_datetime(test_store['Date'])

    # Gắn 500 ngày lịch sử từ toàn bộ train (không chỉ train_split)
    full_context_start = max_date - pd.Timedelta(days=HISTORY_DAYS)
    full_context       = train_store[train_store['Date'] >= full_context_start].copy()

    print(f"  Lịch sử gắn vào : {len(full_context):,} rows "
          f"(từ {full_context_start.date()} đến {max_date.date()})")
    print(f"  Test rows        : {len(test_store):,}")

    test_featured = _concat_with_context(full_context, test_store)

    # Chỉ giữ dòng có 'Id' (tức là phần test thực sự)
    if 'Id' in test_featured.columns:
        test_featured = test_featured[test_featured['Id'].notna()].copy()
        test_featured['Id'] = test_featured['Id'].astype(int)

    out_path = os.path.join(OUT_DIR, 'test_featured.csv')
    test_featured.to_csv(out_path, index=False)
    print(f"  ✅ Đã lưu: {out_path}  ({test_featured.shape})")

    # ── Tóm tắt ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PREPROCESSING HOÀN TẤT")
    print("=" * 60)
    print(f"  Output dir  : {OUT_DIR}")
    print(f"  train_featured  : {train_featured.shape}")
    print(f"  holdout_featured: {holdout_featured.shape}")
    print(f"  test_featured   : {test_featured.shape}")
    print("\nBước tiếp theo:\n  python -m src.train")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rossmann Preprocessing - Feature Engineering")
    parser.add_argument('--input_dir', type=str, default=RAW_DIR, help='Thư mục chứa dữ liệu thô (train.csv, test.csv, store.csv)')
    parser.add_argument('--output_dir', type=str, default=OUT_DIR, help='Thư mục lưu dữ liệu đã xử lý')
    args = parser.parse_args()

    RAW_DIR = args.input_dir
    OUT_DIR = args.output_dir

    main()
