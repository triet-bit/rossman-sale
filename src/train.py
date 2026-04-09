"""
BƯỚC 5 & 7: Training XGBoost & Tạo Submission
================================================
Script này đọc các CSV đã được tạo bởi preprocess.py,
huấn luyện mô hình theo pipeline của paper Rossmann,
rồi xuất submission.csv.

  BƯỚC 5A : Đọc train_featured.csv + holdout_featured.csv
  BƯỚC 5B : Random Feature Selection  (500 probe models, nhanh)
  BƯỚC 5C : Tìm cặp ensemble tốt nhất (harmonic mean)
  BƯỚC 5D : Combined Model             (full XGB_PARAMS + 5000 rounds)
  BƯỚC 5E : Season Model + Month-ahead Model
  BƯỚC 5F : Final Ensemble             (harmonic mean × 0.985)
  BƯỚC 7  : Predict test_featured.csv → submission.csv

YÊU CẦU: Phải chạy preprocess.py trước.

Cách chạy:
    cd /home/minhtriet/Documents/rossman
    python -m src.preprocess   # lần đầu (hoặc khi dữ liệu thay đổi)
    python -m src.train        # train & submit
"""
import sys, os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.model import (
    rmspe, final_predict, get_all_feature_cols,
    run_random_feature_selection, find_best_pairs,
    build_combined_model, build_season_model,
    build_month_ahead_model,
)

# ──── Cấu hình ───────────────────────────────────────────────
ROOT_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT_DIR, 'data', 'processed')
OUT_DIR   = ROOT_DIR   # submission.csv xuất ra thư mục gốc


# ============================================================
# HELPER
# ============================================================

def _check_files():
    """Kiểm tra 3 file đầu vào tồn tại, in cảnh báo nếu thiếu."""
    needed = ['train_featured.csv', 'holdout_featured.csv', 'test_featured.csv']
    missing = [f for f in needed if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print("❌  Thiếu file(s):")
        for f in missing:
            print(f"      {os.path.join(DATA_DIR, f)}")
        print("\n👉  Hãy chạy trước: python -m src.preprocess")
        sys.exit(1)


# ============================================================
# MAIN
# ============================================================

def main():
    _check_files()

    # ── BƯỚC 5A: Đọc dữ liệu đã xử lý ──────────────────────
    print("=" * 60)
    print("BƯỚC 5A: Đọc dữ liệu đã pre-processed")
    print("=" * 60)

    train_df   = pd.read_csv(os.path.join(DATA_DIR, 'train_featured.csv'),   low_memory=False)
    holdout_df = pd.read_csv(os.path.join(DATA_DIR, 'holdout_featured.csv'), low_memory=False)
    test_df    = pd.read_csv(os.path.join(DATA_DIR, 'test_featured.csv'),    low_memory=False)

    for df, name in [(train_df, 'train'), (holdout_df, 'holdout'), (test_df, 'test')]:
        print(f"  {name:8s}: {df.shape[0]:,} rows × {df.shape[1]} cols")

    all_feature_cols = get_all_feature_cols(train_df)
    print(f"\n  Total features available: {len(all_feature_cols)}")

    # ── BƯỚC 5B: Random Feature Selection ───────────────────
    print("\n" + "=" * 60)
    print("BƯỚC 5B: Random Feature Selection (500 probe models)")
    print("=" * 60)

    results_df, models_dict = run_random_feature_selection(
        train_df         = train_df,
        holdout_df       = holdout_df,
        all_feature_cols = all_feature_cols,
        n_models         = 500,
        min_features     = 20,
        max_features     = 60,
    )

    # ── BƯỚC 5C: Tìm cặp ensemble tốt nhất ─────────────────
    print("\n" + "=" * 60)
    print("BƯỚC 5C: Tìm cặp ensemble tốt nhất")
    print("=" * 60)

    pairs_df = find_best_pairs(
        results_df  = results_df,
        models_dict = models_dict,
        holdout_df  = holdout_df,
        top_n       = 50,
    )

    # ── BƯỚC 5D: Combined Model ──────────────────────────────
    print("\n" + "=" * 60)
    print("BƯỚC 5D: Combined Model (full params, 5000 rounds)")
    print("=" * 60)

    combined_model, combined_feats, combined_score = build_combined_model(
        pairs_df   = pairs_df,
        results_df = results_df,
        train_df   = train_df,
        holdout_df = holdout_df,
        top_pairs  = 10,
    )

    # ── BƯỚC 5E: Season & Month-ahead Models ────────────────
    print("\n" + "=" * 60)
    print("BƯỚC 5E: Season Model + Month-ahead Model")
    print("=" * 60)

    season_model, season_score = build_season_model(
        train_df     = train_df,
        holdout_df   = holdout_df,
        feature_cols = combined_feats,
    )

    month_ahead_model, month_ahead_feats, ma_score = build_month_ahead_model(
        train_df     = train_df,
        holdout_df   = holdout_df,
        feature_cols = combined_feats,
    )

    # Top 2 handpicked models từ probe phase
    top2_ids   = results_df.head(2)['model_id'].tolist()
    top2_feats = [
        results_df[results_df['model_id'] == mid]['features'].values[0]
        for mid in top2_ids
    ]
    top2_models = [(models_dict[mid], feats)
                   for mid, feats in zip(top2_ids, top2_feats)]

    # ── BƯỚC 5F: Đánh giá Final Ensemble trên Holdout ───────
    print("\n" + "=" * 60)
    print("BƯỚC 5F: Final Ensemble – Validate trên Holdout")
    print("=" * 60)

    final_models = [
        (combined_model,    combined_feats),
        (season_model,      combined_feats),
        (month_ahead_model, month_ahead_feats),
        *top2_models,
    ]

    holdout_filtered = holdout_df[
        (holdout_df['Open'] == 1) & (holdout_df['Sales'] > 0)
    ].copy()

    holdout_pred = final_predict(final_models, holdout_filtered)
    final_rmspe  = rmspe(holdout_filtered['Sales'].values, holdout_pred)

    print(f"\n{'=' * 60}")
    print(f"  FINAL ENSEMBLE RMSPE (holdout) : {final_rmspe:.5f}")
    print(f"  Combined model RMSPE           : {combined_score:.5f}")
    print(f"  Season model RMSPE             : {season_score:.5f}")
    print(f"  Month-ahead model RMSPE        : {ma_score:.5f}")
    print(f"{'=' * 60}")
    print(f"  (Paper đạt ~0.100 trên private leaderboard)")

    # ── BƯỚC 7: Predict Test & Submission ───────────────────
    print("\n" + "=" * 60)
    print("BƯỚC 7: Predict Test Set → submission.csv")
    print("=" * 60)

    predictions = final_predict(
        models_and_features = final_models,
        test_df             = test_df,
        correction_factor   = 0.985,
    )

    test_out = test_df.copy()
    test_out['Sales'] = predictions

    # Hậu xử lý: cửa hàng đóng cửa → Sales phải bằng 0
    test_out.loc[test_out['Open'] == 0, 'Sales'] = 0

    submission = test_out[['Id', 'Sales']].sort_values('Id')
    sub_path   = os.path.join(OUT_DIR, 'submission.csv')
    submission.to_csv(sub_path, index=False)

    print(f"\n🎉 Đã xuất '{sub_path}' thành công!")
    print(f"   Rows : {len(submission):,}")
    print(f"   Sales: {submission['Sales'].min():.0f} – {submission['Sales'].max():.0f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rossmann Training Pipeline")
    parser.add_argument('--input_dir', type=str, default=DATA_DIR, help='Thư mục chứa dữ liệu đã preprocessed (train_featured.csv, ...)')
    parser.add_argument('--output_dir', type=str, default=OUT_DIR, help='Thư mục lưu file kết quả submission.csv')
    args = parser.parse_args()

    DATA_DIR = args.input_dir
    OUT_DIR = args.output_dir

    main()
