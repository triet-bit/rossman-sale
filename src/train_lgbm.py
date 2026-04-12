"""
train_lgbm.py – LightGBM Pipeline + Cross-Ensemble với XGBoost
================================================================
Chạy pipeline LightGBM song song với XGBoost để:
  1. So sánh RMSPE riêng từng framework
  2. Ensemble XGBoost + LightGBM → submission_cross_ensemble.csv

  MODE A: chỉ train LGBM     → python -m src.train_lgbm --mode lgbm
  MODE B: cross-ensemble      → python -m src.train_lgbm --mode ensemble
          (cần chạy train.py trước để có các model XGB)

YÊU CẦU: Phải chạy preprocess.py trước.
"""
import sys, os, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from src.model_lgbm import (
    rmspe,
    get_all_feature_cols,
    run_random_feature_selection_lgbm,
    find_best_pairs_lgbm,
    build_combined_model_lgbm,
    build_season_model_lgbm,
    build_month_ahead_model_lgbm,
    final_predict_lgbm,
    final_predict_cross_ensemble,
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
OUT_DIR  = ROOT_DIR
MODEL_DIR = os.path.join(ROOT_DIR, 'data', 'models')   # lưu models để reuse


def _check_files():
    needed  = ['train_featured.csv', 'holdout_featured.csv', 'test_featured.csv']
    missing = [f for f in needed if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print("Thiếu file(s):", missing)
        print("Chạy trước: python -m src.preprocess")
        sys.exit(1)


# ============================================================
# PIPELINE LGBM
# ============================================================

def run_lgbm_pipeline(train_df, holdout_df, test_df):
    """Chạy toàn bộ 5B→5F bằng LightGBM. Trả về (final_models, final_rmspe)."""
    all_feature_cols = get_all_feature_cols(train_df)
    print(f"\n  Total features available: {len(all_feature_cols)}")

    # 5B
    print("\n" + "="*60)
    print("Random Feature Selection")
    print("="*60)
    results_df, models_dict = run_random_feature_selection_lgbm(
        train_df, holdout_df, all_feature_cols,
        n_models=500, min_features=20, max_features=60,
    )

    # 5C
    print("\n" + "="*60)
    print("Tìm cặp ensemble tốt nhất")
    print("="*60)
    pairs_df = find_best_pairs_lgbm(results_df, models_dict, holdout_df, top_n=50)

    # 5D
    print("\n" + "="*60)
    print("Combined Model")
    print("="*60)
    combined_model, combined_feats, combined_score = build_combined_model_lgbm(
        pairs_df, results_df, train_df, holdout_df, top_pairs=10,
    )

    # 5E
    print("\n" + "="*60)
    print("Season + Month-ahead Models")
    print("="*60)
    season_model, season_score = build_season_model_lgbm(
        train_df, holdout_df, combined_feats,
    )
    month_ahead_model, month_ahead_feats, ma_score = build_month_ahead_model_lgbm(
        train_df, holdout_df, combined_feats,
    )

    top2_ids   = results_df.head(2)['model_id'].tolist()
    top2_feats = [results_df[results_df['model_id']==mid]['features'].values[0] for mid in top2_ids]
    top2_models = [(models_dict[mid], feats) for mid, feats in zip(top2_ids, top2_feats)]

    # 5F validate
    print("\n" + "="*60)
    print("Final Ensemble – Validate trên Holdout")
    print("="*60)

    final_models = [
        (combined_model,    combined_feats),
        (season_model,      combined_feats),
        (month_ahead_model, month_ahead_feats),
        *top2_models,
    ]

    holdout_filtered = holdout_df[
        (holdout_df['Open'] == 1) & (holdout_df['Sales'] > 0)
    ].copy()

    holdout_pred = final_predict_lgbm(final_models, holdout_filtered)
    final_rmspe  = rmspe(holdout_filtered['Sales'].values, holdout_pred)

    print(f"\n{'='*60}")
    print(f"  [LGBM] FINAL ENSEMBLE RMSPE (holdout) : {final_rmspe:.5f}")
    print(f"  [LGBM] Combined model RMSPE           : {combined_score:.5f}")
    print(f"  [LGBM] Season model RMSPE             : {season_score:.5f}")
    print(f"  [LGBM] Month-ahead model RMSPE        : {ma_score:.5f}")
    print(f"{'='*60}")

    return final_models, final_rmspe, combined_feats


# ============================================================
# MAIN
# ============================================================

def main(mode='lgbm', xgb_weight=0.5):
    _check_files()
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("="*60)
    print("ĐỌC DỮ LIỆU ĐÃ PRE-PROCESSED")
    print("="*60)
    train_df   = pd.read_csv(os.path.join(DATA_DIR, 'train_featured.csv'),   low_memory=False)
    holdout_df = pd.read_csv(os.path.join(DATA_DIR, 'holdout_featured.csv'), low_memory=False)
    test_df    = pd.read_csv(os.path.join(DATA_DIR, 'test_featured.csv'),    low_memory=False)
    for df, name in [(train_df,'train'),(holdout_df,'holdout'),(test_df,'test')]:
        print(f"  {name:8s}: {df.shape[0]:,} rows × {df.shape[1]} cols")

    # ── MODE A: Chỉ LGBM ────────────────────────────────────
    lgbm_models, lgbm_rmspe, _ = run_lgbm_pipeline(train_df, holdout_df, test_df)

    # Lưu LGBM submission
    lgbm_preds = final_predict_lgbm(lgbm_models, test_df)
    test_out = test_df.copy()
    test_out['Sales'] = lgbm_preds
    test_out.loc[test_out['Open'] == 0, 'Sales'] = 0
    sub_lgbm = test_out[['Id', 'Sales']].sort_values('Id')
    path_lgbm = os.path.join(OUT_DIR, 'submission_lgbm.csv')
    sub_lgbm.to_csv(path_lgbm, index=False)
    print(f"\n[LGBM] Đã xuất '{path_lgbm}'  (RMSPE={lgbm_rmspe:.5f})")

    # Lưu models để dùng cross-ensemble sau
    lgbm_model_path = os.path.join(MODEL_DIR, 'lgbm_models.pkl')
    with open(lgbm_model_path, 'wb') as f:
        pickle.dump(lgbm_models, f)
    print(f"  LGBM models đã lưu tại: {lgbm_model_path}")

    # ── MODE B: Cross-Ensemble XGB + LGBM ───────────────────
    if mode == 'ensemble':
        xgb_model_path = os.path.join(MODEL_DIR, 'xgb_models.pkl')
        if not os.path.exists(xgb_model_path):
            print(f"\nKhông tìm thấy '{xgb_model_path}'.")
            print(" Hãy chạy train.py với --save_models trước.")
            return

        with open(xgb_model_path, 'rb') as f:
            xgb_models = pickle.load(f)

        print("\n" + "="*60)
        print(f"CROSS-ENSEMBLE: XGB (w={xgb_weight:.2f}) + LGBM (w={1-xgb_weight:.2f})")
        print("="*60)

        cross_preds = final_predict_cross_ensemble(
            xgb_models, lgbm_models, test_df,
            xgb_weight=xgb_weight,
        )
        test_out2 = test_df.copy()
        test_out2['Sales'] = cross_preds
        test_out2.loc[test_out2['Open'] == 0, 'Sales'] = 0
        sub_cross = test_out2[['Id', 'Sales']].sort_values('Id')
        path_cross = os.path.join(OUT_DIR, 'submission_cross_ensemble.csv')
        sub_cross.to_csv(path_cross, index=False)
        print(f"\nCross-Ensemble submission: '{path_cross}'")

        # So sánh holdout RMSPE của 3 submissions
        print("\n" + "="*60)
        print("TỔNG KẾT SO SÁNH")
        print("="*60)

        # XGB holdout (nếu đã lưu score)
        xgb_score_path = os.path.join(MODEL_DIR, 'xgb_holdout_rmspe.txt')
        xgb_score_str  = open(xgb_score_path).read().strip() if os.path.exists(xgb_score_path) else "N/A"
        print(f"  XGBoost holdout RMSPE      : {xgb_score_str}")
        print(f"  LightGBM holdout RMSPE     : {lgbm_rmspe:.5f}")

        holdout_filtered = holdout_df[(holdout_df['Open']==1)&(holdout_df['Sales']>0)].copy()
        cross_holdout_preds = final_predict_cross_ensemble(
            xgb_models, lgbm_models, holdout_filtered,
            xgb_weight=xgb_weight,
        )
        cross_holdout_rmspe = rmspe(holdout_filtered['Sales'].values, cross_holdout_preds)
        print(f"  Cross-Ensemble holdout RMSPE: {cross_holdout_rmspe:.5f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rossmann LightGBM Pipeline")
    parser.add_argument('--mode',       type=str,   default='lgbm',
                        choices=['lgbm','ensemble'],
                        help='lgbm = chỉ train LGBM | ensemble = XGB+LGBM cross ensemble')
    parser.add_argument('--xgb_weight', type=float, default=0.5,
                        help='Trọng số XGBoost trong cross-ensemble (0.0–1.0)')
    parser.add_argument('--input_dir',  type=str,   default=DATA_DIR)
    parser.add_argument('--output_dir', type=str,   default=OUT_DIR)
    args = parser.parse_args()

    DATA_DIR = args.input_dir
    OUT_DIR  = args.output_dir
    main(mode=args.mode, xgb_weight=args.xgb_weight)