"""
train_ensemble.py – XGBoost + LightGBM trong 1 lệnh duy nhất
=============================================================
Chạy cả 2 pipeline trong cùng session → ensemble trực tiếp trong memory.
Không cần lưu/load pkl, không cần chạy 2 lệnh riêng.

Cách chạy (Kaggle):
    !python -m src.train_ensemble \\
        --input_dir  /kaggle/input/datasets/trietdeptrai/preprocess-dataset \\
        --output_dir /kaggle/working/output \\
        --xgb_weight 0.5

Output (trong --output_dir):
    submission_xgb.csv              ← XGBoost only
    submission_lgbm.csv             ← LightGBM only
    submission_cross_ensemble.csv   ← XGB + LGBM ensemble  ← NỘP CÁI NÀY
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# XGBoost pipeline
from src.model import (
    rmspe as rmspe_xgb_fn,
    get_all_feature_cols,
    run_random_feature_selection,
    find_best_pairs,
    build_combined_model,
    build_season_model,
    build_month_ahead_model,
    final_predict,
)

# LightGBM pipeline
from src.model_lgbm import (
    rmspe,
    run_random_feature_selection_lgbm,
    find_best_pairs_lgbm,
    build_combined_model_lgbm,
    build_season_model_lgbm,
    build_month_ahead_model_lgbm,
    final_predict_lgbm,
    final_predict_cross_ensemble,
)

# ──── Cấu hình mặc định ──────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data', 'processed')
OUT_DIR  = ROOT_DIR


# ============================================================
# HELPER
# ============================================================

def _check_files(data_dir):
    needed  = ['train_featured.csv', 'holdout_featured.csv', 'test_featured.csv']
    missing = [f for f in needed if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        print("Thiếu file(s):")
        for f in missing:
            print(f"      {os.path.join(data_dir, f)}")
        print("\nHãy chạy trước: python -m src.preprocess")
        sys.exit(1)


def _save_submission(test_df, preds, out_dir, filename):
    test_out = test_df.copy()
    test_out['Sales'] = preds
    test_out.loc[test_out['Open'] == 0, 'Sales'] = 0
    sub  = test_out[['Id', 'Sales']].sort_values('Id')
    path = os.path.join(out_dir, filename)
    sub.to_csv(path, index=False)
    print(f"  Đã xuất: {path}  ({len(sub):,} rows, "
          f"Sales: {sub['Sales'].min():.0f}–{sub['Sales'].max():.0f})")
    return path


def _banner(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# XGB PIPELINE
# ============================================================

def run_xgb_pipeline(train_df, holdout_df):
    _banner("PIPELINE XGBoost")
    all_feature_cols = get_all_feature_cols(train_df)
    print(f"  Total features: {len(all_feature_cols)}")

    results_df, models_dict = run_random_feature_selection(
        train_df, holdout_df, all_feature_cols,
        n_models=500, min_features=20, max_features=60,
    )
    pairs_df = find_best_pairs(results_df, models_dict, holdout_df, top_n=50)
    combined_model, combined_feats, combined_score = build_combined_model(
        pairs_df, results_df, train_df, holdout_df, top_pairs=10,
    )
    season_model, season_score = build_season_model(train_df, holdout_df, combined_feats)
    month_ahead_model, month_ahead_feats, ma_score = build_month_ahead_model(
        train_df, holdout_df, combined_feats,
    )

    top2_ids    = results_df.head(2)['model_id'].tolist()
    top2_feats  = [results_df[results_df['model_id']==mid]['features'].values[0] for mid in top2_ids]
    top2_models = [(models_dict[mid], feats) for mid, feats in zip(top2_ids, top2_feats)]

    final_models = [
        (combined_model,    combined_feats),
        (season_model,      combined_feats),
        (month_ahead_model, month_ahead_feats),
        *top2_models,
    ]

    holdout_filtered = holdout_df[(holdout_df['Open']==1) & (holdout_df['Sales']>0)].copy()
    holdout_pred     = final_predict(final_models, holdout_filtered)
    final_rmspe      = rmspe(holdout_filtered['Sales'].values, holdout_pred)

    print(f"\n  [XGB] Final RMSPE (holdout) : {final_rmspe:.5f}")
    print(f"  [XGB] Combined              : {combined_score:.5f}")
    print(f"  [XGB] Season                : {season_score:.5f}")
    print(f"  [XGB] Month-ahead           : {ma_score:.5f}")

    return final_models, final_rmspe


# ============================================================
# LGBM PIPELINE
# ============================================================

def run_lgbm_pipeline(train_df, holdout_df):
    _banner("PIPELINE LightGBM")
    all_feature_cols = get_all_feature_cols(train_df)
    print(f"  Total features: {len(all_feature_cols)}")

    results_df, models_dict = run_random_feature_selection_lgbm(
        train_df, holdout_df, all_feature_cols,
        n_models=500, min_features=20, max_features=60,
    )
    pairs_df = find_best_pairs_lgbm(results_df, models_dict, holdout_df, top_n=50)
    combined_model, combined_feats, combined_score = build_combined_model_lgbm(
        pairs_df, results_df, train_df, holdout_df, top_pairs=10,
    )
    season_model, season_score = build_season_model_lgbm(train_df, holdout_df, combined_feats)
    month_ahead_model, month_ahead_feats, ma_score = build_month_ahead_model_lgbm(
        train_df, holdout_df, combined_feats,
    )

    top2_ids    = results_df.head(2)['model_id'].tolist()
    top2_feats  = [results_df[results_df['model_id']==mid]['features'].values[0] for mid in top2_ids]
    top2_models = [(models_dict[mid], feats) for mid, feats in zip(top2_ids, top2_feats)]

    final_models = [
        (combined_model,    combined_feats),
        (season_model,      combined_feats),
        (month_ahead_model, month_ahead_feats),
        *top2_models,
    ]

    holdout_filtered = holdout_df[(holdout_df['Open']==1) & (holdout_df['Sales']>0)].copy()
    holdout_pred     = final_predict_lgbm(final_models, holdout_filtered)
    final_rmspe      = rmspe(holdout_filtered['Sales'].values, holdout_pred)

    print(f"\n  [LGBM] Final RMSPE (holdout) : {final_rmspe:.5f}")
    print(f"  [LGBM] Combined               : {combined_score:.5f}")
    print(f"  [LGBM] Season                 : {season_score:.5f}")
    print(f"  [LGBM] Month-ahead            : {ma_score:.5f}")

    return final_models, final_rmspe


# ============================================================
# MAIN
# ============================================================

def main(data_dir, out_dir, xgb_weight=0.5):
    _check_files(data_dir)
    os.makedirs(out_dir, exist_ok=True)

    _banner("ĐỌC DỮ LIỆU")
    train_df   = pd.read_csv(os.path.join(data_dir, 'train_featured.csv'),   low_memory=False)
    holdout_df = pd.read_csv(os.path.join(data_dir, 'holdout_featured.csv'), low_memory=False)
    test_df    = pd.read_csv(os.path.join(data_dir, 'test_featured.csv'),    low_memory=False)
    for df, name in [(train_df,'train'),(holdout_df,'holdout'),(test_df,'test')]:
        print(f"  {name:8s}: {df.shape[0]:,} rows × {df.shape[1]} cols")

    xgb_models,  xgb_rmspe  = run_xgb_pipeline(train_df, holdout_df)
    lgbm_models, lgbm_rmspe = run_lgbm_pipeline(train_df, holdout_df)

    _banner("XUẤT SUBMISSIONS")

    xgb_preds  = final_predict(xgb_models, test_df, correction_factor=0.985)
    lgbm_preds = final_predict_lgbm(lgbm_models, test_df, correction_factor=0.985)

    _save_submission(test_df, xgb_preds,  out_dir, 'submission_xgb.csv')
    _save_submission(test_df, lgbm_preds, out_dir, 'submission_lgbm.csv')

    _banner(f"CROSS-ENSEMBLE  (XGB w={xgb_weight:.2f} | LGBM w={1-xgb_weight:.2f})")

    holdout_filtered = holdout_df[(holdout_df['Open']==1) & (holdout_df['Sales']>0)].copy()
    cross_holdout    = final_predict_cross_ensemble(
        xgb_models, lgbm_models, holdout_filtered, xgb_weight=xgb_weight,
    )
    cross_rmspe = rmspe(holdout_filtered['Sales'].values, cross_holdout)

    cross_preds = final_predict_cross_ensemble(
        xgb_models, lgbm_models, test_df, xgb_weight=xgb_weight,
    )
    _save_submission(test_df, cross_preds, out_dir, 'submission_cross_ensemble.csv')

    _banner("KẾT QUẢ SO SÁNH (Holdout RMSPE — thấp hơn = tốt hơn)")
    print(f"  {'Model':<30} {'RMSPE':>10}")
    print(f"  {'-'*42}")
    print(f"  {'XGBoost only':<30} {xgb_rmspe:>10.5f}")
    print(f"  {'LightGBM only':<30} {lgbm_rmspe:>10.5f}")
    print(f"  {'Cross-Ensemble (XGB+LGBM)':<30} {cross_rmspe:>10.5f}")
    print(f"  {'-'*42}")

    best = min(xgb_rmspe, lgbm_rmspe, cross_rmspe)
    if best == cross_rmspe:
        rec = 'submission_cross_ensemble.csv'
    elif best == lgbm_rmspe:
        rec = 'submission_lgbm.csv'
    else:
        rec = 'submission_xgb.csv'
    print(f"\n  Nộp Kaggle: {rec}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rossmann XGB + LGBM Cross-Ensemble")
    parser.add_argument('--input_dir',  type=str,   default=DATA_DIR,
                        help='Thư mục chứa *_featured.csv')
    parser.add_argument('--output_dir', type=str,   default=OUT_DIR,
                        help='Thư mục lưu submission*.csv')
    parser.add_argument('--xgb_weight', type=float, default=0.5,
                        help='Trọng số XGBoost (0.0–1.0), LGBM = 1 - xgb_weight')
    args = parser.parse_args()

    main(
        data_dir   = args.input_dir,
        out_dir    = args.output_dir,
        xgb_weight = args.xgb_weight,
    )