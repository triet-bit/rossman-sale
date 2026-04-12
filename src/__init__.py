# Rossmann Feature Engineering & Modelling Pipeline
# Two-stage design:
#   Stage 1 (preprocess.py) → Feature Engineering → data/processed/*.csv
#   Stage 2 (train.py)      → XGBoost training    → submission.csv
from .features_recent  import add_recent_features
from .features_temporal import add_temporal_features
from .features_trend    import add_trend_features_optimized
from .model             import (
    XGB_PARAMS, PROBE_PARAMS, N_ROUNDS, EARLY_STOPPING,
    rmspe, rmspe_xgb,
    prepare_data, train_xgb_model, get_all_feature_cols,
    run_random_feature_selection, find_best_pairs,
    build_combined_model, build_season_model,
    build_month_ahead_model, final_predict,
)
from .model_lgbm import (
    run_random_feature_selection_lgbm,
    find_best_pairs_lgbm,
    build_combined_model_lgbm,
    build_season_model_lgbm,
    build_month_ahead_model_lgbm,
    final_predict_lgbm,
    final_predict_cross_ensemble,
)
__all__ = [
    'add_recent_features',
    'add_temporal_features',
    'add_trend_features_optimized',
    'XGB_PARAMS', 'PROBE_PARAMS', 'N_ROUNDS', 'EARLY_STOPPING',
    'rmspe', 'rmspe_xgb',
    'prepare_data', 'train_xgb_model', 'get_all_feature_cols',
    'run_random_feature_selection', 'find_best_pairs',
    'build_combined_model', 'build_season_model',
    'build_month_ahead_model', 'final_predict',
]
