"""
BƯỚC 5 – LightGBM Training & Ensemble
=======================================
Phiên bản LightGBM tương đương với model.py (XGBoost).
Cấu trúc giữ nguyên để dễ so sánh và ensemble 2 framework.

Gồm:
  5A – Lấy feature columns          (dùng chung với XGBoost)
  5B – Random Feature Selection      (500 probe models, LGBM nhanh hơn ~2-3x)
  5C – Tìm cặp ensemble tốt nhất    (harmonic mean)
  5D – Combined Model                (LGBM_PARAMS + N_ROUNDS)
  5E – Season Model + Month-ahead Model
  5F – Final Ensemble                (harmonic mean × correction_factor)

Cách sử dụng cùng train.py:
  from src.model_lgbm import (
      run_random_feature_selection_lgbm,
      find_best_pairs_lgbm,
      build_combined_model_lgbm,
      build_season_model_lgbm,
      build_month_ahead_model_lgbm,
      final_predict_lgbm,
  )
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# METRIC: RMSPE  (dùng chung, import từ model.py nếu muốn)
# ============================================================

def rmspe(y_true, y_pred):
    """Root Mean Square Percentage Error (chỉ tính trên Sales > 0)."""
    mask   = y_true > 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))


def rmspe_lgbm(y_pred, dataset):
    """Custom eval metric cho LightGBM (dự đoán trong log-space)."""
    y_true = np.expm1(dataset.get_label())
    y_pred = np.expm1(y_pred)
    score  = rmspe(y_true, y_pred)
    # LightGBM: trả về (name, value, higher_is_better)
    return 'rmspe', score, False


# ============================================================
# LGBM PARAMS
# ============================================================

LGBM_PARAMS = {
    'objective':        'regression',
    'metric':           'rmse',
    'learning_rate':    0.02,
    'num_leaves':       31,          
    'max_depth':        -1,          
    'subsample':        0.9,
    'subsample_freq':   1,
    'colsample_bytree': 0.3,         
    'min_child_samples': 20,         
    'reg_alpha':        0.0,
    'reg_lambda':       1.0,
    'n_jobs':           -1,
    'seed':             42,
    'verbose':          -1,
    'device':           'gpu',
}

PROBE_PARAMS_LGBM = {
    **LGBM_PARAMS,
    'learning_rate':    0.1,
    'colsample_bytree': 0.5,
}

PROBE_ROUNDS   = 100
N_ROUNDS       = 1000
EARLY_STOPPING = 100


# ============================================================
# HELPERS
# ============================================================

def prepare_data_lgbm(df, feature_cols, target_col='log_Sales'):
    """
    Tách X, y từ DataFrame cho LightGBM.
    - Lọc Open=1 và Sales>0.
    - Tạo log_Sales nếu chưa có.
    - Dùng label-encode (int16) thay vì native category để tránh lỗi
      "train and valid dataset categorical_feature do not match"
      khi train/val có số lượng unique values khác nhau sau khi filter.
    """
    df = df.copy()
    if 'Open' in df.columns:
        df = df[df['Open'] == 1]
    if 'Sales' in df.columns:
        df = df[df['Sales'] > 0]

    if target_col not in df.columns:
        df[target_col] = np.log1p(df['Sales'])

    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  Warning: {len(missing)} features không tìm thấy, bỏ qua.")

    X = df[available].fillna(-999)

    # Label-encode tất cả non-numeric → int16 (giống XGBoost)
    # Tránh lỗi categorical mismatch giữa train và valid dataset
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        X = X.copy()
        for col in non_numeric:
            X[col] = X[col].astype('category').cat.codes.astype(np.int16)

    y = df[target_col]
    return X, y, df


def get_all_feature_cols(df):
    """
    Trả về list tất cả feature columns dùng được.
    (Hàm này giống hệt model.py — có thể import trực tiếp từ đó.)
    """
    exclude = {
        'Sales', 'log_Sales', 'Customers', 'Open',
        'Store', 'Date', 'state_holiday_flag', 'competition_open_date', 'Id',
    }
    result = []
    for col in df.columns:
        if col in exclude:
            continue
        dtype_str = str(df[col].dtype)
        if dtype_str in ('object', 'string', 'StringDtype') or df[col].dtype == 'object':
            if df[col].nunique() > 100:
                continue
        result.append(col)
    return result


def train_lgbm_model(X_train, y_train, X_val, y_val,
                     params=None, n_rounds=N_ROUNDS):
    """Train một LightGBM model, trả về (model, val_rmspe)."""
    if params is None:
        params = LGBM_PARAMS

    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dval   = lgb.Dataset(X_val,   label=y_val,   reference=dtrain, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING, verbose=False),
        lgb.log_evaluation(period=-1),   # tắt log từng round
    ]

    model = lgb.train(
        params,
        dtrain,
        num_boost_round    = n_rounds,
        valid_sets         = [dtrain, dval],
        valid_names        = ['train', 'val'],
        feval              = rmspe_lgbm,
        callbacks          = callbacks,
    )

    val_pred = np.expm1(model.predict(X_val, num_iteration=model.best_iteration))
    val_true = np.expm1(y_val.values)
    score    = rmspe(val_true, val_pred)
    return model, score


# ============================================================
# BƯỚC 5B: RANDOM FEATURE SELECTION (LGBM)
# ============================================================

def run_random_feature_selection_lgbm(
    train_df,
    holdout_df,
    all_feature_cols,
    n_models     = 500,
    min_features = 20,
    max_features = 60,
    random_seed  = 42,
):
    """
    Chạy n_models LightGBM probe models để xếp hạng feature subsets.
    LightGBM nhanh hơn XGBoost ~2-3x ở bước này.
    Trả về (results_df, models_dict) sắp xếp theo RMSPE tăng dần.
    """
    np.random.seed(random_seed)
    results = []

    print(f"[LGBM] Chạy {n_models} probe models "
          f"(lr={PROBE_PARAMS_LGBM['learning_rate']}, rounds={PROBE_ROUNDS})...")
    print(f"Features available: {len(all_feature_cols)} | subset: {min_features}–{max_features}\n")

    for i in range(n_models):
        n_feats  = np.random.randint(min_features, max_features + 1)
        feat_sub = list(np.random.choice(all_feature_cols, size=n_feats, replace=False))

        X_train, y_train, _ = prepare_data_lgbm(train_df,   feat_sub)
        X_val,   y_val,   _ = prepare_data_lgbm(holdout_df, feat_sub)

        model, score = train_lgbm_model(
            X_train, y_train, X_val, y_val,
            params   = PROBE_PARAMS_LGBM,
            n_rounds = PROBE_ROUNDS,
        )

        results.append({
            'model_id':   i,
            'features':   feat_sub,
            'n_features': len(feat_sub),
            'rmspe':      score,
            'model':      model,
        })

        if (i + 1) % 50 == 0:
            best = min(r['rmspe'] for r in results)
            print(f"  [LGBM {i+1}/{n_models}] Best RMSPE so far: {best:.5f}")

    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'model'}
        for r in results
    ]).sort_values('rmspe').reset_index(drop=True)

    models_dict = {r['model_id']: r['model'] for r in results}

    print(f"\n[LGBM] Top 10 probe models:")
    print(results_df[['model_id', 'n_features', 'rmspe']].head(10).to_string())
    return results_df, models_dict


# ============================================================
# BƯỚC 5C: TÌM CẶP ENSEMBLE TỐT NHẤT (LGBM)
# ============================================================

def find_best_pairs_lgbm(results_df, models_dict, holdout_df, top_n=50):
    """
    Tìm cặp ensemble tốt nhất từ top_n LGBM models (harmonic mean).
    """
    top_models = results_df.head(top_n)
    n_pairs    = top_n * (top_n - 1) // 2
    print(f"[LGBM] Tính ensemble RMSPE cho {n_pairs} cặp models...")

    cache = {}
    for _, row in top_models.iterrows():
        mid  = row['model_id']
        X_val, y_val, _ = prepare_data_lgbm(holdout_df, row['features'])
        model    = models_dict[mid]
        pred_log = model.predict(X_val, num_iteration=model.best_iteration)
        cache[mid] = {
            'pred': np.expm1(pred_log),
            'true': np.expm1(y_val.values),
        }

    pair_results = []
    for id1, id2 in combinations(list(top_models['model_id']), 2):
        p1  = cache[id1]['pred']
        p2  = cache[id2]['pred']
        ens = 2 / (1 / p1 + 1 / p2)
        pair_results.append({
            'model_1':        id1,
            'model_2':        id2,
            'rmspe_single1':  results_df[results_df['model_id'] == id1]['rmspe'].values[0],
            'rmspe_single2':  results_df[results_df['model_id'] == id2]['rmspe'].values[0],
            'rmspe_ensemble': rmspe(cache[id1]['true'], ens),
        })

    pairs_df = pd.DataFrame(pair_results).sort_values('rmspe_ensemble').reset_index(drop=True)
    print("\n[LGBM] Top 10 model pairs:")
    print(pairs_df.head(10).to_string())
    return pairs_df


# ============================================================
# BƯỚC 5D: COMBINED MODEL (LGBM)
# ============================================================

def build_combined_model_lgbm(pairs_df, results_df, train_df, holdout_df,
                               top_pairs=10, n_rounds=N_ROUNDS):
    """
    Gộp features từ top N cặp → train 1 LGBM model chính xác.
    """
    top_ids = set()
    for _, row in pairs_df.head(top_pairs).iterrows():
        top_ids.add(row['model_1'])
        top_ids.add(row['model_2'])

    combined_feats = set()
    for mid in top_ids:
        feats = results_df[results_df['model_id'] == mid]['features'].values[0]
        combined_feats.update(feats)
    combined_feats = list(combined_feats)

    print(f"[LGBM] Combined model: {len(combined_feats)} features từ {len(top_ids)} models")

    X_train, y_train, _ = prepare_data_lgbm(train_df,   combined_feats)
    X_val,   y_val,   _ = prepare_data_lgbm(holdout_df, combined_feats)

    print(f"[LGBM] Training combined model (lr=0.02, {n_rounds} rounds)...")
    model, score = train_lgbm_model(X_train, y_train, X_val, y_val, n_rounds=n_rounds)
    print(f"[LGBM] Combined model RMSPE: {score:.5f}")
    return model, combined_feats, score


# ============================================================
# BƯỚC 5E: SEASON + MONTH-AHEAD MODELS (LGBM)
# ============================================================

def build_season_model_lgbm(train_df, holdout_df, feature_cols, n_rounds=N_ROUNDS):
    """Train chỉ trên tháng 5–9 (May–Sep) để capture pattern mùa hè."""
    if isinstance(train_df.index, pd.DatetimeIndex):
        months = train_df.index.month
    else:
        months = pd.to_datetime(train_df['Date']).dt.month
    season_train = train_df[months.isin([5, 6, 7, 8, 9])]
    print(f"[LGBM] Season model: {len(season_train):,} rows (May–Sep only)")

    X_train, y_train, _ = prepare_data_lgbm(season_train, feature_cols)
    X_val,   y_val,   _ = prepare_data_lgbm(holdout_df,   feature_cols)
    model, score = train_lgbm_model(X_train, y_train, X_val, y_val, n_rounds=n_rounds)
    print(f"[LGBM] Season model RMSPE: {score:.5f}")
    return model, score


def build_month_ahead_model_lgbm(train_df, holdout_df, feature_cols, n_rounds=N_ROUNDS):
    """Model 'month-ahead': bỏ recent_q_ features (simulate thiếu 1 tháng gần nhất)."""
    feats = [f for f in feature_cols if not f.startswith('recent_q_')]
    print(f"[LGBM] Month-ahead model: {len(feats)} features (bỏ recent_q)")

    X_train, y_train, _ = prepare_data_lgbm(train_df,   feats)
    X_val,   y_val,   _ = prepare_data_lgbm(holdout_df, feats)
    model, score = train_lgbm_model(X_train, y_train, X_val, y_val, n_rounds=n_rounds)
    print(f"[LGBM] Month-ahead model RMSPE: {score:.5f}")
    return model, feats, score


# ============================================================
# BƯỚC 5F: FINAL ENSEMBLE (LGBM)
# ============================================================

def final_predict_lgbm(models_and_features, test_df, correction_factor=0.985):
    """
    Ensemble cuối bằng Harmonic Mean của tất cả LGBM models × correction_factor.

    Parameters
    ----------
    models_and_features : list of (lgb_model, feature_cols)
    test_df             : DataFrame cần predict
    correction_factor   : 0.985 để bù log-space bias
    """
    all_preds = []
    for model, feats in models_and_features:
        available = [f for f in feats if f in test_df.columns]
        X_test    = test_df[available].fillna(-999)

        non_numeric = [c for c in X_test.columns if not pd.api.types.is_numeric_dtype(X_test[c])]
        if non_numeric:
            X_test = X_test.copy()
            for col in non_numeric:
                X_test[col] = X_test[col].astype('category').cat.codes.astype(np.int16)

        pred_log = model.predict(X_test, num_iteration=model.best_iteration)
        all_preds.append(np.expm1(pred_log))

    all_preds  = np.array(all_preds)
    eps        = 1e-6
    safe_preds = np.maximum(all_preds, eps)
    n_m        = safe_preds.shape[0]
    ensemble   = n_m / np.sum(1.0 / safe_preds, axis=0)
    return ensemble * correction_factor


# ============================================================
# CROSS-FRAMEWORK ENSEMBLE (XGBoost + LightGBM)
# ============================================================

def final_predict_cross_ensemble(
    xgb_models_and_features,
    lgbm_models_and_features,
    test_df,
    xgb_weight         = 0.5,
    correction_factor  = 0.985,
):
    """
    Ensemble XGBoost + LightGBM bằng weighted harmonic mean.

    Parameters
    ----------
    xgb_models_and_features  : list of (xgb_model, feature_cols)
    lgbm_models_and_features : list of (lgb_model, feature_cols)
    test_df                  : DataFrame cần predict
    xgb_weight               : trọng số cho XGBoost (0–1); LGBM = 1 - xgb_weight
    correction_factor        : 0.985

    Notes
    -----
    Weighted harmonic mean:
        H = (w1 + w2) / (w1/p1 + w2/p2)
    Với w1 + w2 = 1, đây là dạng harmonic mean có trọng số.
    Mặc định 50/50, nhưng nên tune dựa trên holdout RMSPE riêng.
    """
    import xgboost as xgb

    # XGBoost predictions
    xgb_preds = []
    for model, feats in xgb_models_and_features:
        available = [f for f in feats if f in test_df.columns]
        X_test    = test_df[available].fillna(-999)
        non_num   = [c for c in X_test.columns if not pd.api.types.is_numeric_dtype(X_test[c])]
        if non_num:
            X_test = X_test.copy()
            for col in non_num:
                X_test[col] = X_test[col].astype('category').cat.codes.astype('int16')
        xgb_preds.append(np.expm1(model.predict(xgb.DMatrix(X_test))))

    # LightGBM predictions
    lgbm_preds = []
    for model, feats in lgbm_models_and_features:
        available = [f for f in feats if f in test_df.columns]
        X_test    = test_df[available].fillna(-999)
        non_num   = [c for c in X_test.columns if not pd.api.types.is_numeric_dtype(X_test[c])]
        if non_num:
            X_test = X_test.copy()
            for col in non_num:
                X_test[col] = X_test[col].astype('category').cat.codes.astype(np.int16)
        lgbm_preds.append(np.expm1(model.predict(X_test, num_iteration=model.best_iteration)))

    eps = 1e-6

    # Ensemble riêng từng framework (harmonic mean)
    xgb_arr  = np.maximum(np.array(xgb_preds), eps)
    lgbm_arr = np.maximum(np.array(lgbm_preds), eps)

    n_xgb  = xgb_arr.shape[0]
    n_lgbm = lgbm_arr.shape[0]

    p_xgb  = n_xgb  / np.sum(1.0 / xgb_arr,  axis=0)
    p_lgbm = n_lgbm / np.sum(1.0 / lgbm_arr, axis=0)

    # Weighted harmonic mean giữa 2 framework
    lgbm_weight = 1.0 - xgb_weight
    w_sum       = xgb_weight + lgbm_weight  # = 1.0
    p_cross     = w_sum / (xgb_weight / np.maximum(p_xgb, eps) +
                            lgbm_weight / np.maximum(p_lgbm, eps))

    return p_cross * correction_factor