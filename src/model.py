"""
BƯỚC 5 – XGBoost Training & Ensemble
=======================================
Gồm:
  5A – Lấy feature columns
  5B – Random Feature Selection (500 probe models)
  5C – Tìm cặp ensemble tốt nhất (harmonic mean)
  5D – Combined Model (gộp features từ top pairs)
  5E – Season Model + Month-ahead Model
  5F – Final Ensemble (harmonic mean × correction_factor)

Thứ tự tham số training:
  - Giai đoạn THĂM DÒ (5B): PROBE_PARAMS + PROBE_ROUNDS (nhanh ~10x)
  - Giai đoạn CHÍNH THỨC (5D/5E): XGB_PARAMS + N_ROUNDS (chính xác)
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# METRIC: RMSPE
# ============================================================

def rmspe(y_true, y_pred):
    """Root Mean Square Percentage Error (chỉ tính trên Sales > 0)."""
    mask   = y_true > 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return np.sqrt(np.mean(((y_true - y_pred) / y_true) ** 2))


def rmspe_xgb(y_pred, dtrain):
    """Custom eval metric cho XGBoost (dự đoán trong log-space)."""
    y_true = np.expm1(dtrain.get_label())
    y_pred = np.expm1(y_pred)
    return 'rmspe', rmspe(y_true, y_pred)


# ============================================================
# XGB PARAMS
# ============================================================

XGB_PARAMS = {
    'objective':        'reg:squarederror',
    'eta':              0.02,
    'max_depth':        10,
    'subsample':        0.9,
    'colsample_bytree': 0.3,     # paper đổi từ 0.7 → 0.3 vì có nhiều features
    'min_child_weight': 5,
    'nthread':          -1,
    'seed':             42,
    'eval_metric':      'rmse',
    'tree_method':      'hist',  # histogram-based: nhanh ~3-5x trên CPU
    'device':           'cuda',  # GPU nếu có; đổi 'cpu' nếu chạy CPU
}

# Params nhẹ hơn cho giai đoạn THĂM DÒ (500 models)
PROBE_PARAMS = {
    **XGB_PARAMS,
    'eta':              0.1,
    'colsample_bytree': 0.5,
}
PROBE_ROUNDS   = 500   # đủ để xếp hạng feature subsets

N_ROUNDS       = 5000  # paper dùng 5000
EARLY_STOPPING = 100


# ============================================================
# HELPERS
# ============================================================

def prepare_data(df, feature_cols, target_col='log_Sales'):
    """
    Tách X, y từ DataFrame.
    - Lọc Open=1 và Sales>0.
    - Tạo log_Sales nếu chưa có.
    - Chỉ lấy feature_cols tồn tại trong df.
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
    y = df[target_col]
    return X, y, df


def train_xgb_model(X_train, y_train, X_val, y_val,
                    params=None, n_rounds=N_ROUNDS):
    """Train một XGBoost model, trả về (model, val_rmspe)."""
    if params is None:
        params = XGB_PARAMS

    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval   = xgb.DMatrix(X_val,   label=y_val, enable_categorical=True)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round       = n_rounds,
        evals                 = [(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds = EARLY_STOPPING,
        verbose_eval          = False,
        custom_metric         = rmspe_xgb,
    )

    val_pred = np.expm1(model.predict(dval))
    val_true = np.expm1(y_val.values)
    score    = rmspe(val_true, val_pred)
    return model, score


def get_all_feature_cols(df):
    """Trả về list tất cả numeric feature columns (bỏ target và meta)."""
    exclude = {
        'Sales', 'log_Sales', 'Customers', 'Open',
        'Store', 'Date', 'state_holiday_flag', 'competition_open_date',
    }
    return [
        col for col in df.columns
        if col not in exclude
    ]


# ============================================================
# BƯỚC 5B: RANDOM FEATURE SELECTION
# ============================================================

def run_random_feature_selection(
    train_df,
    holdout_df,
    all_feature_cols,
    n_models     = 500,
    min_features = 20,
    max_features = 60,
    random_seed  = 42,
):
    """
    Chạy n_models XGBoost với PROBE_PARAMS (nhanh) để xếp hạng feature subsets.
    Trả về (results_df, models_dict) sắp xếp theo RMSPE tăng dần.
    """
    np.random.seed(random_seed)
    results = []

    print(f"Chạy {n_models} probe models (eta={PROBE_PARAMS['eta']}, rounds={PROBE_ROUNDS})...")
    print(f"Features available: {len(all_feature_cols)} | subset: {min_features}–{max_features}\n")

    for i in range(n_models):
        n_feats  = np.random.randint(min_features, max_features + 1)
        feat_sub = list(np.random.choice(all_feature_cols, size=n_feats, replace=False))

        X_train, y_train, _ = prepare_data(train_df,   feat_sub)
        X_val,   y_val,   _ = prepare_data(holdout_df, feat_sub)

        # Dùng PROBE_PARAMS (eta=0.1, 500 rounds) thay vì full params
        # → nhanh ~10x, vẫn đủ để xếp hạng chất lượng feature subsets
        model, score = train_xgb_model(
            X_train, y_train, X_val, y_val,
            params=PROBE_PARAMS,
            n_rounds=PROBE_ROUNDS,
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
            print(f"  [{i+1}/{n_models}] Best RMSPE so far: {best:.5f}")

    results_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'model'}
        for r in results
    ]).sort_values('rmspe').reset_index(drop=True)

    models_dict = {r['model_id']: r['model'] for r in results}

    print(f"\nTop 10 probe models:")
    print(results_df[['model_id', 'n_features', 'rmspe']].head(10).to_string())
    return results_df, models_dict


# ============================================================
# BƯỚC 5C: TÌM CẶP ENSEMBLE TỐT NHẤT
# ============================================================

def find_best_pairs(results_df, models_dict, holdout_df, top_n=50):
    """
    Với top_n models tốt nhất, tính RMSPE của từng cặp ensemble
    (harmonic mean của 2 predictions).
    Trả về DataFrame sắp xếp theo rmspe_ensemble tăng dần.
    """
    top_models = results_df.head(top_n)
    n_pairs    = top_n * (top_n - 1) // 2
    print(f"Tính ensemble RMSPE cho {n_pairs} cặp models...")

    # Cache predictions
    cache = {}
    for _, row in top_models.iterrows():
        mid  = row['model_id']
        X_val, y_val, _ = prepare_data(holdout_df, row['features'])
        pred_log  = models_dict[mid].predict(xgb.DMatrix(X_val, enable_categorical=True))
        cache[mid] = {
            'pred': np.expm1(pred_log),
            'true': np.expm1(y_val.values),
        }

    pair_results = []
    for id1, id2 in combinations(list(top_models['model_id']), 2):
        p1   = cache[id1]['pred']
        p2   = cache[id2]['pred']
        ens  = 2 / (1 / p1 + 1 / p2)   # harmonic mean of 2
        pair_results.append({
            'model_1':        id1,
            'model_2':        id2,
            'rmspe_single1':  results_df[results_df['model_id'] == id1]['rmspe'].values[0],
            'rmspe_single2':  results_df[results_df['model_id'] == id2]['rmspe'].values[0],
            'rmspe_ensemble': rmspe(cache[id1]['true'], ens),
        })

    pairs_df = pd.DataFrame(pair_results).sort_values('rmspe_ensemble').reset_index(drop=True)
    print("\nTop 10 model pairs:")
    print(pairs_df.head(10).to_string())
    return pairs_df


# ============================================================
# BƯỚC 5D: COMBINED MODEL
# ============================================================

def build_combined_model(pairs_df, results_df, train_df, holdout_df,
                         top_pairs=10, n_rounds=N_ROUNDS):
    """
    Gộp features từ top N cặp → train 1 model chính xác (XGB_PARAMS + N_ROUNDS).
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

    print(f"Combined model: {len(combined_feats)} features từ {len(top_ids)} models")

    X_train, y_train, _ = prepare_data(train_df,   combined_feats)
    X_val,   y_val,   _ = prepare_data(holdout_df, combined_feats)

    print(f"Training combined model (eta=0.02, {n_rounds} rounds)...")
    model, score = train_xgb_model(X_train, y_train, X_val, y_val, n_rounds=n_rounds)
    print(f"Combined model RMSPE: {score:.5f}")
    return model, combined_feats, score


# ============================================================
# BƯỚC 5E: SEASON + MONTH-AHEAD MODELS
# ============================================================

def build_season_model(train_df, holdout_df, feature_cols, n_rounds=N_ROUNDS):
    """Train chỉ trên tháng 5–9 (May–Sep) để capture pattern mùa hè."""
    if isinstance(train_df.index, pd.DatetimeIndex):
        months = train_df.index.month
    else:
        months = pd.to_datetime(train_df['Date']).dt.month
    season_train = train_df[months.isin([5, 6, 7, 8, 9])]
    print(f"Season model: {len(season_train):,} rows (May–Sep only)")

    X_train, y_train, _ = prepare_data(season_train, feature_cols)
    X_val,   y_val,   _ = prepare_data(holdout_df,   feature_cols)
    model, score = train_xgb_model(X_train, y_train, X_val, y_val, n_rounds=n_rounds)
    print(f"Season model RMSPE: {score:.5f}")
    return model, score


def build_month_ahead_model(train_df, holdout_df, feature_cols, n_rounds=N_ROUNDS):
    """
    Model 'month-ahead': bỏ các recent_q_ features (quarter = 90 ngày gần nhất)
    để simulate trường hợp không có 1 tháng dữ liệu gần nhất.
    """
    feats = [f for f in feature_cols if not f.startswith('recent_q_')]
    print(f"Month-ahead model: {len(feats)} features (bỏ recent_q)")

    X_train, y_train, _ = prepare_data(train_df,   feats)
    X_val,   y_val,   _ = prepare_data(holdout_df, feats)
    model, score = train_xgb_model(X_train, y_train, X_val, y_val, n_rounds=n_rounds)
    print(f"Month-ahead model RMSPE: {score:.5f}")
    return model, feats, score


# ============================================================
# BƯỚC 5F: FINAL ENSEMBLE
# ============================================================

def final_predict(models_and_features, test_df, correction_factor=0.985):
    """
    Ensemble cuối bằng Harmonic Mean của tất cả models × correction_factor.

    Parameters
    ----------
    models_and_features : list of (xgb_model, feature_cols)
    test_df             : DataFrame cần predict
    correction_factor   : 0.985 (paper) để bù log-space bias

    Notes
    -----
    Hàm mục tiêu RMSE trên log(sales) có xu hướng under-estimate khi
    chuyển về không gian gốc. Nhân 0.985 kéo phân phối về đúng trung tâm.
    """
    all_preds = []
    for model, feats in models_and_features:
        available = [f for f in feats if f in test_df.columns]
        X_test    = test_df[available].fillna(-999)
        pred_log  = model.predict(xgb.DMatrix(X_test, enable_categorical=True))
        all_preds.append(np.expm1(pred_log))

    all_preds  = np.array(all_preds)              # (n_models, n_rows)

    # Harmonic Mean ổn định số học
    eps        = 1e-6
    safe_preds = np.maximum(all_preds, eps)
    n_m        = safe_preds.shape[0]
    ensemble   = n_m / np.sum(1.0 / safe_preds, axis=0)

    return ensemble * correction_factor
