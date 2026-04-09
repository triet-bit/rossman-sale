"""
BƯỚC 2 – Recent Features (Vectorized)
======================================
Tính rolling statistics (mean, median, std, skew, harmonic mean, CV, percentiles)
theo 4 timeframes: quarter (90d), half-year (180d), year (365d), 2-year (730d).
Mỗi timeframe được phân tách theo 5 nhóm: all / dow / promo / holiday / dow+promo.

Quan trọng: index của `past_df` bị dịch +30 ngày để tự động loại bỏ tháng gần
nhất, tránh data-leakage từ tương lai vào rolling window.
"""
import numpy as np
import pandas as pd


# ============================================================
# CÔNG THỨC VECTORIZED THAY THẾ .apply()
# ============================================================

def rolling_skew(roll):
    """
    Skew xấp xỉ (Pearson's 2nd): (mean - median) / std.
    Không cần .apply(), chạy hoàn toàn vectorized.
    """
    return (roll.mean() - roll.median()) / (roll.std() + 1e-8)


def rolling_harmonic_mean(roll_sum_reciprocal, roll_count):
    """
    Harmonic mean = n / Σ(1/xᵢ).
    Cần 2 rolling riêng: rolling(1/x).sum() và rolling.count().
    """
    return roll_count / (roll_sum_reciprocal + 1e-8)


def rolling_cv(roll):
    """
    Coefficient of Variation = std / mean.
    Đo mức độ biến động tương đối, thay thế kurtosis.
    """
    return roll.std() / (roll.mean() + 1e-8)


# ============================================================
# HÀM CHÍNH
# ============================================================

def add_recent_features(df, target_col='Sales'):
    """
    Thêm rolling recent features vào `df`.

    Parameters
    ----------
    df         : DataFrame với cột 'Date', 'Store', 'DayOfWeek',
                 'Promo', 'StateHoliday', và `target_col`.
    target_col : Cột doanh thu (mặc định 'Sales').

    Returns
    -------
    DataFrame với DatetimeIndex và các cột `recent_*` được gắn thêm.
    """
    print("Chuẩn bị dữ liệu recent features...")
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

    df = df.sort_values(['Store', 'Date'])
    df['is_holiday'] = (df['StateHoliday'].astype(str) != '0').astype(int)

    # Dịch index +30 ngày → tự động bỏ tháng gần nhất (tránh leakage)
    cols_needed = ['Store', 'DayOfWeek', 'Promo', 'is_holiday', target_col]
    past_df = df[cols_needed].copy()
    past_df.index = past_df.index + pd.Timedelta(days=30)

    # 1/Sales để tính harmonic mean (1 lần duy nhất)
    past_df['recip_sales'] = 1.0 / past_df[target_col].replace(0, np.nan)

    timeframes = {'q': 90, 'h': 180, 'y': 365, 'y2': 730}
    new_features = []

    for tf_name, days in timeframes.items():
        w = f'{days}D'
        print(f"  Timeframe {tf_name} ({days}d)...")

        # ── SPLIT 1: ALL ──────────────────────────────────────
        g    = past_df.groupby('Store')
        roll = g[target_col].rolling(w, min_periods=2)

        new_features += [
            roll.mean()           .rename(f'recent_{tf_name}_all_mean'),
            roll.median()         .rename(f'recent_{tf_name}_all_median'),
            roll.std()            .rename(f'recent_{tf_name}_all_std'),
            roll.quantile(0.1)    .rename(f'recent_{tf_name}_all_p10'),
            roll.quantile(0.9)    .rename(f'recent_{tf_name}_all_p90'),
            rolling_skew(roll)    .rename(f'recent_{tf_name}_all_skew'),
            rolling_cv(roll)      .rename(f'recent_{tf_name}_all_cv'),
        ]
        roll_recip = g['recip_sales'].rolling(w, min_periods=2)
        roll_count = g[target_col].rolling(w, min_periods=2).count()
        new_features.append(
            rolling_harmonic_mean(roll_recip.sum(), roll_count)
            .rename(f'recent_{tf_name}_all_harmonic_mean')
        )

        # ── SPLIT 2: DOW ──────────────────────────────────────
        g_dow    = past_df.groupby(['Store', 'DayOfWeek'])
        roll_dow = g_dow[target_col].rolling(w, min_periods=2)

        new_features += [
            roll_dow.mean()        .rename(f'recent_{tf_name}_dow_mean'),
            roll_dow.median()      .rename(f'recent_{tf_name}_dow_median'),
            roll_dow.std()         .rename(f'recent_{tf_name}_dow_std'),
            roll_dow.quantile(0.1) .rename(f'recent_{tf_name}_dow_p10'),
            roll_dow.quantile(0.9) .rename(f'recent_{tf_name}_dow_p90'),
            rolling_skew(roll_dow) .rename(f'recent_{tf_name}_dow_skew'),
            rolling_cv(roll_dow)   .rename(f'recent_{tf_name}_dow_cv'),
        ]
        roll_recip_dow = g_dow['recip_sales'].rolling(w, min_periods=2)
        roll_count_dow = g_dow[target_col].rolling(w, min_periods=2).count()
        new_features.append(
            rolling_harmonic_mean(roll_recip_dow.sum(), roll_count_dow)
            .rename(f'recent_{tf_name}_dow_harmonic_mean')
        )

        # ── SPLIT 3: PROMO ────────────────────────────────────
        g_promo    = past_df.groupby(['Store', 'Promo'])
        roll_promo = g_promo[target_col].rolling(w, min_periods=2)

        new_features += [
            roll_promo.mean()        .rename(f'recent_{tf_name}_promo_mean'),
            roll_promo.median()      .rename(f'recent_{tf_name}_promo_median'),
            roll_promo.std()         .rename(f'recent_{tf_name}_promo_std'),
            roll_promo.quantile(0.1) .rename(f'recent_{tf_name}_promo_p10'),
            roll_promo.quantile(0.9) .rename(f'recent_{tf_name}_promo_p90'),
            rolling_skew(roll_promo) .rename(f'recent_{tf_name}_promo_skew'),
            rolling_cv(roll_promo)   .rename(f'recent_{tf_name}_promo_cv'),
        ]
        roll_recip_promo = g_promo['recip_sales'].rolling(w, min_periods=2)
        roll_count_promo = g_promo[target_col].rolling(w, min_periods=2).count()
        new_features.append(
            rolling_harmonic_mean(roll_recip_promo.sum(), roll_count_promo)
            .rename(f'recent_{tf_name}_promo_harmonic_mean')
        )

        # ── SPLIT 4: HOLIDAY ──────────────────────────────────
        g_hol    = past_df.groupby(['Store', 'is_holiday'])
        roll_hol = g_hol[target_col].rolling(w, min_periods=2)

        new_features += [
            roll_hol.mean()        .rename(f'recent_{tf_name}_holiday_mean'),
            roll_hol.median()      .rename(f'recent_{tf_name}_holiday_median'),
            roll_hol.std()         .rename(f'recent_{tf_name}_holiday_std'),
            roll_hol.quantile(0.1) .rename(f'recent_{tf_name}_holiday_p10'),
            roll_hol.quantile(0.9) .rename(f'recent_{tf_name}_holiday_p90'),
            rolling_skew(roll_hol) .rename(f'recent_{tf_name}_holiday_skew'),
            rolling_cv(roll_hol)   .rename(f'recent_{tf_name}_holiday_cv'),
        ]
        roll_recip_hol = g_hol['recip_sales'].rolling(w, min_periods=2)
        roll_count_hol = g_hol[target_col].rolling(w, min_periods=2).count()
        new_features.append(
            rolling_harmonic_mean(roll_recip_hol.sum(), roll_count_hol)
            .rename(f'recent_{tf_name}_holiday_harmonic_mean')
        )

        # ── SPLIT 5: DOW + PROMO ─────────────────────────────
        g_dp    = past_df.groupby(['Store', 'DayOfWeek', 'Promo'])
        roll_dp = g_dp[target_col].rolling(w, min_periods=2)

        new_features += [
            roll_dp.mean()        .rename(f'recent_{tf_name}_dow_promo_mean'),
            roll_dp.median()      .rename(f'recent_{tf_name}_dow_promo_median'),
            roll_dp.std()         .rename(f'recent_{tf_name}_dow_promo_std'),
            rolling_skew(roll_dp) .rename(f'recent_{tf_name}_dow_promo_skew'),
        ]

    # ============================================================
    # GỘP TẤT CẢ VÀO DATAFRAME GỐC
    # ============================================================
    print("Ghép features vào DataFrame gốc...")

    cleaned_features = []
    for s in new_features:
        levels_to_drop = [lvl for lvl in s.index.names if lvl not in ['Store', 'Date']]
        if levels_to_drop:
            s = s.droplevel(levels_to_drop)
        cleaned_features.append(s)

    features_df = pd.concat(cleaned_features, axis=1).reset_index()
    df_out = df.reset_index().merge(features_df, on=['Store', 'Date'], how='left')
    df_out = df_out.set_index('Date')

    new_cols = [c for c in df_out.columns if c.startswith('recent_')]
    print(f"Tổng features mới : {len(new_cols)}")
    print(f"Shape cuối        : {df_out.shape}")
    return df_out
