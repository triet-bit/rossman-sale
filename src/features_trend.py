"""
BƯỚC 4 – Trend Features (Ridge Regression theo từng tháng)
============================================================
Fit Ridge Regression trên lịch sử ~365 / 90 ngày để cung cấp:
  - trend_q/y_pred      : dự báo xu hướng tuyến tính
  - trend_q/y_slope     : hệ số góc (tốc độ thay đổi doanh thu)
  - trend_q_dow_coef    : tác động của ngày trong tuần lên xu hướng ngắn hạn
  - trend_q_promo_coef  : tác động của Promo lên xu hướng ngắn hạn
  - trend_yoy           : tăng trưởng Year-over-Year

Tất cả được tính TRƯỚC ngày đầu tháng (cutoff 30 ngày) → không có leakage.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model  import Ridge
from sklearn.preprocessing import StandardScaler


def add_trend_features_optimized(df):
    """
    Thêm trend features vào `df`.

    Parameters
    ----------
    df : DataFrame với cột 'Date', 'Store', 'Sales', 'DayOfWeek', 'Promo'.

    Returns
    -------
    DataFrame với DatetimeIndex và các cột `trend_*` được gắn thêm.
    """
    print("Chuẩn bị tính Trend Features (Chunking theo tháng)...")
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

    min_date      = df['Date'].min()
    df['day_number'] = (df['Date'] - min_date).dt.days.astype(float)
    df['YearMonth']  = df['Date'].dt.to_period('M')

    # Khởi tạo cột kết quả
    trend_cols = [
        'trend_q_pred', 'trend_y_pred',
        'trend_q_slope', 'trend_y_slope',
        'trend_q_dow_coef', 'trend_q_promo_coef',
        'trend_yoy',
    ]
    for col in trend_cols:
        df[col] = np.nan

    stores = df['Store'].unique()

    for i, store_id in enumerate(stores):
        if i % 100 == 0:
            print(f"  Đang xử lý Store {i}/{len(stores)}...")

        store_mask = df['Store'] == store_id
        store_df   = df[store_mask]
        months     = store_df['YearMonth'].unique()

        for ym in months:
            curr_month_mask = store_df['YearMonth'] == ym
            curr_month_df   = store_df[curr_month_mask]
            month_start     = curr_month_df['Date'].min()
            cutoff_end      = month_start - pd.Timedelta(days=30)

            # ── Year-over-Year ────────────────────────────────
            prev_m_start = month_start - pd.Timedelta(days=60)
            yoy_start    = month_start - pd.Timedelta(days=60 + 365)
            yoy_end      = month_start - pd.Timedelta(days=30 + 365)

            curr_sales = store_df[
                (store_df['Date'] >= prev_m_start) &
                (store_df['Date'] <  cutoff_end)   &
                (store_df['Sales'] > 0)
            ]['Sales'].mean()

            yoy_sales = store_df[
                (store_df['Date'] >= yoy_start) &
                (store_df['Date'] <  yoy_end)   &
                (store_df['Sales'] > 0)
            ]['Sales'].mean()

            yoy_val = (
                (curr_sales / yoy_sales - 1.0)
                if (pd.notna(curr_sales) and pd.notna(yoy_sales) and yoy_sales > 0)
                else np.nan
            )
            df.loc[curr_month_df.index, 'trend_yoy'] = yoy_val

            # ── Ridge Regression ─────────────────────────────
            for tf_name, days in [('q', 90), ('y', 365)]:
                cutoff_start = month_start - pd.Timedelta(days=days + 30)
                hist_mask    = (
                    (store_df['Date'] >= cutoff_start) &
                    (store_df['Date'] <  cutoff_end)   &
                    (store_df['Sales'] > 0)
                )
                hist = store_df[hist_mask].dropna(subset=['Sales', 'DayOfWeek', 'Promo'])

                if len(hist) < 10:
                    continue

                X_train = hist[['day_number', 'DayOfWeek', 'Promo']].astype(float).values
                y_train = hist['Sales'].values

                scaler   = StandardScaler()
                X_scaled = scaler.fit_transform(X_train)
                model    = Ridge(alpha=1.0)
                model.fit(X_scaled, y_train)

                X_test        = curr_month_df[['day_number', 'DayOfWeek', 'Promo']].astype(float).values
                X_test_scaled = scaler.transform(X_test)
                preds         = model.predict(X_test_scaled)

                df.loc[curr_month_df.index, f'trend_{tf_name}_pred']  = preds
                df.loc[curr_month_df.index, f'trend_{tf_name}_slope'] = model.coef_[0]

                if tf_name == 'q':
                    df.loc[curr_month_df.index, 'trend_q_dow_coef']   = model.coef_[1]
                    df.loc[curr_month_df.index, 'trend_q_promo_coef'] = model.coef_[2]

    df = df.drop(columns=['YearMonth']).set_index('Date')
    print("Hoàn tất tính Trend Features!")
    return df
