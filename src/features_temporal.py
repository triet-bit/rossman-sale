"""
BƯỚC 3 – Temporal / Event Features
=====================================
Tính các đặc trưng liên quan đến thời gian: lịch, promo cycle, holiday,
refurbishment, competition start, và holiday counts.
Tất cả đều vectorized (không dùng vòng lặp for theo hàng).
"""
import numpy as np
import pandas as pd


# ============================================================
# HELPER FUNCTIONS (100% VECTORIZED)
# ============================================================

def _calc_days_since(df, col, event_val):
    """Số ngày kể từ lần cuối sự kiện xảy ra (forward-fill theo Store)."""
    event_dates  = df['Date'].where(df[col] == event_val)
    last_event   = event_dates.groupby(df['Store']).ffill()
    return (df['Date'] - last_event).dt.days


def _calc_days_until(df, col, event_val):
    """Số ngày đến lần tiếp theo sự kiện xảy ra (backward-fill theo Store)."""
    event_dates  = df['Date'].where(df[col] == event_val)
    next_event   = event_dates.groupby(df['Store']).bfill()
    return (next_event - df['Date']).dt.days


def _detect_refurbishment_vectorized(df, min_closure_days=7):
    """
    Phát hiện đợt đóng cửa dài (refurbishment) mà không dùng vòng for.
    Trả về mảng nhị phân: 1 = ngày bắt đầu đợt refurb, 0 = không phải.
    """
    closed = df[df['Open'] == 0][['Store', 'Date']].copy()
    if closed.empty:
        return np.zeros(len(df), dtype=int)

    closed = closed.sort_values(['Store', 'Date'])
    is_new_group = (
        (closed['Store'] != closed['Store'].shift()) |
        (closed['Date'].diff().dt.days > 2)
    )
    group_id    = is_new_group.cumsum()
    group_stats = closed.groupby(group_id).agg(
        Store      = ('Store', 'first'),
        Start_Date = ('Date',  'min'),
        Count      = ('Date',  'count'),
    )
    refurb_starts = group_stats[group_stats['Count'] >= min_closure_days]

    is_refurb = np.zeros(len(df), dtype=int)
    if not refurb_starts.empty:
        refurb_idx = pd.MultiIndex.from_frame(refurb_starts[['Store', 'Start_Date']])
        df_idx     = pd.MultiIndex.from_arrays([df['Store'], df['Date']])
        is_refurb[df_idx.isin(refurb_idx)] = 1
    return is_refurb


# ============================================================
# HÀM CHÍNH BƯỚC 3
# ============================================================

def add_temporal_features(df):
    """
    Thêm các đặc trưng thời gian / sự kiện vào `df`.

    Parameters
    ----------
    df : DataFrame với cột 'Date', 'Store', 'Open', 'Promo', 'Promo2',
         'SchoolHoliday', 'StateHoliday', 'CompetitionOpenSinceYear/Month'.

    Returns
    -------
    DataFrame với DatetimeIndex và các cột temporal được gắn thêm.
    """
    print("Chuẩn bị dữ liệu temporal...")
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

    # ── 1. BASIC TEMPORAL ────────────────────────────────────
    df['day_of_week']  = df['Date'].dt.dayofweek + 1
    df['day_of_month'] = df['Date'].dt.day
    df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)
    df['month']        = df['Date'].dt.month
    df['year']         = df['Date'].dt.year
    df['day_of_year']  = df['Date'].dt.dayofyear

    # ── 2. PROMO CYCLE (mỗi 14 ngày) ────────────────────────
    min_date = df['Date'].min()
    df['day_in_promo_cycle'] = ((df['Date'] - min_date).dt.days % 14).astype(int)

    print("  Tính Promo cycles...")
    df['days_since_promo'] = _calc_days_since(df, 'Promo', 1)
    df['days_until_promo'] = _calc_days_until(df, 'Promo', 1)

    # ── 3. PROMO2 CYCLE ──────────────────────────────────────
    if 'Promo2' in df.columns:
        df['day_in_promo2_cycle'] = ((df['Date'] - min_date).dt.days % 91).astype(int)
        df['days_since_promo2']   = _calc_days_since(df, 'Promo2', 1)

    # ── 4. SUMMER HOLIDAY ────────────────────────────────────
    print("  Tính Summer Holidays...")
    df['days_since_school_holiday'] = _calc_days_since(df, 'SchoolHoliday', 1)
    df['days_until_school_holiday'] = _calc_days_until(df, 'SchoolHoliday', 1)

    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    summer_start    = pd.to_datetime(df['year'].astype(str) + '-06-01')
    days_into       = (df['Date'] - summer_start).dt.days
    df['days_into_summer'] = np.where(df['month'] >= 6, days_into, 0)
    df['days_into_summer'] = df['days_into_summer'].clip(lower=0)

    # ── 5. STORE REFURBISHMENT ───────────────────────────────
    print("  Tìm Refurbishments...")
    if 'Open' in df.columns:
        df['is_refurb_closure'] = _detect_refurbishment_vectorized(df)
        df['days_since_refurb'] = _calc_days_since(df, 'is_refurb_closure', 1)
        df['days_until_refurb'] = _calc_days_until(df, 'is_refurb_closure', 1)

    # ── 6. COMPETITION START ─────────────────────────────────
    print("  Tính Competition Days...")
    if 'CompetitionOpenSinceYear' in df.columns and 'CompetitionOpenSinceMonth' in df.columns:
        comp_year  = df['CompetitionOpenSinceYear'].fillna(1900).astype(int)
        comp_month = df['CompetitionOpenSinceMonth'].fillna(1).astype(int)
        comp_date  = pd.to_datetime(
            comp_year.astype(str) + '-' + comp_month.astype(str).str.zfill(2) + '-01',
            errors='coerce',
        )
        days_since_comp = (df['Date'] - comp_date).dt.days
        df['days_since_competition'] = np.where(
            df['CompetitionOpenSinceYear'].isna(),
            -1,
            days_since_comp.clip(lower=0),
        )

    # ── 7 & 8. HOLIDAY COUNTS ────────────────────────────────
    print("  Tính Holiday Counts...")
    df['state_holiday_flag'] = df['StateHoliday'].astype(str).ne('0').astype(int)

    for col, flag_col in [('holidays', 'state_holiday_flag'), ('school_holidays', 'SchoolHoliday')]:
        if flag_col in df.columns:
            roll = df.groupby('Store')[flag_col]
            df[f'{col}_this_week'] = roll.transform(
                lambda x: x.rolling(7, min_periods=1, center=True).sum()
            )
            df[f'{col}_last_week'] = roll.transform(
                lambda x: x.shift(7).rolling(7, min_periods=1).sum()
            )
            df[f'{col}_next_week'] = roll.transform(
                lambda x: x.shift(-7).rolling(7, min_periods=1).sum()
            )

    df = df.set_index('Date')

    temporal_cols = [c for c in df.columns if any(c.startswith(p) for p in [
        'day_', 'week_', 'month', 'year', 'days_', 'is_',
        'holidays_', 'school_holidays_', 'competition_',
    ])]
    print(f"Xong! Đã thêm {len(temporal_cols)} temporal features.")
    return df
