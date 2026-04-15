"""
Benchmark Framework for Market Crash Prediction
================================================
Provides standardised metrics for comparing classical random walker
against a quantum Szegedy walker (to be added later).

Crash definition: S&P 500 drawdown from rolling 252-day peak exceeds
a configurable threshold (default 15%).

Metrics:
  - Precision, Recall, F1
  - Lead Time (days before crash onset that alert fires)
  - AUC-ROC
  - Signal-to-noise ratio of the early-warning indicator
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve,
)


def identify_crash_periods(
    spx,
    drawdown_threshold: float = 0.15,
    peak_window: int = 252,
) -> pd.Series:
    """
    Label each trading day as crash (1) or normal (0).

    A crash period starts when drawdown from rolling peak exceeds
    the threshold and ends when the index recovers above 90% of peak.

    Parameters
    ----------
    spx             : index close prices (Series or DataFrame)
    drawdown_threshold : fraction (e.g. 0.15 = 15% decline)
    peak_window     : rolling window for computing peak

    Returns
    -------
    pd.Series of 0/1 labels aligned to spx.index
    """
    if isinstance(spx, pd.DataFrame):
        spx = spx.squeeze()
    rolling_peak = spx.rolling(peak_window, min_periods=1).max()
    drawdown = (spx - rolling_peak) / rolling_peak

    is_crash = (drawdown < -drawdown_threshold).astype(int)
    return is_crash


def generate_alerts(
    mixing_time: pd.Series,
    threshold_percentile: float = 25,
    direction: str = "below",
) -> pd.Series:
    """
    Generate binary alerts from the mixing time signal.

    For classical walker on PMFG:
      Hypothesis: mixing time DECREASES during crash
      → alert when mixing time drops BELOW a threshold

    Parameters
    ----------
    mixing_time          : mixing time series
    threshold_percentile : percentile of training data for threshold
    direction            : 'below' = alert when signal < threshold
                           'above' = alert when signal > threshold

    Returns
    -------
    pd.Series of 0/1 alerts
    """
    # Use first 30% of data as calibration window
    n_cal = max(int(len(mixing_time) * 0.30), 10)
    cal_data = mixing_time.iloc[:n_cal]
    threshold = np.percentile(cal_data.dropna(), threshold_percentile)

    if direction == "below":
        alerts = (mixing_time < threshold).astype(int)
    else:
        alerts = (mixing_time > threshold).astype(int)

    return alerts, threshold


def compute_lead_time(
    alerts: pd.Series,
    crash_labels: pd.Series,
) -> float:
    """
    Average number of trading days between first alert and crash onset.

    Positive = alert leads crash (good).
    Negative = alert lags crash (bad).
    """
    # Find crash onset dates (0→1 transitions)
    crash_starts = crash_labels.diff().fillna(0)
    crash_onset_dates = crash_starts[crash_starts == 1].index

    lead_times = []
    for onset in crash_onset_dates:
        # Look for alerts in the 60-day window before crash
        window_start = onset - pd.Timedelta(days=90)
        pre_crash_alerts = alerts.loc[window_start:onset]
        alert_dates = pre_crash_alerts[pre_crash_alerts == 1].index

        if len(alert_dates) > 0:
            first_alert = alert_dates[0]
            lead_days = (onset - first_alert).days
            lead_times.append(lead_days)

    return float(np.mean(lead_times)) if lead_times else float('nan')


def compute_benchmarks(
    results: pd.DataFrame,
    spx: pd.Series,
    mixing_time_col: str = "mixing_time_pmfg",
    model_name: str = "Classical Walker",
    direction: str = "below",
) -> dict:
    """
    Full benchmark suite for a crash prediction model.

    Parameters
    ----------
    results         : DataFrame with mixing time column
    spx             : S&P 500 close prices
    mixing_time_col : column name for the mixing time signal
    model_name      : name for reporting
    direction       : 'below' if mixing time decreases in crash

    Returns
    -------
    dict with all benchmark metrics
    """
    # Align data
    if isinstance(spx, pd.DataFrame):
        spx = spx.squeeze()
    common_idx = results.index.intersection(spx.index)
    if len(common_idx) == 0:
        # Use nearest reindex
        spx_aligned = spx.reindex(results.index, method='nearest')
    else:
        spx_aligned = spx.reindex(results.index, method='nearest')

    crash_labels = identify_crash_periods(spx_aligned)
    mixing_time = results[mixing_time_col]

    alerts, threshold = generate_alerts(mixing_time, direction=direction)

    # Align all series
    valid = crash_labels.notna() & alerts.notna() & mixing_time.notna()
    y_true = crash_labels[valid].values
    y_pred = alerts[valid].values

    # Handle edge case: no crashes or all crashes
    if len(np.unique(y_true)) < 2:
        return dict(
            model=model_name,
            precision=float('nan'),
            recall=float('nan'),
            f1=float('nan'),
            lead_time_days=float('nan'),
            auc_roc=float('nan'),
            threshold=float(threshold),
            n_alerts=int(y_pred.sum()),
            n_crash_days=int(y_true.sum()),
            n_total_days=len(y_true),
        )

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC — use negative mixing time if direction is 'below'
    if direction == "below":
        scores = -mixing_time[valid].values
    else:
        scores = mixing_time[valid].values
    auc = roc_auc_score(y_true, scores)

    lead_time = compute_lead_time(alerts, crash_labels)

    return dict(
        model=model_name,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        lead_time_days=float(lead_time),
        auc_roc=float(auc),
        threshold=float(threshold),
        n_alerts=int(y_pred.sum()),
        n_crash_days=int(y_true.sum()),
        n_total_days=len(y_true),
    )


def print_benchmark_table(benchmarks: list) -> str:
    """
    Pretty-print benchmark comparison table.
    """
    header = (
        f"{'Model':<25} {'Precision':>9} {'Recall':>8} "
        f"{'F1':>8} {'Lead(d)':>8} {'AUC':>8} "
        f"{'Alerts':>7} {'Crashes':>8}"
    )
    sep = "─" * len(header)

    lines = ["\n" + sep, "  BENCHMARK COMPARISON", sep, header, sep]

    for b in benchmarks:
        line = (
            f"{b['model']:<25} "
            f"{b['precision']:>9.3f} "
            f"{b['recall']:>8.3f} "
            f"{b['f1']:>8.3f} "
            f"{b.get('lead_time_days', float('nan')):>8.1f} "
            f"{b['auc_roc']:>8.3f} "
            f"{b['n_alerts']:>7d} "
            f"{b['n_crash_days']:>8d}"
        )
        lines.append(line)

    lines.append(sep)
    table = "\n".join(lines)
    print(table)
    return table
