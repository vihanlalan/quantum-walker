"""
================================================================================
  Classical Random Walker — Market Crash Detection
================================================================================
  Reads pre-cleaned Nifty 50 data (from scripts/clean_data.py) and runs:
    Rolling correlation → MP denoising → PMFG → Spectral analysis →
    Benchmarks → Publication plot

  Usage (Colab):
    # Cell 1 — install
    !pip install -q yfinance numpy pandas scipy networkx matplotlib scikit-learn

    # Cell 2 — clean data (run once)
    import os, sys
    sys.path.insert(0, "scripts")
    from clean_data import *
    download_and_clean()

    # Cell 3 — run walker
    %run src/qwalk/classical/walker.py

  Usage (local):
    python scripts/clean_data.py                # run once to download data
    python src/qwalk/classical/walker.py        # run the walker

  Key hypothesis:
    On the PMFG graph, mixing time τ DECREASES during crashes because
    all assets lock to a single factor → homogeneous graph → faster mixing.
================================================================================
"""

# ── Standard library ─────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time

# ── Path setup: add scripts/ to sys.path for shared imports ─────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))       # 3 levels up from walker.py
SCRIPTS_DIR  = os.path.join(PROJECT_ROOT, "scripts")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, "outputs")

sys.path.insert(0, SCRIPTS_DIR)

# ── Third-party ──────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from scipy import linalg
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score,
)

# ── Shared modules (from scripts/) ───────────────────────────────────────────
from clean_data import (
    empirical_correlation,
    mp_denoise,
    build_pmfg,
    pmfg_transition_matrix,
    CRISIS_EVENTS,
    CFG as DATA_CFG,
)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

os.makedirs(OUTPUT_DIR, exist_ok=True)

CFG = dict(
    T_WINDOW   = DATA_CFG["T_WINDOW"],       # 252
    STEP       = DATA_CFG["STEP"],            # 21
    OUTPUT_PNG = os.path.join(OUTPUT_DIR, "classical_mixing_times.png"),
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, "classical_results.csv"),
    BENCH_CSV  = os.path.join(OUTPUT_DIR, "classical_benchmarks.csv"),
)


# ══════════════════════════════════════════════════════════════════════════════
#  1. LOAD PRE-CLEANED DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_clean_data():
    """Load CSVs produced by scripts/clean_data.py."""
    log_ret_path = os.path.join(DATA_DIR, "clean_log_returns.csv")
    nifty_path   = os.path.join(DATA_DIR, "clean_nifty_index.csv")

    if not os.path.exists(log_ret_path):
        print(f"ERROR: {log_ret_path} not found.")
        print("       Run  python scripts/clean_data.py  first.")
        sys.exit(1)

    log_ret = pd.read_csv(log_ret_path, index_col=0, parse_dates=True)
    nifty   = pd.read_csv(nifty_path,   index_col=0, parse_dates=True)

    # Squeeze DataFrame → Series
    if isinstance(nifty, pd.DataFrame):
        nifty = nifty.squeeze()

    print(f"[Data] Loaded log returns: {log_ret.shape}")
    print(f"[Data] Loaded Nifty index: {nifty.shape}")
    print(f"[Data] Date range: {log_ret.index[0].date()} → "
          f"{log_ret.index[-1].date()}")

    return log_ret, nifty


# ══════════════════════════════════════════════════════════════════════════════
#  2. CLASSICAL SPECTRAL GAP  (walker-specific)
# ══════════════════════════════════════════════════════════════════════════════

def pmfg_spectral_gap(P: np.ndarray) -> dict:
    """
    Compute spectral gap and mixing time from the PMFG transition matrix.

    Uses the symmetric normalised form M = D^{-1/2} W D^{-1/2} for
    guaranteed real eigenvalues.

    Spectral gap:  Δ = 1 - |λ₂(M)/λ₁(M)|
    Mixing time:   τ = 1/Δ   (up to log factors)

    Hypothesis: τ DECREASES during crashes on PMFG
      • Normal: heterogeneous graph → slow mixing → high τ
      • Crisis: homogeneous graph   → fast mixing → low τ
    """
    N = P.shape[0]

    # Recover weighted adjacency from transition matrix
    A = P * (P.sum(axis=1))[:, None]
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)

    deg = A.sum(axis=1)
    deg[deg < 1e-12] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    M_sym = A * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]

    ev = np.sort(np.real(linalg.eigvalsh(M_sym)))[::-1]   # descending

    lam1 = float(ev[0])
    lam2 = float(ev[1]) if len(ev) > 1 else 0.0

    gap = 1.0 - abs(lam2 / lam1) if abs(lam1) > 1e-10 else 0.0
    tau = 1.0 / gap if gap > 1e-10 else 1e8

    return dict(
        lambda1_pmfg     = lam1,
        lambda2_pmfg     = lam2,
        spectral_gap_pmfg= gap,
        mixing_time_pmfg = tau,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  3. CORRELATION SPECTRAL METRICS  (reference signal)
# ══════════════════════════════════════════════════════════════════════════════

def correlation_spectral_metrics(C_clean: np.ndarray, N: int) -> dict:
    """
    Spectral gap from the cleaned correlation matrix (reference).
    Δ_C = λ₂(C)/N,  τ_C = N/λ₂(C)
    NOTE: τ_C INCREASES during crashes (opposite to PMFG τ).
    """
    ev = np.sort(np.real(linalg.eigvalsh(C_clean)))[::-1]
    lam1 = float(ev[0])
    lam2 = float(ev[1])
    return dict(
        lambda1_C = lam1,
        lambda2_C = lam2,
        delta_C   = lam2 / N,
        tau_C     = N / lam2 if lam2 > 1e-10 else 1e8,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  4. SINGLE-WINDOW PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

def _process_window(args):
    """
    Process one rolling window: correlation → MP → PMFG → spectral.
    Top-level function (pickleable for potential multiprocessing).
    """
    idx, t_start, t_end, R, Q, date_end, date_start, n_windows = args
    wt0 = time.time()

    # 1. Correlation
    C_emp = empirical_correlation(R)

    # 2. MP denoising
    C_clean, mp = mp_denoise(C_emp, Q)

    # 3. PMFG
    pt0 = time.time()
    G = build_pmfg(C_clean, verbose=False)
    pmfg_sec = time.time() - pt0

    # 4. Transition matrix + classical spectral gap
    P, pinfo = pmfg_transition_matrix(G)
    spec = pmfg_spectral_gap(P)

    # 5. Correlation spectral (reference)
    N = R.shape[1]
    cmet = correlation_spectral_metrics(C_clean, N)

    return dict(
        _idx              = idx,
        date              = date_end,
        window_start      = date_start,
        # PMFG classical walker (primary)
        mixing_time_pmfg  = spec["mixing_time_pmfg"],
        spectral_gap_pmfg = spec["spectral_gap_pmfg"],
        lambda1_pmfg      = spec["lambda1_pmfg"],
        lambda2_pmfg      = spec["lambda2_pmfg"],
        # Correlation (reference)
        tau_C             = cmet["tau_C"],
        delta_C           = cmet["delta_C"],
        lambda1_C         = cmet["lambda1_C"],
        lambda2_C         = cmet["lambda2_C"],
        # PMFG stats
        pmfg_n_edges      = pinfo["n_edges"],
        pmfg_mean_degree  = pinfo["mean_degree"],
        pmfg_density      = pinfo["density"],
        # MP diagnostics
        lambda_plus       = mp["lambda_plus"],
        n_noise           = mp["n_noise"],
        n_signal          = mp["n_signal"],
        noise_fraction    = mp["noise_fraction"],
        min_ev_clean      = mp["min_ev_clean"],
        # Timing
        pmfg_time_s       = pmfg_sec,
        window_time_s     = time.time() - wt0,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  5. ROLLING ANALYSIS  (sequential — safe for Colab)
# ══════════════════════════════════════════════════════════════════════════════

def rolling_analysis(log_ret, T_window=252, step=21):
    """
    Execute the full pipeline on every rolling window.
    Sequential processing for stability in Colab / notebook environments.
    """
    T_total, N = log_ret.shape
    dates = log_ret.index
    Q  = T_window / N
    lp = (1.0 + np.sqrt(1.0 / Q)) ** 2
    n_windows = (T_total - T_window) // step + 1

    print(f"\n{'─'*62}")
    print(f"  Rolling Window Analysis — Classical Walker")
    print(f"  N={N}  T_window={T_window}  Q={Q:.4f}  λ+={lp:.4f}")
    print(f"  Total windows: {n_windows} (step={step} days)")
    print(f"{'─'*62}")

    # Build window argument list
    window_args = []
    for i, t_end in enumerate(range(T_window, T_total, step)):
        t_start = t_end - T_window
        R = log_ret.iloc[t_start:t_end].values
        window_args.append((i, t_start, t_end, R, Q,
                            dates[t_end - 1], dates[t_start], n_windows))

    # First window — verbose (sanity check)
    print(f"  [  1/{n_windows}] {dates[window_args[0][1]].date()} → "
          f"{dates[window_args[0][2]-1].date()}  (verbose)")
    ft0 = time.time()
    C0  = empirical_correlation(window_args[0][3])
    C0c, _ = mp_denoise(C0, Q)
    _   = build_pmfg(C0c, verbose=True)
    first = _process_window(window_args[0])
    print(f"    First window: {time.time()-ft0:.1f}s")

    # Remaining windows — sequential
    t_all = time.time()
    records = [first]
    print(f"\n  Processing remaining {len(window_args)-1} windows …")

    for args in window_args[1:]:
        records.append(_process_window(args))
        done = len(records)
        if done % 10 == 0 or done == n_windows:
            elapsed = time.time() - t_all
            eta = elapsed / (done - 1) * (n_windows - done) if done > 1 else 0
            print(f"  [{done:3d}/{n_windows}] "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    # Sort chronologically and build DataFrame
    records.sort(key=lambda r: r["_idx"])
    for r in records:
        del r["_idx"]

    df = pd.DataFrame(records).set_index("date")

    total = time.time() - t_all + first["window_time_s"]
    pmfg_total = df["pmfg_time_s"].sum()
    print(f"\n  Complete — {len(df)} windows.")
    print(f"  PMFG total: {pmfg_total:.1f}s (avg {pmfg_total/len(df):.1f}s)")
    print(f"  Wall-clock: {total:.1f}s ({total/60:.1f} min)\n")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  6. MATHEMATICAL VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def verify(results, N):
    """Verify pipeline invariants and test crash hypothesis."""
    sep = "═" * 62
    print(f"\n{sep}")
    print("  MATHEMATICAL VERIFICATION SUITE")
    print(sep)
    checks = 0

    # 1. PSD
    min_ev = results["min_ev_clean"].min()
    ok = min_ev >= -1e-6
    print(f"\n[1] C_clean PSD: min_ev={min_ev:.3e} → {'✓' if ok else '✗'}")
    checks += ok

    # 2. PMFG planarity
    exp = 3 * (N - 2)
    ok = (results["pmfg_n_edges"] == exp).all()
    print(f"[2] PMFG edges = {exp}: {'✓' if ok else '✗'}")
    checks += ok

    # 3. Finite mixing time
    ok = (results["mixing_time_pmfg"] < 1e7).all()
    print(f"[3] Mixing time finite: {'✓' if ok else '✗'}")
    checks += ok

    # 4. Noise fraction
    nf = results["noise_fraction"].mean()
    print(f"[4] MP noise fraction: {nf*100:.1f}%")
    checks += 1

    # 5. COVID hypothesis
    pre  = results.loc["2019-06-01":"2020-01-31", "mixing_time_pmfg"]
    cris = results.loc["2020-02-20":"2020-04-30", "mixing_time_pmfg"]
    if len(cris) > 0 and len(pre) > 0:
        ok = cris.median() < pre.median()
        print(f"\n[5] COVID hypothesis: τ drops in crisis")
        print(f"    Pre-COVID median:  {pre.median():.2f}")
        print(f"    COVID median:      {cris.median():.2f}")
        print(f"    → {'✓ CONFIRMED' if ok else '✗ NOT CONFIRMED'}")
        checks += ok

    # 6. IL&FS/NBFC crisis
    pre18  = results.loc["2018-01-01":"2018-06-30", "mixing_time_pmfg"]
    cris18 = results.loc["2018-09-01":"2019-01-31", "mixing_time_pmfg"]
    if len(cris18) > 0 and len(pre18) > 0:
        ok = cris18.median() < pre18.median()
        print(f"\n[6] IL&FS/NBFC hypothesis: τ drops in crisis")
        print(f"    Pre-crisis median: {pre18.median():.2f}")
        print(f"    Crisis median:     {cris18.median():.2f}")
        print(f"    → {'✓ CONFIRMED' if ok else '✗ NOT CONFIRMED'}")
        checks += ok

    print(f"\n{'─'*62}")
    print(f"  {checks} checks completed")
    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  7. BENCHMARK FRAMEWORK  (inlined from benchmark.py)
# ══════════════════════════════════════════════════════════════════════════════

def identify_crash_periods(
    spx,
    drawdown_threshold: float = 0.15,
    peak_window: int = 252,
) -> pd.Series:
    """
    Label each trading day as crash (1) or normal (0).
    A crash period starts when drawdown from rolling peak exceeds
    the threshold and ends when the index recovers.
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
) -> tuple:
    """
    Generate binary alerts from the mixing time signal.

    For classical walker on PMFG:
      Hypothesis: mixing time DECREASES during crash
      → alert when mixing time drops BELOW a threshold
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
    crash_starts = crash_labels.diff().fillna(0)
    crash_onset_dates = crash_starts[crash_starts == 1].index

    lead_times = []
    for onset in crash_onset_dates:
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
    spx             : index close prices
    mixing_time_col : column name for the mixing time signal
    model_name      : name for reporting
    direction       : 'below' if mixing time decreases in crash
    """
    if isinstance(spx, pd.DataFrame):
        spx = spx.squeeze()

    spx_aligned = spx.reindex(results.index, method='nearest')
    crash_labels = identify_crash_periods(spx_aligned)
    mixing_time = results[mixing_time_col]

    alerts, threshold = generate_alerts(mixing_time, direction=direction)

    valid = crash_labels.notna() & alerts.notna() & mixing_time.notna()
    y_true = crash_labels[valid].values
    y_pred = alerts[valid].values

    # Handle edge case: no crashes or all crashes
    if len(np.unique(y_true)) < 2:
        return dict(
            model=model_name,
            precision=float('nan'), recall=float('nan'),
            f1=float('nan'), lead_time_days=float('nan'),
            auc_roc=float('nan'), threshold=float(threshold),
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
    """Pretty-print benchmark comparison table."""
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


# ══════════════════════════════════════════════════════════════════════════════
#  8. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(results, nifty_idx, N, bench, output_path):
    """Five-panel publication figure with dark theme."""
    BG, PANEL, GRID     = "#0a0e14", "#111720", "#1e2a36"
    SPX_C, TAU_C, GAP_C = "#60a5fa", "#f87171", "#34d399"
    REF_C, EVT_C, SHADE = "#a78bfa", "#fbbf24", "#f87171"
    TEXT, SUB, MA_C      = "#e2e8f0", "#94a3b8", "#f1f5f9"
    TIME_C               = "#fb923c"

    if isinstance(nifty_idx, pd.DataFrame):
        nifty_idx = nifty_idx.squeeze()
    idx_a = nifty_idx.reindex(results.index, method="nearest")
    crash_labels = identify_crash_periods(idx_a)

    fig = plt.figure(figsize=(22, 20), facecolor=BG)
    fig.suptitle(
        "Classical Random Walker — Market Crash Detection\n"
        "Marchenko-Pastur + PMFG · "
        f"Nifty 50 ({N} stocks, 2010–2025)",
        fontsize=13, fontweight="bold", color=TEXT, y=0.99,
        fontfamily="monospace",
    )

    gs = gridspec.GridSpec(5, 1, hspace=0.08,
                           top=0.96, bottom=0.05, left=0.07, right=0.76)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    # Format all panels
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=SUB, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(axis="y", color=GRID, lw=0.5, ls="--", alpha=0.7)
        ax.set_xlim(results.index[0], results.index[-1])
        # Shade crash periods
        in_crash = False
        for dt, val in crash_labels.items():
            if val == 1 and not in_crash:
                cs = dt; in_crash = True
            elif val == 0 and in_crash:
                ax.axvspan(cs, dt, alpha=0.10, color=SHADE, zorder=0)
                in_crash = False
        if in_crash:
            ax.axvspan(cs, crash_labels.index[-1], alpha=0.10,
                       color=SHADE, zorder=0)

    # Crisis event lines
    for dt_str in CRISIS_EVENTS:
        dt = pd.Timestamp(dt_str)
        if results.index[0] <= dt <= results.index[-1]:
            for ax in axes:
                ax.axvline(dt, color=EVT_C, lw=0.6, ls=":", alpha=0.7)

    # Panel 0 — Nifty 50 index
    ax = axes[0]
    ax.plot(idx_a.index, idx_a.values, color=SPX_C, lw=1.4, alpha=0.9)
    ax.fill_between(idx_a.index, idx_a.values, idx_a.min(),
                    alpha=0.10, color=SPX_C)
    ax.set_ylabel("Index Level", color=TEXT, fontsize=9)
    ax.set_title("Nifty 50 Index", color=SUB, fontsize=8.5, loc="left", pad=3)

    # Panel 1 — PMFG mixing time (PRIMARY)
    ax = axes[1]
    tau = results["mixing_time_pmfg"].clip(
        upper=results["mixing_time_pmfg"].quantile(0.995))
    ax.plot(tau.index, tau.values, color=TAU_C, lw=1.3, alpha=0.9,
            label="τ_PMFG (classical)")
    ma = tau.rolling(5).mean()
    ax.plot(ma.index, ma.values, color=MA_C, lw=0.7, alpha=0.5, ls="--",
            label="5-window MA")
    ax.set_ylabel("Mixing Time τ", color=TEXT, fontsize=9)
    ax.set_title("PMFG Mixing Time τ  [↓ = crash signal]",
                 color=SUB, fontsize=8.5, loc="left", pad=3)
    ax.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, fontsize=7.5)

    # Panel 2 — Spectral gap
    ax = axes[2]
    gap = results["spectral_gap_pmfg"]
    ax.plot(gap.index, gap.values, color=GAP_C, lw=1.3, alpha=0.9,
            label="Δ_PMFG")
    ax.set_ylabel("Spectral Gap Δ", color=TEXT, fontsize=9)
    ax.set_title("PMFG Spectral Gap Δ = 1-|λ₂(P)|  [↑ = crash]",
                 color=SUB, fontsize=8.5, loc="left", pad=3)
    ax.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, fontsize=7.5)

    # Panel 3 — τ_C reference
    ax = axes[3]
    tc = results["tau_C"].clip(upper=results["tau_C"].quantile(0.995))
    ax.plot(tc.index, tc.values, color=REF_C, lw=1.3, alpha=0.9,
            label="τ_C (correlation)")
    ax.set_ylabel("τ_C (ref)", color=TEXT, fontsize=9)
    ax.set_title("Correlation Mixing Time τ_C  [↑ = crash]",
                 color=SUB, fontsize=8.5, loc="left", pad=3)
    ax.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, fontsize=7.5)

    # Panel 4 — PMFG computation time
    ax = axes[4]
    tp = results["pmfg_time_s"]
    ax.bar(tp.index, tp.values, width=15, color=TIME_C, alpha=0.7)
    ax.set_ylabel("PMFG Time (s)", color=TEXT, fontsize=9)
    ax.set_title(f"PMFG Time/Window  [avg={tp.mean():.1f}s, "
                 f"total={tp.sum():.0f}s]",
                 color=SUB, fontsize=8.5, loc="left", pad=3)

    # X-axis formatting
    for i, ax in enumerate(axes):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right",
                 color=SUB, fontsize=7.5)
        if i < 4:
            ax.set_xticklabels([])

    # Sidebar
    sidebar = (
        "  ┌─────────────────────────────┐\n"
        "  │   CLASSICAL WALKER          │\n"
        "  ├─────────────────────────────┤\n"
        f"  │  Assets (N):       {N:<8d} │\n"
        f"  │  Window (T):       {CFG['T_WINDOW']:<8d} │\n"
        f"  │  Step:             {CFG['STEP']:<8d} │\n"
        f"  │  Windows:          {len(results):<8d} │\n"
        "  ├─────────────────────────────┤\n"
        "  │   BENCHMARKS                │\n"
        "  ├─────────────────────────────┤\n"
        f"  │  Precision:  {bench.get('precision',0):.3f}         │\n"
        f"  │  Recall:     {bench.get('recall',0):.3f}         │\n"
        f"  │  F1 Score:   {bench.get('f1',0):.3f}         │\n"
        f"  │  AUC-ROC:    {bench.get('auc_roc',0):.3f}         │\n"
        f"  │  Lead Time:  {bench.get('lead_time_days',0):.0f}d           │\n"
        "  ├─────────────────────────────┤\n"
        "  │   TIMING                    │\n"
        "  ├─────────────────────────────┤\n"
        f"  │  Avg PMFG:   {tp.mean():.1f}s          │\n"
        f"  │  Total PMFG: {tp.sum():.0f}s         │\n"
        "  └─────────────────────────────┘"
    )
    fig.text(0.78, 0.50, sidebar, fontsize=7.5, color=TEXT,
             fontfamily="monospace", va="center", ha="left",
             bbox=dict(boxstyle="round,pad=0.5", fc=PANEL, ec=GRID, alpha=0.95))

    crisis_patch = mpatches.Patch(color=SHADE, alpha=0.35,
                                  label="Crash Periods (>15% drawdown)")
    fig.legend(handles=[crisis_patch], loc="lower center",
               bbox_to_anchor=(0.42, 0.002), facecolor=PANEL, edgecolor=GRID,
               labelcolor=TEXT, fontsize=8)

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.show()
    print(f"[Plot] Saved → {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  9. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    sep = "═" * 62
    print(f"\n{sep}")
    print("  Classical Random Walker — Nifty 50")
    print("  Pipeline: Clean Data → MP Denoise → PMFG → Spectral Analysis")
    print(sep)

    total_t0 = time.time()

    # 1. Load clean data
    log_ret, nifty_idx = load_clean_data()
    N = log_ret.shape[1]
    Q = CFG["T_WINDOW"] / N
    print(f"\n[Config] N={N}, Q={Q:.4f}, λ+={(1+np.sqrt(1/Q))**2:.4f}")

    # 2. Rolling pipeline
    results = rolling_analysis(log_ret, CFG["T_WINDOW"], CFG["STEP"])

    # 3. Verification
    verify(results, N)

    # 4. Benchmarks
    print("[Benchmarks] Computing crash prediction metrics …")

    bench_pmfg = compute_benchmarks(
        results, nifty_idx,
        mixing_time_col="mixing_time_pmfg",
        model_name="Classical Walker (PMFG)",
        direction="below",
    )

    bench_corr = compute_benchmarks(
        results, nifty_idx,
        mixing_time_col="tau_C",
        model_name="Correlation τ_C (ref)",
        direction="above",
    )

    bench_szegedy = dict(
        model="Quantum Szegedy Walker",
        precision=float('nan'), recall=float('nan'),
        f1=float('nan'), lead_time_days=float('nan'),
        auc_roc=float('nan'), threshold=float('nan'),
        n_alerts=0, n_crash_days=0, n_total_days=0,
    )

    all_bench = [bench_pmfg, bench_corr, bench_szegedy]
    print_benchmark_table(all_bench)

    # 5. Save outputs
    results.to_csv(CFG["OUTPUT_CSV"])
    print(f"\n[CSV] Results    → {CFG['OUTPUT_CSV']}")

    pd.DataFrame(all_bench).to_csv(CFG["BENCH_CSV"], index=False)
    print(f"[CSV] Benchmarks → {CFG['BENCH_CSV']}")

    # 6. Plot
    plot_results(results, nifty_idx, N, bench_pmfg, CFG["OUTPUT_PNG"])

    total = time.time() - total_t0
    print(f"\n[Done] Classical walker complete in {total:.0f}s "
          f"({total/60:.1f} min)\n")

    return results


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results = main()
