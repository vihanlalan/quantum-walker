"""
================================================================================
Phase 1 — Proof of Concept: Market Crash Detection via Classical Random Walk
           Mixing Times on Nifty 50 (Indian Market)
================================================================================
SINGLE-FILE FOR GOOGLE COLAB — paste into one cell and run.

Target  : Nifty 50, 2010–2025
Pipeline: yfinance → NaN imputation → Log-returns → Rolling correlation →
          MP denoising → PMFG graph → Classical random walker →
          Spectral gap Δ → Mixing time τ → Crash detection benchmarks

HYPOTHESIS
──────────
On the PMFG-filtered graph:
  • Normal markets → heterogeneous graph → walker mixes SLOWER (high τ)
  • Crisis         → homogeneous graph   → walker mixes FASTER  (low τ)
  ⇒ Mixing time τ DECREASES during crashes — early warning signal.
================================================================================
"""

# ── Install dependencies ────────────────────────────────────────────────────
import subprocess, sys
for pkg in ["yfinance", "scikit-learn", "networkx"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# ── Standard library ────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
import os
import time
import builtins

os.environ["PYTHONUNBUFFERED"] = "1"
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)

# ── Third-party ─────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from scipy import linalg
import networkx as nx
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score,
)


###############################################################################
#                                                                             #
#  MODULE 1: PMFG  (Planar Maximally Filtered Graph)                         #
#                                                                             #
###############################################################################

def build_pmfg(corr_matrix: np.ndarray, verbose: bool = True) -> nx.Graph:
    """
    Greedy PMFG: add edges by descending |corr| if graph stays planar.
    Stops at 3(N-2) edges.
    """
    N = corr_matrix.shape[0]
    max_edges = 3 * (N - 2)

    rows, cols = np.triu_indices(N, k=1)
    abs_corr = np.abs(corr_matrix[rows, cols])
    signed_corr = corr_matrix[rows, cols]

    order = np.argsort(-abs_corr)
    rows, cols = rows[order], cols[order]
    abs_corr, signed_corr = abs_corr[order], signed_corr[order]
    n_candidates = len(rows)

    if verbose:
        print(f"  [PMFG] N={N}, max_edges={max_edges}, "
              f"candidate_edges={n_candidates}")

    G = nx.Graph()
    G.add_nodes_from(range(N))

    t0 = time.time()
    added = checked = 0

    for idx in range(n_candidates):
        if added >= max_edges:
            break
        i, j = int(rows[idx]), int(cols[idx])
        ac, sc = float(abs_corr[idx]), float(signed_corr[idx])
        checked += 1

        G.add_edge(i, j, weight=ac, corr=sc)
        if nx.check_planarity(G)[0]:
            added += 1
            if verbose and added % 50 == 0:
                print(f"    edges: {added}/{max_edges} "
                      f"(checked {checked}, {time.time()-t0:.1f}s)")
        else:
            G.remove_edge(i, j)

    if verbose:
        print(f"  [PMFG] Complete: {added} edges in {time.time()-t0:.1f}s "
              f"(checked {checked} candidates)")
    return G


def pmfg_transition_matrix(G: nx.Graph) -> tuple:
    """Row-stochastic transition matrix from PMFG."""
    N = G.number_of_nodes()
    A = np.zeros((N, N))
    for i, j, d in G.edges(data=True):
        w = d.get('weight', 1.0)
        A[i, j] = A[j, i] = w

    deg = A.sum(axis=1)
    deg[deg < 1e-12] = 1e-12
    P = A / deg[:, None]

    info = dict(
        n_nodes=N, n_edges=G.number_of_edges(),
        mean_degree=2 * G.number_of_edges() / N,
        min_degree=min(dict(G.degree()).values()) if N > 0 else 0,
        max_degree=max(dict(G.degree()).values()) if N > 0 else 0,
        density=nx.density(G),
    )
    return P, info


def pmfg_spectral_gap(P: np.ndarray) -> dict:
    """Spectral gap Δ = 1 - |λ₂(P)| and mixing time τ = 1/Δ."""
    N = P.shape[0]
    A = P * (P.sum(axis=1))[:, None]
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)

    deg = A.sum(axis=1)
    deg[deg < 1e-12] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    M_sym = A * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]

    ev = np.sort(np.real(linalg.eigvalsh(M_sym)))[::-1]
    lam1, lam2 = float(ev[0]), float(ev[1]) if len(ev) > 1 else 0.0

    gap = 1.0 - abs(lam2 / lam1) if abs(lam1) > 1e-10 else 0.0
    tau = 1.0 / gap if gap > 1e-10 else 1e8

    return dict(lambda1_pmfg=lam1, lambda2_pmfg=lam2,
                spectral_gap_pmfg=gap, mixing_time_pmfg=tau)


###############################################################################
#                                                                             #
#  MODULE 2: BENCHMARKS                                                       #
#                                                                             #
###############################################################################

def identify_crash_periods(idx_series, drawdown_threshold=0.15, peak_window=252):
    # Ensure we have a Series, not a DataFrame
    if isinstance(idx_series, pd.DataFrame):
        idx_series = idx_series.squeeze()
    rolling_peak = idx_series.rolling(peak_window, min_periods=1).max()
    drawdown = (idx_series - rolling_peak) / rolling_peak
    return (drawdown < -drawdown_threshold).astype(int)


def generate_alerts(mixing_time, threshold_percentile=25, direction="below"):
    n_cal = max(int(len(mixing_time) * 0.30), 10)
    thr = np.percentile(mixing_time.iloc[:n_cal].dropna(), threshold_percentile)
    if direction == "below":
        alerts = (mixing_time < thr).astype(int)
    else:
        alerts = (mixing_time > thr).astype(int)
    return alerts, thr


def compute_lead_time(alerts, crash_labels):
    starts = crash_labels.diff().fillna(0)
    onset_dates = starts[starts == 1].index
    leads = []
    for onset in onset_dates:
        w_start = onset - pd.Timedelta(days=90)
        pre = alerts.loc[w_start:onset]
        fired = pre[pre == 1].index
        if len(fired) > 0:
            leads.append((onset - fired[0]).days)
    return float(np.mean(leads)) if leads else float('nan')


def compute_benchmarks(results, idx_series, col="mixing_time_pmfg",
                       name="Classical Walker", direction="below"):
    # Ensure we have a Series, not a DataFrame
    if isinstance(idx_series, pd.DataFrame):
        idx_series = idx_series.squeeze()
    idx_a = idx_series.reindex(results.index, method='nearest')
    crash = identify_crash_periods(idx_a)
    mt = results[col]
    alerts, thr = generate_alerts(mt, direction=direction)

    valid = crash.notna() & alerts.notna() & mt.notna()
    y_true, y_pred = crash[valid].values, alerts[valid].values

    if len(np.unique(y_true)) < 2:
        return dict(model=name, precision=float('nan'), recall=float('nan'),
                    f1=float('nan'), lead_time_days=float('nan'),
                    auc_roc=float('nan'), threshold=float(thr),
                    n_alerts=int(y_pred.sum()), n_crash_days=int(y_true.sum()),
                    n_total_days=len(y_true))

    scores = -mt[valid].values if direction == "below" else mt[valid].values
    lead = compute_lead_time(alerts, crash)

    return dict(
        model=name,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        lead_time_days=float(lead),
        auc_roc=float(roc_auc_score(y_true, scores)),
        threshold=float(thr),
        n_alerts=int(y_pred.sum()),
        n_crash_days=int(y_true.sum()),
        n_total_days=len(y_true),
    )


def print_benchmark_table(benchmarks):
    hdr = (f"{'Model':<25} {'Prec':>6} {'Rec':>6} {'F1':>6} "
           f"{'Lead':>6} {'AUC':>6} {'Alerts':>7} {'Crash':>6}")
    sep = "─" * len(hdr)
    lines = ["\n" + sep, "  BENCHMARK COMPARISON", sep, hdr, sep]
    for b in benchmarks:
        lines.append(
            f"{b['model']:<25} {b['precision']:>6.3f} {b['recall']:>6.3f} "
            f"{b['f1']:>6.3f} {b.get('lead_time_days',float('nan')):>6.0f} "
            f"{b['auc_roc']:>6.3f} {b['n_alerts']:>7d} "
            f"{b['n_crash_days']:>6d}")
    lines.append(sep)
    table = "\n".join(lines)
    print(table)
    return table


###############################################################################
#                                                                             #
#  MODULE 3: PIPELINE                                                         #
#                                                                             #
###############################################################################

# ─── Nifty 50 tickers (Yahoo Finance: .NS suffix) ───────────────────────────
NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "NESTLEIND.NS", "TRENT.NS", "M&M.NS", "POWERGRID.NS", "NTPC.NS",
    "TATASTEEL.NS", "TECHM.NS", "JSWSTEEL.NS", "INDUSINDBK.NS", "HINDALCO.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "COALINDIA.NS", "ONGC.NS", "BPCL.NS",
    "GRASIM.NS", "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS", "EICHERMOT.NS",
    "BAJAJ-AUTO.NS", "DIVISLAB.NS", "HEROMOTOCO.NS", "SBILIFE.NS",
    "BAJAJFINSV.NS", "BRITANNIA.NS", "TATACONSUM.NS", "HDFCLIFE.NS",
    "LTIM.NS", "SHRIRAMFIN.NS",
]

OUTPUT_DIR = "/content/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CFG = dict(
    T_WINDOW   = 252,
    STEP       = 21,
    START      = "2010-01-01",       # Nifty 50 data richer from 2010+
    END        = "2025-03-31",
    RANDOM_SEED = 42,
    TICKERS    = NIFTY50_TICKERS,
    OUTPUT_PNG = os.path.join(OUTPUT_DIR, "nifty50_mixing_times.png"),
    OUTPUT_CSV = os.path.join(OUTPUT_DIR, "nifty50_results.csv"),
    BENCH_CSV  = os.path.join(OUTPUT_DIR, "nifty50_benchmarks.csv"),
)

np.random.seed(CFG["RANDOM_SEED"])

# Indian market crisis events
CRISIS_EVENTS = {
    "2011-08-09": ("US Downgrade\n+ Euro Crisis",     "top"),
    "2013-08-28": ("Taper Tantrum\nINR Crash",         "bottom"),
    "2015-08-24": ("China Deval.\nGlobal Selloff",     "top"),
    "2016-11-08": ("Demonetisation",                    "bottom"),
    "2018-09-21": ("IL&FS / NBFC\nCrisis",             "top"),
    "2020-03-23": ("COVID-19\nCrash Bottom",           "bottom"),
    "2022-06-17": ("Rate Hike\nSelloff",               "top"),
}


# ─── Data ────────────────────────────────────────────────────────────────────

def impute_missing_values(df, window=10):
    result = df.copy()
    for col in result.columns:
        s = result[col]
        nans = s[s.isna()].index
        for idx in nans:
            pos = s.index.get_loc(idx)
            vals = []
            for k in range(1, window + 1):
                if pos - k >= 0 and not np.isnan(s.iloc[pos - k]):
                    vals.append(s.iloc[pos - k])
                if pos + k < len(s) and not np.isnan(s.iloc[pos + k]):
                    vals.append(s.iloc[pos + k])
            if vals:
                result.at[idx, col] = np.mean(vals)
    return result.ffill().bfill()


def load_nifty50_data(tickers, start, end):
    import yfinance as yf

    print(f"\n[Data] Downloading {len(tickers)} Nifty 50 tickers...")
    print(f"[Data] Period: {start} to {end}")
    t0 = time.time()

    raw = yf.download(tickers, start=start, end=end,
                      auto_adjust=True, progress=True, threads=True)["Close"]

    print(f"[Data] Download: {raw.shape} in {time.time()-t0:.1f}s")

    # Drop tickers with >20% missing
    thresh = int(0.80 * len(raw))
    n_before = raw.shape[1]
    raw = raw.dropna(axis=1, thresh=thresh)
    print(f"[Data] Dropped {n_before - raw.shape[1]} tickers "
          f"(>20% missing), {raw.shape[1]} remaining")

    print("[Data] Imputing NaNs (±10 neighbor mean)...")
    raw = impute_missing_values(raw)
    print(f"[Data] NaNs remaining: {raw.isna().sum().sum()}")

    log_ret = np.log(raw / raw.shift(1)).dropna()

    # Download Nifty 50 index
    print("[Data] Downloading ^NSEI (Nifty 50 index)...")
    nifty = yf.download("^NSEI", start=start, end=end,
                        auto_adjust=True, progress=False)["Close"]
    # yfinance may return DataFrame for single ticker — squeeze to Series
    if isinstance(nifty, pd.DataFrame):
        nifty = nifty.squeeze()
    nifty = nifty.reindex(log_ret.index, method="ffill")

    print(f"[Data] Final: {log_ret.shape}, "
          f"{log_ret.index[0].date()} to {log_ret.index[-1].date()}")
    print(f"[Data] μ={log_ret.values.mean():.6f}  σ={log_ret.values.std():.4f}")
    return log_ret, nifty


# ─── Correlation ─────────────────────────────────────────────────────────────

def empirical_correlation(R):
    T, N = R.shape
    Rc = R - R.mean(axis=0)
    cov = (Rc.T @ Rc) / (T - 1)
    sd = np.sqrt(np.diag(cov))
    sd[sd < 1e-12] = 1.0
    C = cov / np.outer(sd, sd)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0)


# ─── MP Denoising ────────────────────────────────────────────────────────────

def mp_denoise(C, Q):
    N = C.shape[0]
    ev, evec = linalg.eigh(C)

    lp = (1.0 + np.sqrt(1.0 / Q)) ** 2
    noise = ev <= lp
    n_noise = int(noise.sum())
    mu = ev[noise].mean() if n_noise > 0 else 0.0

    ev_clean = ev.copy()
    ev_clean[noise] = mu

    C_clean = evec @ np.diag(ev_clean) @ evec.T
    C_clean = 0.5 * (C_clean + C_clean.T)

    d = np.sqrt(np.abs(np.diag(C_clean)))
    d[d < 1e-12] = 1.0
    C_clean /= np.outer(d, d)
    np.fill_diagonal(C_clean, 1.0)
    C_clean = np.clip(C_clean, -1.0, 1.0)

    min_ev = linalg.eigvalsh(C_clean).min()
    if min_ev < 0:
        C_clean += (-min_ev + 1e-9) * np.eye(N)

    info = dict(lambda_plus=float(lp), n_noise=n_noise,
                n_signal=N - n_noise, noise_fraction=n_noise / N,
                min_ev_clean=float(linalg.eigvalsh(C_clean).min()))
    return C_clean, info


# ─── Correlation spectral (reference) ───────────────────────────────────────

def correlation_spectral_metrics(C_clean, N):
    ev = np.sort(np.real(linalg.eigvalsh(C_clean)))[::-1]
    lam1, lam2 = float(ev[0]), float(ev[1])
    return dict(lambda1_C=lam1, lambda2_C=lam2,
                delta_C=lam2 / N,
                tau_C=N / lam2 if lam2 > 1e-10 else 1e8)


# ─── Window processor ───────────────────────────────────────────────────────

def _process_window(args):
    idx, t_start, t_end, R, Q, date_end, date_start, n_windows = args
    wt0 = time.time()

    C_emp = empirical_correlation(R)
    C_clean, mp = mp_denoise(C_emp, Q)

    pt0 = time.time()
    G = build_pmfg(C_clean, verbose=False)
    pmfg_sec = time.time() - pt0

    P, pinfo = pmfg_transition_matrix(G)
    spec = pmfg_spectral_gap(P)
    cmet = correlation_spectral_metrics(C_clean, R.shape[1])

    return dict(
        _idx=idx, date=date_end, window_start=date_start,
        mixing_time_pmfg=spec["mixing_time_pmfg"],
        spectral_gap_pmfg=spec["spectral_gap_pmfg"],
        lambda1_pmfg=spec["lambda1_pmfg"],
        lambda2_pmfg=spec["lambda2_pmfg"],
        tau_C=cmet["tau_C"], delta_C=cmet["delta_C"],
        lambda1_C=cmet["lambda1_C"], lambda2_C=cmet["lambda2_C"],
        pmfg_n_edges=pinfo["n_edges"],
        pmfg_mean_degree=pinfo["mean_degree"],
        pmfg_density=pinfo["density"],
        lambda_plus=mp["lambda_plus"],
        n_noise=mp["n_noise"], n_signal=mp["n_signal"],
        noise_fraction=mp["noise_fraction"],
        min_ev_clean=mp["min_ev_clean"],
        pmfg_time_s=pmfg_sec, window_time_s=time.time() - wt0,
    )


# ─── Rolling analysis ───────────────────────────────────────────────────────

def rolling_analysis(log_ret, T_window=252, step=21):
    T_total, N = log_ret.shape
    dates = log_ret.index
    Q = T_window / N
    lp = (1.0 + np.sqrt(1.0 / Q)) ** 2
    n_windows = (T_total - T_window) // step + 1

    print(f"\n{'─'*62}")
    print(f"  Rolling Window Analysis  (Nifty 50 — sequential)")
    print(f"  N={N}  T_window={T_window}  Q={Q:.4f}  λ+={lp:.4f}")
    print(f"  Total windows: {n_windows} (step={step} days)")
    print(f"{'─'*62}")

    window_args = []
    for i, t_end in enumerate(range(T_window, T_total, step)):
        t_start = t_end - T_window
        R = log_ret.iloc[t_start:t_end].values
        window_args.append((i, t_start, t_end, R, Q,
                            dates[t_end - 1], dates[t_start], n_windows))

    # First window — verbose
    print(f"  [  1/{n_windows}] {dates[window_args[0][1]].date()} → "
          f"{dates[window_args[0][2]-1].date()}  (verbose)")
    ft0 = time.time()
    C0 = empirical_correlation(window_args[0][3])
    C0c, _ = mp_denoise(C0, Q)
    _ = build_pmfg(C0c, verbose=True)
    first = _process_window(window_args[0])
    print(f"    First window: {time.time()-ft0:.1f}s")

    # Remaining — sequential
    t_all = time.time()
    print(f"\n  Processing remaining {len(window_args)-1} windows...")
    records = [first]
    for i, args in enumerate(window_args[1:], start=2):
        records.append(_process_window(args))
        done = len(records)
        if done % 10 == 0 or done == n_windows:
            elapsed = time.time() - t_all
            eta = elapsed / (done - 1) * (n_windows - done) if done > 1 else 0
            print(f"  [{done:3d}/{n_windows}] "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    records.sort(key=lambda r: r["_idx"])
    for r in records:
        del r["_idx"]

    df = pd.DataFrame(records).set_index("date")
    total = time.time() - t_all + first["window_time_s"]
    pmfg_total = df["pmfg_time_s"].sum()

    print(f"\n  Complete — {len(df)} windows.")
    print(f"  PMFG total: {pmfg_total:.1f}s  (avg {pmfg_total/len(df):.1f}s)")
    print(f"  Wall-clock: {total:.1f}s ({total/60:.1f} min)\n")
    return df


# ─── Verification ───────────────────────────────────────────────────────────

def verify(results, N):
    sep = "═" * 62
    print(f"\n{sep}\n  MATHEMATICAL VERIFICATION SUITE\n{sep}")
    checks = 0

    min_ev = results["min_ev_clean"].min()
    ok = min_ev >= -1e-6
    print(f"\n[1] C_clean PSD: min_ev={min_ev:.3e} → {'✓' if ok else '✗'}")
    checks += ok

    exp = 3 * (N - 2)
    ok = (results["pmfg_n_edges"] == exp).all()
    print(f"[2] PMFG edges = {exp}: {'✓' if ok else '✗'}")
    checks += ok

    ok = (results["mixing_time_pmfg"] < 1e7).all()
    print(f"[3] Mixing time finite: {'✓' if ok else '✗'}")
    checks += ok

    nf = results["noise_fraction"].mean()
    print(f"[4] MP noise fraction: {nf*100:.1f}%")
    checks += 1

    # Hypothesis: COVID crash
    pre = results.loc["2019-06-01":"2020-01-31", "mixing_time_pmfg"]
    crisis = results.loc["2020-02-20":"2020-04-30", "mixing_time_pmfg"]
    if len(crisis) > 0 and len(pre) > 0:
        ok = crisis.median() < pre.median()
        print(f"\n[5] COVID hypothesis: τ drops in crisis")
        print(f"    Pre-COVID median:  {pre.median():.2f}")
        print(f"    COVID median:      {crisis.median():.2f}")
        print(f"    → {'✓ CONFIRMED' if ok else '✗ NOT CONFIRMED'}")
        checks += ok

    # Hypothesis: 2018 IL&FS / NBFC crisis
    pre18 = results.loc["2018-01-01":"2018-06-30", "mixing_time_pmfg"]
    crisis18 = results.loc["2018-09-01":"2019-01-31", "mixing_time_pmfg"]
    if len(crisis18) > 0 and len(pre18) > 0:
        ok = crisis18.median() < pre18.median()
        print(f"\n[6] IL&FS/NBFC hypothesis: τ drops in crisis")
        print(f"    Pre-crisis median: {pre18.median():.2f}")
        print(f"    Crisis median:     {crisis18.median():.2f}")
        print(f"    → {'✓ CONFIRMED' if ok else '✗ NOT CONFIRMED'}")
        checks += ok

    print(f"\n{'─'*62}\n  {checks} checks completed\n{sep}\n")


# ─── Plot ────────────────────────────────────────────────────────────────────

def plot_results(results, nifty_idx, N, bench, output_path):
    BG, PANEL, GRID = "#0a0e14", "#111720", "#1e2a36"
    SPX_C, TAU_C, GAP_C = "#60a5fa", "#f87171", "#34d399"
    REF_C, EVT_C, SHADE = "#a78bfa", "#fbbf24", "#f87171"
    TEXT, SUB, MA_C, TIME_C = "#e2e8f0", "#94a3b8", "#f1f5f9", "#fb923c"

    if isinstance(nifty_idx, pd.DataFrame):
        nifty_idx = nifty_idx.squeeze()
    idx_a = nifty_idx.reindex(results.index, method="nearest")
    crash_labels = identify_crash_periods(idx_a)

    fig = plt.figure(figsize=(22, 20), facecolor=BG)
    fig.suptitle(
        "Proof of Concept · Market Crash Detection via Classical Random Walker\n"
        "Marchenko-Pastur + PMFG · "
        f"Nifty 50 ({N} stocks, 2010–2025)",
        fontsize=13, fontweight="bold", color=TEXT, y=0.99,
        fontfamily="monospace",
    )

    gs = gridspec.GridSpec(5, 1, hspace=0.08,
                           top=0.96, bottom=0.05, left=0.07, right=0.76)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=SUB, labelsize=8)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
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

    for dt_str in CRISIS_EVENTS:
        dt = pd.Timestamp(dt_str)
        if results.index[0] <= dt <= results.index[-1]:
            for ax in axes:
                ax.axvline(dt, color=EVT_C, lw=0.6, ls=":", alpha=0.7)

    # Panel 0: Nifty 50
    ax = axes[0]
    ax.plot(idx_a.index, idx_a.values, color=SPX_C, lw=1.4, alpha=0.9)
    ax.fill_between(idx_a.index, idx_a.values, idx_a.min(),
                    alpha=0.10, color=SPX_C)
    ax.set_ylabel("Index Level", color=TEXT, fontsize=9)
    ax.set_title("Nifty 50 Index", color=SUB, fontsize=8.5, loc="left", pad=3)

    # Panel 1: PMFG mixing time
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

    # Panel 2: Spectral gap
    ax = axes[2]
    gap = results["spectral_gap_pmfg"]
    ax.plot(gap.index, gap.values, color=GAP_C, lw=1.3, alpha=0.9,
            label="Δ_PMFG")
    ax.set_ylabel("Spectral Gap Δ", color=TEXT, fontsize=9)
    ax.set_title("PMFG Spectral Gap Δ = 1-|λ₂(P)|  [↑ = crash]",
                 color=SUB, fontsize=8.5, loc="left", pad=3)
    ax.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, fontsize=7.5)

    # Panel 3: τ_C reference
    ax = axes[3]
    tc = results["tau_C"].clip(upper=results["tau_C"].quantile(0.995))
    ax.plot(tc.index, tc.values, color=REF_C, lw=1.3, alpha=0.9,
            label="τ_C (correlation)")
    ax.set_ylabel("τ_C (ref)", color=TEXT, fontsize=9)
    ax.set_title("Correlation Mixing Time τ_C  [↑ = crash]",
                 color=SUB, fontsize=8.5, loc="left", pad=3)
    ax.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, fontsize=7.5)

    # Panel 4: PMFG time
    ax = axes[4]
    tp = results["pmfg_time_s"]
    ax.bar(tp.index, tp.values, width=15, color=TIME_C, alpha=0.7)
    ax.set_ylabel("PMFG Time (s)", color=TEXT, fontsize=9)
    ax.set_title(f"PMFG Time/Window  [avg={tp.mean():.1f}s, "
                 f"total={tp.sum():.0f}s]",
                 color=SUB, fontsize=8.5, loc="left", pad=3)

    for i, ax in enumerate(axes):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right",
                 color=SUB, fontsize=7.5)
        if i < 4: ax.set_xticklabels([])

    # Sidebar
    sidebar = (
        "  ┌─────────────────────────────┐\n"
        "  │   NIFTY 50 — POC            │\n"
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
    print(f"[Plot] Saved → {output_path}")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    sep = "═" * 62
    print(f"\n{sep}")
    print("  Proof of Concept: Crash Detection on Nifty 50")
    print("  Pipeline: Nifty 50 → MP Denoise → PMFG → Random Walker")
    print(f"  Period: {CFG['START']} to {CFG['END']}")
    print(sep)

    total_t0 = time.time()

    # 1. Data
    log_ret, nifty_idx = load_nifty50_data(
        CFG["TICKERS"], CFG["START"], CFG["END"])
    N = log_ret.shape[1]
    Q = CFG["T_WINDOW"] / N
    print(f"\n[Config] N={N}, Q={Q:.4f}, λ+={(1+np.sqrt(1/Q))**2:.4f}")

    # 2. Rolling pipeline
    results = rolling_analysis(log_ret, CFG["T_WINDOW"], CFG["STEP"])

    # 3. Verification
    verify(results, N)

    # 4. Benchmarks
    print("[Benchmarks] Computing crash prediction metrics...")

    bench_pmfg = compute_benchmarks(
        results, nifty_idx, col="mixing_time_pmfg",
        name="Classical Walker (PMFG)", direction="below")

    bench_corr = compute_benchmarks(
        results, nifty_idx, col="tau_C",
        name="Correlation τ_C (ref)", direction="above")

    bench_szegedy = dict(
        model="Quantum Szegedy Walker",
        precision=float('nan'), recall=float('nan'),
        f1=float('nan'), lead_time_days=float('nan'),
        auc_roc=float('nan'), threshold=float('nan'),
        n_alerts=0, n_crash_days=0, n_total_days=0)

    all_bench = [bench_pmfg, bench_corr, bench_szegedy]
    print_benchmark_table(all_bench)

    # 5. Save
    results.to_csv(CFG["OUTPUT_CSV"])
    print(f"[CSV] Results → {CFG['OUTPUT_CSV']}")
    pd.DataFrame(all_bench).to_csv(CFG["BENCH_CSV"], index=False)
    print(f"[CSV] Benchmarks → {CFG['BENCH_CSV']}")

    # 6. Plot
    plot_results(results, nifty_idx, N, bench_pmfg, CFG["OUTPUT_PNG"])

    total = time.time() - total_t0
    print(f"\n[Done] Proof of concept complete in {total:.0f}s "
          f"({total/60:.1f} min)\n")
    return results


# ── RUN ──────────────────────────────────────────────────────────────────────
results = main()
