"""
================================================================================
  Data Cleaning, Denoising & Shared Mathematics
================================================================================
  Run this ONCE to download and prepare Nifty 50 data.
  Both the classical and quantum walkers import functions from this module.

  Usage (Colab):
    !pip install -q yfinance numpy pandas scipy networkx matplotlib
    %run scripts/clean_data.py

  Usage (local):
    python scripts/clean_data.py

  Shared functions (importable):
    empirical_correlation(R)              → (N×N) sample correlation
    mp_denoise(C, Q)                      → (C_clean, info_dict)
    build_pmfg(corr_matrix)               → nx.Graph with 3(N-2) edges
    pmfg_transition_matrix(G)             → (P, info_dict)

  Outputs (all in ./data/):
    clean_log_returns.csv                 — (T × N) log-return matrix
    clean_nifty_index.csv                 — (T × 1) ^NSEI close
    clean_crash_labels.csv                — binary crash labels + drawdown
    clean_metadata.csv                    — per-window MP diagnostics
    clean_ticker_audit.csv                — per-ticker NaN audit
    clean_config.csv                      — snapshot of all parameters
================================================================================
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import linalg
import networkx as nx


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Resolve paths relative to THIS file so it works from any working directory
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

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

CFG = dict(
    START         = "2010-01-01",
    END           = "2025-03-31",
    T_WINDOW      = 252,           # rolling window (1 trading year)
    STEP          = 21,            # step size (~1 month)
    IMPUTE_WINDOW = 10,            # ±N-neighbour mean for NaN imputation
    DROP_THRESH   = 0.80,          # drop ticker if <80% of days have data
    DRAWDOWN_THR  = 0.15,          # crash = >15% below rolling 252-day peak
    PEAK_WINDOW   = 252,
    OUTPUT_DIR    = DATA_DIR,
)

# Indian market crisis events (for annotation, not computation)
CRISIS_EVENTS = {
    "2011-08-09": ("US Downgrade + Euro Crisis",    "top"),
    "2013-08-28": ("Taper Tantrum / INR Crash",      "bottom"),
    "2015-08-24": ("China Devaluation Selloff",      "top"),
    "2016-11-08": ("Demonetisation",                  "bottom"),
    "2018-09-21": ("IL&FS / NBFC Crisis",            "top"),
    "2020-03-23": ("COVID-19 Crash Bottom",          "bottom"),
    "2022-06-17": ("Rate Hike Selloff",              "top"),
}


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED FUNCTIONS — used by both classical and quantum walkers
# ══════════════════════════════════════════════════════════════════════════════

# ─── Imputation ──────────────────────────────────────────────────────────────

def impute_missing_values(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Impute NaN values using the mean of up to `window` valid neighbours
    on each side. Falls back to forward-fill then back-fill for edges.
    Operates on raw prices (before log-return transformation).
    """
    result = df.copy()
    for col in result.columns:
        s = result[col]
        nan_idx = s[s.isna()].index
        for idx in nan_idx:
            pos = s.index.get_loc(idx)
            vals = []
            for k in range(1, window + 1):
                if pos - k >= 0 and not np.isnan(s.iloc[pos - k]):
                    vals.append(s.iloc[pos - k])
                if pos + k < len(s) and not np.isnan(s.iloc[pos + k]):
                    vals.append(s.iloc[pos + k])
            if vals:
                result.at[idx, col] = float(np.mean(vals))
    return result.ffill().bfill()


# ─── Empirical Correlation ───────────────────────────────────────────────────

def empirical_correlation(R: np.ndarray) -> np.ndarray:
    """
    Compute sample correlation matrix from (T × N) return matrix.
    Demean → covariance → standardise → clip → force diagonal=1.
    """
    T, N = R.shape
    Rc = R - R.mean(axis=0)
    cov = (Rc.T @ Rc) / (T - 1)
    sd = np.sqrt(np.diag(cov))
    sd[sd < 1e-12] = 1.0
    C = cov / np.outer(sd, sd)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0)


# ─── Marchenko-Pastur Denoising ─────────────────────────────────────────────

def mp_denoise(C: np.ndarray, Q: float) -> tuple:
    """
    Marchenko-Pastur eigenvalue clipping.

    1. Decompose C = V Λ Vᵀ
    2. MP upper edge: λ+ = (1 + 1/√Q)²
    3. Clip noise eigenvalues (≤ λ+) to their mean
    4. Reconstruct, re-normalise diagonal, project PSD

    Parameters
    ----------
    C : (N, N) symmetric correlation matrix
    Q : ratio T/N (number of observations / number of assets)

    Returns
    -------
    C_clean : (N, N) denoised correlation matrix (PSD)
    info    : dict with MP diagnostics
    """
    N = C.shape[0]
    ev, evec = linalg.eigh(C)
    ev   = np.real(ev)
    evec = np.real(evec)

    lambda_plus = (1.0 + np.sqrt(1.0 / Q)) ** 2

    noise_mask = ev <= lambda_plus
    n_noise  = int(noise_mask.sum())
    n_signal = N - n_noise
    mu_noise = ev[noise_mask].mean() if n_noise > 0 else 0.0

    ev_clean = ev.copy()
    ev_clean[noise_mask] = mu_noise

    # Reconstruct and symmetrise
    C_clean = evec @ np.diag(ev_clean) @ evec.T
    C_clean = 0.5 * (C_clean + C_clean.T)

    # Re-normalise diagonal to 1
    d = np.sqrt(np.abs(np.diag(C_clean)))
    d[d < 1e-12] = 1.0
    C_clean /= np.outer(d, d)
    np.fill_diagonal(C_clean, 1.0)
    C_clean = np.clip(C_clean, -1.0, 1.0)

    # PSD projection: shift so min eigenvalue ≥ 0
    min_ev = linalg.eigvalsh(C_clean).min()
    if min_ev < 0.0:
        C_clean += (-min_ev + 1e-9) * np.eye(N)

    info = dict(
        lambda_plus    = float(lambda_plus),
        n_noise        = n_noise,
        n_signal       = n_signal,
        noise_fraction = n_noise / N,
        mu_noise       = float(mu_noise),
        trace_original = float(ev.sum()),
        trace_clean    = float(ev_clean.sum()),
        trace_delta    = abs(float(ev.sum()) - float(ev_clean.sum())),
        min_ev_clean   = float(linalg.eigvalsh(C_clean).min()),
    )
    return C_clean, info


# ─── PMFG Construction ──────────────────────────────────────────────────────

def build_pmfg(corr_matrix: np.ndarray, verbose: bool = True) -> nx.Graph:
    """
    Construct the Planar Maximally Filtered Graph (PMFG).

    Greedy algorithm: add edges by descending |correlation| if the graph
    remains planar.  Stops at 3(N-2) edges.

    Parameters
    ----------
    corr_matrix : (N, N) symmetric MP-denoised correlation matrix
    verbose     : print progress

    Returns
    -------
    G : nx.Graph with N nodes and 3(N-2) edges
        Edge attributes: 'weight' = |corr|, 'corr' = signed correlation
    """
    N = corr_matrix.shape[0]
    max_edges = 3 * (N - 2)

    # Vectorised edge extraction and sorting
    rows, cols = np.triu_indices(N, k=1)
    abs_corr    = np.abs(corr_matrix[rows, cols])
    signed_corr = corr_matrix[rows, cols]

    order = np.argsort(-abs_corr)
    rows, cols       = rows[order], cols[order]
    abs_corr         = abs_corr[order]
    signed_corr      = signed_corr[order]
    n_candidates     = len(rows)

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


# ─── Transition Matrix ──────────────────────────────────────────────────────

def pmfg_transition_matrix(G: nx.Graph) -> tuple:
    """
    Build a row-stochastic transition matrix from the PMFG graph.

    P_ij = w_ij / Σ_k w_ik   for edges (i,j) in G
    P_ij = 0                   if (i,j) not in G

    Returns
    -------
    P    : (N, N) row-stochastic transition matrix
    info : dict with graph statistics
    """
    N = G.number_of_nodes()
    A = np.zeros((N, N))
    for i, j, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        A[i, j] = A[j, i] = w

    deg = A.sum(axis=1)
    deg[deg < 1e-12] = 1e-12
    P = A / deg[:, None]

    info = dict(
        n_nodes    = N,
        n_edges    = G.number_of_edges(),
        mean_degree= 2 * G.number_of_edges() / N,
        min_degree = min(dict(G.degree()).values()) if N > 0 else 0,
        max_degree = max(dict(G.degree()).values()) if N > 0 else 0,
        density    = nx.density(G),
    )
    return P, info


# ══════════════════════════════════════════════════════════════════════════════
#  DATA DOWNLOAD & CLEANING  (run when executed as __main__)
# ══════════════════════════════════════════════════════════════════════════════

def download_and_clean():
    """Download Nifty 50 data, clean it, and save CSVs."""
    import yfinance as yf

    os.makedirs(CFG["OUTPUT_DIR"], exist_ok=True)

    sep = "=" * 70
    print(f"\n{sep}")
    print("  Data Cleaning — Nifty 50")
    print(sep)

    # ── 1. Download ──────────────────────────────────────────────────────────
    tickers = NIFTY50_TICKERS
    print(f"\n[1/6] Downloading {len(tickers)} tickers + ^NSEI "
          f"({CFG['START']} → {CFG['END']}) …")
    t0 = time.time()

    raw = yf.download(
        tickers + ["^NSEI"],
        start=CFG["START"], end=CFG["END"],
        auto_adjust=True, progress=True, threads=True,
    )["Close"]
    print(f"    Downloaded {raw.shape} in {time.time()-t0:.1f}s")

    # ── 2. Separate index, drop bad tickers ──────────────────────────────────
    print("\n[2/6] Separating index and quality-filtering tickers …")

    nifty_raw = raw["^NSEI"].copy()
    if isinstance(nifty_raw, pd.DataFrame):
        nifty_raw = nifty_raw.squeeze()

    stocks_raw = raw.drop(columns=["^NSEI"], errors="ignore")

    n_before = stocks_raw.shape[1]
    min_obs  = int(CFG["DROP_THRESH"] * len(stocks_raw))
    stocks_raw = stocks_raw.dropna(axis=1, thresh=min_obs)
    dropped = n_before - stocks_raw.shape[1]
    print(f"    Dropped {dropped} tickers (<{CFG['DROP_THRESH']*100:.0f}% coverage), "
          f"{stocks_raw.shape[1]} remaining")

    # ── 3. Imputation ────────────────────────────────────────────────────────
    print(f"\n[3/6] Imputing NaNs (±{CFG['IMPUTE_WINDOW']}-neighbour mean) …")

    nan_counts_before = stocks_raw.isna().sum()
    total_days        = len(stocks_raw)

    stocks_clean = impute_missing_values(stocks_raw, window=CFG["IMPUTE_WINDOW"])
    nan_after    = stocks_clean.isna().sum().sum()
    print(f"    NaNs remaining: {nan_after}")
    if nan_after > 0:
        stocks_clean.dropna(inplace=True)

    # ── 4. Log returns ───────────────────────────────────────────────────────
    print("\n[4/6] Computing log returns …")

    log_ret = np.log(stocks_clean / stocks_clean.shift(1)).dropna()
    N, T = log_ret.shape[1], log_ret.shape[0]
    print(f"    Shape : {log_ret.shape}  "
          f"({log_ret.index[0].date()} → {log_ret.index[-1].date()})")
    print(f"    μ = {log_ret.values.mean():.6f}  σ = {log_ret.values.std():.6f}")

    # ── 5. Align Nifty index + crash labels ──────────────────────────────────
    print("\n[5/6] Aligning Nifty index and computing crash labels …")

    nifty_aligned = nifty_raw.reindex(log_ret.index, method="ffill").dropna()
    nifty_aligned.name = "NSEI"

    rolling_peak = nifty_aligned.rolling(CFG["PEAK_WINDOW"], min_periods=1).max()
    drawdown     = (nifty_aligned - rolling_peak) / rolling_peak
    crash_labels = (drawdown < -CFG["DRAWDOWN_THR"]).astype(int)
    crash_labels.name = "crash"

    n_crash = crash_labels.sum()
    pct     = 100 * n_crash / len(crash_labels)
    print(f"    Crash days: {n_crash} / {len(crash_labels)} ({pct:.1f}%)")

    # ── 6. Rolling MP parameters (metadata for both walkers) ─────────────────
    print("\n[6/6] Computing rolling MP parameters …")

    T_WINDOW = CFG["T_WINDOW"]
    STEP     = CFG["STEP"]
    Q        = T_WINDOW / N
    lp       = (1.0 + np.sqrt(1.0 / Q)) ** 2

    print(f"    N={N}  Q={Q:.4f}  λ+={lp:.4f}")

    ret_arr   = log_ret.values.astype(float)
    dates_arr = log_ret.index
    n_windows = (T - T_WINDOW) // STEP + 1

    meta_records = []
    for i, t_end in enumerate(range(T_WINDOW, T, STEP)):
        t_start = t_end - T_WINDOW
        R       = ret_arr[t_start:t_end]
        date    = dates_arr[t_end - 1]

        C = empirical_correlation(R)
        ev = linalg.eigvalsh(C)
        noise_mask = ev <= lp

        triu_idx  = np.triu_indices(N, k=1)
        mean_corr = float(C[triu_idx].mean())

        meta_records.append(dict(
            date           = date,
            window_start   = dates_arr[t_start],
            Q              = Q,
            lambda_plus    = lp,
            n_assets       = N,
            n_noise        = int(noise_mask.sum()),
            n_signal       = N - int(noise_mask.sum()),
            noise_fraction = int(noise_mask.sum()) / N,
            mean_corr      = mean_corr,
            ev_min         = float(ev[0]),
            ev_max         = float(ev[-1]),
            ev_lambda2     = float(ev[-2]) if N >= 2 else 0.0,
        ))

        if i % 50 == 0:
            print(f"    window {i+1}/{n_windows}  {date.date()}  "
                  f"noise={int(noise_mask.sum())}/{N}  mean_ρ={mean_corr:.3f}")

    meta_df = pd.DataFrame(meta_records).set_index("date")

    # ── SAVE ─────────────────────────────────────────────────────────────────
    paths = dict(
        log_ret      = os.path.join(DATA_DIR, "clean_log_returns.csv"),
        nifty        = os.path.join(DATA_DIR, "clean_nifty_index.csv"),
        crash        = os.path.join(DATA_DIR, "clean_crash_labels.csv"),
        meta         = os.path.join(DATA_DIR, "clean_metadata.csv"),
        ticker_audit = os.path.join(DATA_DIR, "clean_ticker_audit.csv"),
        config       = os.path.join(DATA_DIR, "clean_config.csv"),
    )

    log_ret.to_csv(paths["log_ret"])
    print(f"\n[Saved] Log returns      → {paths['log_ret']}  {log_ret.shape}")

    nifty_aligned.to_frame().to_csv(paths["nifty"])
    print(f"[Saved] Nifty index      → {paths['nifty']}  {nifty_aligned.shape}")

    crash_df = pd.DataFrame({
        "NSEI": nifty_aligned, "drawdown": drawdown, "crash": crash_labels,
    })
    crash_df.to_csv(paths["crash"])
    print(f"[Saved] Crash labels     → {paths['crash']}  {crash_df.shape}")

    meta_df.to_csv(paths["meta"])
    print(f"[Saved] MP metadata      → {paths['meta']}  {meta_df.shape}")

    # Ticker audit
    audit_df = pd.DataFrame({
        "ticker"   : nan_counts_before.index,
        "nan_before": nan_counts_before.values,
        "total_days": total_days,
        "nan_pct"  : (nan_counts_before.values / total_days * 100).round(2),
        "retained" : [t in log_ret.columns for t in nan_counts_before.index],
    }).set_index("ticker").sort_values("nan_pct", ascending=False)
    audit_df.to_csv(paths["ticker_audit"])
    print(f"[Saved] Ticker audit     → {paths['ticker_audit']}")

    # Config snapshot (exclude non-serialisable keys)
    cfg_save = {k: v for k, v in CFG.items() if isinstance(v, (str, int, float))}
    config_df = pd.Series(cfg_save).to_frame("value")
    config_df.to_csv(paths["config"])
    print(f"[Saved] Config snapshot  → {paths['config']}")

    # ── SUMMARY ──────────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  CLEANING SUMMARY")
    print(sep)
    print(f"  Assets retained   : {N} / {n_before}")
    print(f"  Trading days (T)  : {T}  ({log_ret.index[0].date()} → {log_ret.index[-1].date()})")
    print(f"  Q = T/N           : {Q:.4f}")
    print(f"  MP λ+             : {lp:.4f}")
    print(f"  Rolling windows   : {n_windows}")
    print(f"  Crash days        : {n_crash} ({pct:.1f}%)")
    print(f"\n  ✓ Clean data ready. Run walker.py next.\n")

    return log_ret, nifty_aligned, crash_labels, meta_df


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    download_and_clean()
