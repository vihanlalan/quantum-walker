"""
================================================================================
Phase 1 — Predicting Market Crashes via Classical Random Walk Mixing Times
           on Denoised Correlation Complexes (MP + PMFG)
================================================================================
Target  : S&P 500, 2005–2025 (multiple crisis periods)
Pipeline: yfinance data → NaN imputation (±10 neighbor mean) →
          Log-returns → Rolling correlation → MP denoising →
          PMFG graph construction → Classical random walker →
          Spectral gap Δ → Mixing time τ → Crash detection benchmarks

HYPOTHESIS
──────────
On the PMFG-filtered graph:
  • Normal markets: diverse sector correlations → heterogeneous graph
    → random walker takes LONGER to mix (high τ)
  • Crisis: all assets lock to single market factor → homogeneous graph
    → random walker mixes FASTER (low τ)

  ⇒ Mixing time τ DECREASES during crashes — early warning signal.

BENCHMARK OUTPUT
────────────────
All metrics are stored in a standardised format for direct comparison
with a quantum Szegedy walker implementation.
================================================================================
"""

# ── Standard library ─────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time
import builtins

# Force UTF-8 output on Windows (needed for ═, ─, →, λ, etc.)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Force unbuffered output (critical for seeing progress with multiprocessing)
os.environ["PYTHONUNBUFFERED"] = "1"
_original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _original_print(*args, **kwargs)

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

# ── Local modules ────────────────────────────────────────────────────────────
from pmfg import build_pmfg, pmfg_transition_matrix, pmfg_spectral_gap
from benchmarks import (
    compute_benchmarks, print_benchmark_table,
    identify_crash_periods, generate_alerts,
)

# ─────────────────────────────────────────────────────────────────────────────
#  0.  GLOBAL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# S&P 500 tickers — comprehensive list of constituents
SP500_TICKERS = [
    "AAPL","ABBV","ABT","ACN","ADBE","ADI","ADM","ADP","ADSK","AEE",
    "AEP","AES","AFL","AIG","AIZ","AJG","AKAM","ALB","ALGN","ALK",
    "ALL","ALLE","AMAT","AMCR","AMD","AME","AMGN","AMP","AMT","AMZN",
    "ANET","ANSS","AON","AOS","APA","APD","APH","APTV","ARE","ATO",
    "ATVI","AVB","AVGO","AVY","AWK","AXP","AZO","BA","BAC","BAX",
    "BBWI","BBY","BDX","BEN","BF.B","BIIB","BIO","BK","BKNG","BKR",
    "BLK","BMY","BR","BRK.B","BRO","BSX","BWA","BXP","C","CAG",
    "CAH","CARR","CAT","CB","CBOE","CBRE","CCI","CCL","CDAY","CDNS",
    "CDW","CE","CEG","CF","CFG","CHD","CHRW","CHTR","CI","CINF",
    "CL","CLX","CMA","CMCSA","CME","CMG","CMI","CMS","CNC","CNP",
    "COF","COO","COP","COST","CPB","CPRT","CPT","CRL","CRM","CSCO",
    "CSGP","CSX","CTAS","CTLT","CTRA","CTSH","CTVA","CVS","CVX","CZR",
    "D","DAL","DD","DE","DFS","DG","DGX","DHI","DHR","DIS",
    "DISH","DLR","DLTR","DOV","DOW","DPZ","DRI","DTE","DUK","DVA",
    "DVN","DXC","DXCM","EA","EBAY","ECL","ED","EFX","EIX","EL",
    "EMN","EMR","ENPH","EOG","EPAM","EQIX","EQR","EQT","ES","ESS",
    "ETN","ETR","ETSY","EVRG","EW","EXC","EXPD","EXPE","EXR","F",
    "FANG","FAST","FBHS","FCX","FDS","FDX","FE","FFIV","FIS","FISV",
    "FITB","FLT","FMC","FOX","FOXA","FRC","FRT","FTNT","FTV","GD",
    "GE","GILD","GIS","GL","GLW","GM","GNRC","GOOG","GOOGL","GPC",
    "GPN","GRMN","GS","GWW","HAL","HAS","HBAN","HCA","HD","HOLX",
    "HON","HPE","HPQ","HRL","HSIC","HST","HSY","HUM","HWM","IBM",
    "ICE","IDXX","IEX","IFF","ILMN","INCY","INTC","INTU","INVH","IP",
    "IPG","IQV","IR","IRM","ISRG","IT","ITW","IVZ","J","JBHT",
    "JCI","JKHY","JNJ","JNPR","JPM","K","KDP","KEY","KEYS","KHC",
    "KIM","KLAC","KMB","KMI","KMX","KO","KR","L","LDOS","LEN",
    "LH","LHX","LIN","LKQ","LLY","LMT","LNC","LNT","LOW","LRCX",
    "LUMN","LUV","LVS","LW","LYB","LYV","MA","MAA","MAR","MAS",
    "MCD","MCHP","MCK","MCO","MDLZ","MDT","MET","META","MGM","MHK",
    "MKC","MKTX","MLM","MMC","MMM","MNST","MO","MOH","MOS","MPC",
    "MPWR","MRK","MRNA","MRO","MS","MSCI","MSFT","MSI","MTB","MTCH",
    "MTD","MU","NCLH","NDAQ","NDSN","NEE","NEM","NFLX","NI","NKE",
    "NOC","NOW","NRG","NSC","NTAP","NTRS","NUE","NVDA","NVR","NWL",
    "NWS","NWSA","NXPI","O","ODFL","OGN","OKE","OMC","ON","ORCL",
    "ORLY","OTIS","OXY","PARA","PAYC","PAYX","PCAR","PCG","PEAK","PEG",
    "PEP","PFE","PFG","PG","PGR","PH","PHM","PKG","PKI","PLD",
    "PM","PNC","PNR","PNW","POOL","PPG","PPL","PRU","PSA","PSX",
    "PTC","PVH","PWR","PXD","PYPL","QCOM","QRVO","RCL","RE","REG",
    "REGN","RF","RHI","RJF","RL","RMD","ROK","ROL","ROP","ROST",
    "RSG","RTX","SBAC","SBNY","SBUX","SCHW","SEE","SHW","SIVB","SJM",
    "SLB","SNA","SNPS","SO","SPG","SPGI","SRE","STE","STT","STX",
    "STZ","SWK","SWKS","SYF","SYK","SYY","T","TAP","TDG","TDY",
    "TECH","TEL","TER","TFC","TFX","TGT","TMO","TMUS","TPR","TRGP",
    "TRMB","TROW","TRV","TSCO","TSLA","TSN","TT","TTWO","TXN","TXT",
    "TYL","UAL","UDR","UHS","ULTA","UNH","UNP","UPS","URI","USB",
    "V","VFC","VICI","VLO","VMC","VNO","VRSK","VRSN","VRTX","VTR",
    "VTRS","VZ","WAB","WAT","WBA","WBD","WDC","WEC","WELL","WFC",
    "WHR","WM","WMB","WMT","WRB","WRK","WST","WTW","WY","WYNN",
    "XEL","XOM","XRAY","XYL","YUM","ZBH","ZBRA","ZION","ZTS",
]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CFG = dict(
    T_WINDOW     = 252,     # rolling window (1 trading year)
    STEP         = 21,      # window advance (1 month) — balance speed vs granularity
    START        = "2005-01-01",
    END          = "2025-03-31",
    RANDOM_SEED  = 42,
    TICKERS      = SP500_TICKERS,
    OUTPUT_PNG   = os.path.join(OUTPUT_DIR, "phase1_mixing_times.png"),
    OUTPUT_CSV   = os.path.join(OUTPUT_DIR, "phase1_results.csv"),
    BENCH_CSV    = os.path.join(OUTPUT_DIR, "phase1_benchmarks.csv"),
)

np.random.seed(CFG["RANDOM_SEED"])

# Known crisis periods for annotations
CRISIS_EVENTS = {
    "2007-06-07": ("Bear Stearns\nHF Collapse",     "top"),
    "2008-09-15": ("Lehman\nBankruptcy",             "bottom"),
    "2008-10-03": ("TARP\nSigned",                   "top"),
    "2009-03-09": ("S&P 500\nCycle Bottom",          "bottom"),
    "2011-08-05": ("US Credit\nDowngrade",           "top"),
    "2015-08-24": ("China\nDevaluation",             "bottom"),
    "2018-12-24": ("Fed Tightening\nSelloff",        "top"),
    "2020-03-23": ("COVID-19\nCrash Bottom",         "bottom"),
    "2022-06-16": ("Rate Hike\nSelloff",             "top"),
}


# ─────────────────────────────────────────────────────────────────────────────
#  1.  DATA INGESTION & CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def impute_missing_values(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Impute NaN values using the mean of up to 10 valid values before
    and 10 valid values after the missing position.

    For edge cases (start/end of series), uses whatever neighbors exist.
    If still NaN after neighbor-mean, falls back to forward-fill then back-fill.
    """
    result = df.copy()

    for col in result.columns:
        series = result[col]
        nan_mask = series.isna()

        if not nan_mask.any():
            continue

        nan_indices = nan_mask[nan_mask].index
        for idx in nan_indices:
            pos = series.index.get_loc(idx)

            # Gather up to 10 valid values before
            before_vals = []
            for k in range(1, window + 1):
                if pos - k >= 0:
                    v = series.iloc[pos - k]
                    if not np.isnan(v):
                        before_vals.append(v)

            # Gather up to 10 valid values after
            after_vals = []
            for k in range(1, window + 1):
                if pos + k < len(series):
                    v = series.iloc[pos + k]
                    if not np.isnan(v):
                        after_vals.append(v)

            neighbors = before_vals + after_vals
            if neighbors:
                result.at[idx, col] = np.mean(neighbors)

    # Final fallback
    result = result.ffill().bfill()
    return result


def load_sp500_data(tickers: list, start: str, end: str):
    """
    Download S&P 500 constituent adjusted-close prices from Yahoo Finance.

    Returns (log_returns DataFrame, spx_close Series).
    """
    import yfinance as yf

    print(f"\n[Data] Downloading {len(tickers)} S&P 500 tickers...")
    print(f"[Data] Period: {start} to {end}")
    t0 = time.time()

    raw = yf.download(
        tickers, start=start, end=end,
        auto_adjust=True, progress=True, threads=True,
    )["Close"]

    elapsed = time.time() - t0
    print(f"[Data] Download complete in {elapsed:.1f}s")
    print(f"[Data] Raw shape: {raw.shape}")

    # Drop tickers missing >20% of data
    thresh = int(0.80 * len(raw))
    n_before = raw.shape[1]
    raw = raw.dropna(axis=1, thresh=thresh)
    n_after = raw.shape[1]
    print(f"[Data] Dropped {n_before - n_after} tickers "
          f"with >20% missing data ({n_after} remaining)")

    # Impute remaining missing values with ±10 neighbor mean
    print("[Data] Imputing missing values (±10 neighbor mean)...")
    raw = impute_missing_values(raw, window=10)

    remaining_nans = raw.isna().sum().sum()
    print(f"[Data] Remaining NaNs after imputation: {remaining_nans}")

    # Compute log-returns
    log_ret = np.log(raw / raw.shift(1)).dropna()

    # Download S&P 500 index
    print("[Data] Downloading ^GSPC (S&P 500 index)...")
    spx = yf.download(
        "^GSPC", start=start, end=end,
        auto_adjust=True, progress=False,
    )["Close"].reindex(log_ret.index, method="ffill")

    print(f"[Data] Final log-returns shape: {log_ret.shape}")
    print(f"[Data] Date range: {log_ret.index[0].date()} to "
          f"{log_ret.index[-1].date()}")
    print(f"[Data] Return stats — μ={log_ret.values.mean():.6f}  "
          f"σ={log_ret.values.std():.4f}")

    return log_ret, spx


# ─────────────────────────────────────────────────────────────────────────────
#  2.  EMPIRICAL CORRELATION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def empirical_correlation(R: np.ndarray) -> np.ndarray:
    """
    Compute sample correlation matrix from (T × N) return matrix.
    Demean → covariance → standardise → clip → force diagonal=1.
    """
    T, N = R.shape
    Rc = R - R.mean(axis=0)
    cov = (Rc.T @ Rc) / (T - 1)
    std = np.sqrt(np.diag(cov))
    std[std < 1e-12] = 1.0
    C = cov / np.outer(std, std)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, -1.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  3.  MARCHENKO-PASTUR DENOISING
# ─────────────────────────────────────────────────────────────────────────────

def mp_denoise(C: np.ndarray, Q: float) -> tuple:
    """
    Marchenko-Pastur eigenvalue clipping (trace-preserving).

    1. Decompose C = V Λ Vᵀ
    2. MP upper edge: λ+ = (1 + 1/√Q)²
    3. Clip noise eigenvalues to their mean
    4. Reconstruct, re-normalise diagonal, project PSD
    """
    N = C.shape[0]
    ev, evec = linalg.eigh(C)
    ev = np.real(ev)
    evec = np.real(evec)

    lambda_plus = (1.0 + np.sqrt(1.0 / Q)) ** 2

    noise_mask = ev <= lambda_plus
    n_noise = int(noise_mask.sum())
    n_signal = N - n_noise

    mu_noise = ev[noise_mask].mean() if n_noise > 0 else 0.0
    ev_clean = ev.copy()
    ev_clean[noise_mask] = mu_noise

    C_clean = evec @ np.diag(ev_clean) @ evec.T
    C_clean = 0.5 * (C_clean + C_clean.T)

    diag = np.sqrt(np.abs(np.diag(C_clean)))
    diag[diag < 1e-12] = 1.0
    C_clean /= np.outer(diag, diag)
    np.fill_diagonal(C_clean, 1.0)
    C_clean = np.clip(C_clean, -1.0, 1.0)

    # PSD projection
    ev_check = linalg.eigvalsh(C_clean)
    min_ev = float(ev_check.min())
    if min_ev < 0.0:
        C_clean += (-min_ev + 1e-9) * np.eye(N)

    info = dict(
        lambda_plus=float(lambda_plus),
        n_noise=n_noise,
        n_signal=n_signal,
        noise_fraction=n_noise / N,
        mu_noise=float(mu_noise),
        trace_original=float(ev.sum()),
        trace_clean=float(ev_clean.sum()),
        trace_delta=abs(float(ev.sum()) - float(ev_clean.sum())),
        min_ev_clean=float(linalg.eigvalsh(C_clean).min()),
    )
    return C_clean, info


# ─────────────────────────────────────────────────────────────────────────────
#  4.  SPECTRAL METRICS (from both C_clean and PMFG)
# ─────────────────────────────────────────────────────────────────────────────

def correlation_spectral_metrics(C_clean: np.ndarray, N: int) -> dict:
    """
    Extract spectral gap and mixing time from the cleaned correlation matrix.
    (Original approach — kept for comparison.)

    Δ_C = λ₂(C_clean) / N
    τ_C = N / λ₂(C_clean)
    """
    ev_C = linalg.eigvalsh(C_clean)
    ev_C = np.sort(ev_C)[::-1]

    lambda1_C = float(ev_C[0])
    lambda2_C = float(ev_C[1])

    delta_C = lambda2_C / N
    tau_C = N / lambda2_C if lambda2_C > 1e-10 else 1e8

    return dict(
        lambda1_C=lambda1_C,
        lambda2_C=lambda2_C,
        delta_C=delta_C,
        tau_C=tau_C,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  5.  ROLLING WINDOW ENGINE  (parallelised with multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

import multiprocessing
from functools import partial


def _process_window(args):
    """
    Process a single rolling window — top-level function for pickling.

    Each call is independent: correlation → MP denoise → PMFG → spectral.
    """
    idx, t_start, t_end, R, Q, date_end, date_start, n_windows = args

    window_t0 = time.time()

    # Step 1: Empirical correlation
    C_emp = empirical_correlation(R)

    # Step 2: Marchenko-Pastur denoising
    C_clean, mp_info = mp_denoise(C_emp, Q)

    # Step 3: Build PMFG on the MP-denoised correlation matrix
    pmfg_t0 = time.time()
    G_pmfg = build_pmfg(C_clean, verbose=False)
    pmfg_elapsed = time.time() - pmfg_t0

    # Step 4: Classical random walker on PMFG
    P_pmfg, pmfg_info = pmfg_transition_matrix(G_pmfg)
    pmfg_metrics = pmfg_spectral_gap(P_pmfg)

    # Step 5: Correlation-based metrics (for comparison)
    N = R.shape[1]
    corr_metrics = correlation_spectral_metrics(C_clean, N)

    window_elapsed = time.time() - window_t0

    return dict(
        _idx=idx,
        date=date_end,
        window_start=date_start,
        # PMFG-based (primary signal — hypothesis: τ DECREASES in crash)
        mixing_time_pmfg=pmfg_metrics["mixing_time_pmfg"],
        spectral_gap_pmfg=pmfg_metrics["spectral_gap_pmfg"],
        lambda1_pmfg=pmfg_metrics["lambda1_pmfg"],
        lambda2_pmfg=pmfg_metrics["lambda2_pmfg"],
        # Correlation-based (reference — τ INCREASES in crash)
        tau_C=corr_metrics["tau_C"],
        delta_C=corr_metrics["delta_C"],
        lambda1_C=corr_metrics["lambda1_C"],
        lambda2_C=corr_metrics["lambda2_C"],
        # PMFG graph stats
        pmfg_n_edges=pmfg_info["n_edges"],
        pmfg_mean_degree=pmfg_info["mean_degree"],
        pmfg_density=pmfg_info["density"],
        # MP diagnostics
        lambda_plus=mp_info["lambda_plus"],
        n_noise=mp_info["n_noise"],
        n_signal=mp_info["n_signal"],
        noise_fraction=mp_info["noise_fraction"],
        min_ev_clean=mp_info["min_ev_clean"],
        # Timing
        pmfg_time_s=pmfg_elapsed,
        window_time_s=window_elapsed,
    )


def rolling_analysis(
    log_ret: pd.DataFrame,
    T_window: int = 252,
    step: int = 21,
) -> pd.DataFrame:
    """
    Execute the full pipeline on every rolling window (parallelised):
      R → C_emp → MP denoise → PMFG → classical walker → spectral metrics

    Uses multiprocessing.Pool to distribute windows across all CPU cores.
    """
    T_total, N = log_ret.shape
    dates = log_ret.index
    Q = T_window / N
    lambda_plus = (1.0 + np.sqrt(1.0 / Q)) ** 2

    n_windows = (T_total - T_window) // step + 1
    n_workers = max(1, multiprocessing.cpu_count() - 1)  # leave 1 core free

    print(f"\n{'─'*62}")
    print(f"  Rolling Window Analysis  (parallelised)")
    print(f"  N={N}  T_window={T_window}  Q={Q:.4f}")
    print(f"  MP λ+ = (1 + 1/√Q)² = {lambda_plus:.4f}")
    print(f"  Total windows: {n_windows} (step={step} days)")
    print(f"  Workers: {n_workers} CPU cores")
    print(f"{'─'*62}")

    # Pre-extract all window data as numpy arrays (avoids pickling DataFrames)
    window_args = []
    for idx, t_end in enumerate(range(T_window, T_total, step)):
        t_start = t_end - T_window
        R = log_ret.iloc[t_start:t_end].values  # (T_window, N) numpy array
        window_args.append((
            idx, t_start, t_end, R, Q,
            dates[t_end - 1],    # date_end
            dates[t_start],      # date_start
            n_windows,
        ))

    # Run first window sequentially with verbose output (for sanity check)
    print(f"  [  1/{n_windows}] {dates[window_args[0][1]].date()} → "
          f"{dates[window_args[0][2]-1].date()}  (verbose, sequential)")
    first_args = list(window_args[0])
    # Run first window with verbose PMFG
    first_t0 = time.time()
    C_emp_0 = empirical_correlation(first_args[3])
    C_clean_0, _ = mp_denoise(C_emp_0, Q)
    _ = build_pmfg(C_clean_0, verbose=True)  # show PMFG progress once
    first_result = _process_window(window_args[0])
    print(f"    First window: {time.time() - first_t0:.1f}s")

    # Process remaining windows in parallel
    remaining_args = window_args[1:]
    t_parallel_start = time.time()

    print(f"\n  Processing remaining {len(remaining_args)} windows "
          f"across {n_workers} cores...")

    records = [first_result]
    with multiprocessing.Pool(processes=n_workers) as pool:
        for i, result in enumerate(
            pool.imap_unordered(_process_window, remaining_args)
        ):
            records.append(result)
            done = len(records)
            if done % 20 == 0 or done == n_windows:
                elapsed = time.time() - t_parallel_start
                eta = elapsed / (done - 1) * (n_windows - done) if done > 1 else 0
                print(f"  [{done:3d}/{n_windows}] "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    # Sort by original index to preserve chronological order
    records.sort(key=lambda r: r["_idx"])
    for r in records:
        del r["_idx"]

    df = pd.DataFrame(records).set_index("date")

    total_pmfg_time = df["pmfg_time_s"].sum()
    total_wall_time = time.time() - t_parallel_start + (first_result["window_time_s"])

    print(f"\n  Complete — {len(df)} windows processed.")
    print(f"  Total PMFG CPU-time: {total_pmfg_time:.1f}s "
          f"(avg {total_pmfg_time/len(df):.1f}s/window)")
    print(f"  Wall-clock time: {total_wall_time:.1f}s "
          f"({total_wall_time/60:.1f} min)")
    print(f"  Speedup from {n_workers} cores: "
          f"~{total_pmfg_time / total_wall_time:.1f}×\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  6.  VERIFICATION SUITE
# ─────────────────────────────────────────────────────────────────────────────

def verify(results: pd.DataFrame, N: int) -> None:
    """Mathematical verification of all pipeline invariants."""
    sep = "═" * 62
    print(f"\n{sep}")
    print("  MATHEMATICAL VERIFICATION SUITE")
    print(sep)
    checks = 0

    # CHECK 1 — PSD
    min_ev = results["min_ev_clean"].min()
    ok = min_ev >= -1e-6
    print(f"\n[1] C_clean PSD: min_ev={min_ev:.3e} → {'✓ PASS' if ok else '✗ FAIL'}")
    checks += ok

    # CHECK 2 — PMFG planarity (edges = 3(N-2))
    expected = 3 * (N - 2)
    ok = (results["pmfg_n_edges"] == expected).all()
    print(f"[2] PMFG planarity: expected {expected} edges → "
          f"{'✓ PASS' if ok else '✗ FAIL'}")
    checks += ok

    # CHECK 3 — PMFG mixing time is finite
    ok = (results["mixing_time_pmfg"] < 1e7).all()
    print(f"[3] PMFG mixing time finite → {'✓ PASS' if ok else '✗ FAIL'}")
    checks += ok

    # CHECK 4 — Noise fraction statistics
    nf = results["noise_fraction"].mean()
    print(f"[4] MP noise fraction: {nf*100:.1f}% (informational)")
    checks += 1

    # CHECK 5 — Hypothesis: τ_PMFG decreases during known crashes
    # Check 2008 crisis
    pre_2008 = results.loc[:"2008-06-30", "mixing_time_pmfg"]
    crisis_2008 = results.loc["2008-09-01":"2009-03-31", "mixing_time_pmfg"]
    if len(crisis_2008) > 0 and len(pre_2008) > 0:
        ok = crisis_2008.median() < pre_2008.median()
        print(f"\n[5] Hypothesis (2008): τ_PMFG drops in crisis")
        print(f"    Pre-crisis median: {pre_2008.median():.2f}")
        print(f"    Crisis median:     {crisis_2008.median():.2f}")
        print(f"    → {'✓ CONFIRMED' if ok else '✗ NOT CONFIRMED'}")
        checks += ok
    else:
        print(f"\n[5] Hypothesis (2008): insufficient data to test")

    # Check COVID crash
    pre_covid = results.loc["2019-06-01":"2020-01-31", "mixing_time_pmfg"]
    covid = results.loc["2020-02-20":"2020-04-30", "mixing_time_pmfg"]
    if len(covid) > 0 and len(pre_covid) > 0:
        ok = covid.median() < pre_covid.median()
        print(f"\n[6] Hypothesis (COVID): τ_PMFG drops in crisis")
        print(f"    Pre-COVID median:  {pre_covid.median():.2f}")
        print(f"    COVID median:      {covid.median():.2f}")
        print(f"    → {'✓ CONFIRMED' if ok else '✗ NOT CONFIRMED'}")
        checks += ok
    else:
        print(f"\n[6] Hypothesis (COVID): insufficient data to test")

    print(f"\n{'─'*62}")
    print(f"  {checks} checks completed")
    print(f"{sep}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  7.  VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(
    results: pd.DataFrame,
    spx: pd.Series,
    N: int,
    bench: dict,
    output_path: str,
) -> None:
    """
    Five-panel publication figure:
      0: S&P 500 index
      1: PMFG Mixing Time τ (primary signal — DECREASES in crash)
      2: PMFG Spectral Gap Δ (INCREASES in crash)
      3: Correlation-based τ_C (reference — INCREASES in crash)
      4: Benchmark summary + PMFG computation time
    """
    BG    = "#0a0e14"
    PANEL = "#111720"
    GRID  = "#1e2a36"
    SPX_C = "#60a5fa"
    TAU_C = "#f87171"
    GAP_C = "#34d399"
    REF_C = "#a78bfa"
    EVT_C = "#fbbf24"
    SHADE = "#f87171"
    TEXT  = "#e2e8f0"
    SUB   = "#94a3b8"
    MA_C  = "#f1f5f9"
    TIME_C = "#fb923c"

    spx_a = spx.reindex(results.index, method="nearest")
    crash_labels = identify_crash_periods(spx_a)

    fig = plt.figure(figsize=(22, 20), facecolor=BG)
    fig.suptitle(
        "Phase 1  ·  Market Crash Detection via Classical Random Walker\n"
        "Marchenko-Pastur + PMFG Denoised Correlation Complexes  ·  "
        f"S&P 500 ({N} assets, 2005–2025)",
        fontsize=13, fontweight="bold", color=TEXT, y=0.99,
        fontfamily="monospace",
    )

    gs = gridspec.GridSpec(
        5, 1, hspace=0.08,
        top=0.96, bottom=0.05, left=0.07, right=0.76,
    )
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    # Crisis shading based on drawdown
    for ax in axes:
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=SUB, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(axis="y", color=GRID, lw=0.5, ls="--", alpha=0.7)
        ax.set_xlim(results.index[0], results.index[-1])

        # Shade crash periods
        crash_starts = []
        in_crash = False
        for i_d, (dt, val) in enumerate(crash_labels.items()):
            if val == 1 and not in_crash:
                crash_starts.append(dt)
                in_crash = True
            elif val == 0 and in_crash:
                ax.axvspan(crash_starts[-1], dt, alpha=0.10,
                           color=SHADE, zorder=0)
                in_crash = False
        if in_crash:
            ax.axvspan(crash_starts[-1], crash_labels.index[-1],
                       alpha=0.10, color=SHADE, zorder=0)

    # Event lines
    for dt_str, (lbl, pos) in CRISIS_EVENTS.items():
        dt = pd.Timestamp(dt_str)
        if results.index[0] <= dt <= results.index[-1]:
            for ax in axes:
                ax.axvline(dt, color=EVT_C, lw=0.6, ls=":", alpha=0.7)

    # Panel 0: SPX
    ax = axes[0]
    ax.plot(spx_a.index, spx_a.values, color=SPX_C, lw=1.4, alpha=0.9)
    ax.fill_between(spx_a.index, spx_a.values, spx_a.min(),
                    alpha=0.10, color=SPX_C)
    ax.set_ylabel("Index Level", color=TEXT, fontsize=9)
    ax.set_title("S&P 500 Index (Real Data)", color=SUB, fontsize=8.5,
                 loc="left", pad=3)

    # Panel 1: PMFG mixing time (PRIMARY — decreases in crash)
    ax = axes[1]
    tau_pmfg = results["mixing_time_pmfg"].clip(
        upper=results["mixing_time_pmfg"].quantile(0.995))
    ax.plot(tau_pmfg.index, tau_pmfg.values, color=TAU_C, lw=1.3, alpha=0.9,
            label="τ_PMFG (classical walker)")
    ma = tau_pmfg.rolling(5).mean()
    ax.plot(ma.index, ma.values, color=MA_C, lw=0.7, alpha=0.5, ls="--",
            label="5-window MA")
    ax.set_ylabel("Mixing Time τ", color=TEXT, fontsize=9)
    ax.set_title(
        "PMFG Mixing Time τ  [↓ = crash signal — walker mixes faster]",
        color=SUB, fontsize=8.5, loc="left", pad=3)
    ax.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, fontsize=7.5)

    # Panel 2: PMFG spectral gap
    ax = axes[2]
    gap = results["spectral_gap_pmfg"]
    ax.plot(gap.index, gap.values, color=GAP_C, lw=1.3, alpha=0.9,
            label="Δ_PMFG")
    ax.set_ylabel("Spectral Gap Δ", color=TEXT, fontsize=9)
    ax.set_title(
        "PMFG Spectral Gap Δ = 1-|λ₂(P)|  [↑ = crash — faster convergence]",
        color=SUB, fontsize=8.5, loc="left", pad=3)
    ax.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, fontsize=7.5)

    # Panel 3: Correlation τ_C (reference)
    ax = axes[3]
    tau_c = results["tau_C"].clip(upper=results["tau_C"].quantile(0.995))
    ax.plot(tau_c.index, tau_c.values, color=REF_C, lw=1.3, alpha=0.9,
            label="τ_C (correlation-based)")
    ax.set_ylabel("τ_C (ref)", color=TEXT, fontsize=9)
    ax.set_title(
        "Correlation Mixing Time τ_C  [↑ = crash — reference signal]",
        color=SUB, fontsize=8.5, loc="left", pad=3)
    ax.legend(loc="upper right", facecolor=PANEL, edgecolor=GRID,
              labelcolor=TEXT, fontsize=7.5)

    # Panel 4: PMFG computation time
    ax = axes[4]
    t_pmfg = results["pmfg_time_s"]
    ax.bar(t_pmfg.index, t_pmfg.values, width=15, color=TIME_C, alpha=0.7)
    ax.set_ylabel("PMFG Time (s)", color=TEXT, fontsize=9)
    ax.set_title(
        f"PMFG Computation Time per Window  "
        f"[avg={t_pmfg.mean():.1f}s, total={t_pmfg.sum():.0f}s]",
        color=SUB, fontsize=8.5, loc="left", pad=3)

    # X-axis formatting
    for i, ax in enumerate(axes):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right",
                 color=SUB, fontsize=7.5)
        if i < 4:
            ax.set_xticklabels([])

    # Sidebar with benchmarks
    sidebar = (
        "  ┌─────────────────────────────┐\n"
        "  │   PIPELINE SUMMARY          │\n"
        "  ├─────────────────────────────┤\n"
        f"  │  Assets (N):       {N:<8d} │\n"
        f"  │  Window (T):       {CFG['T_WINDOW']:<8d} │\n"
        f"  │  Step:             {CFG['STEP']:<8d} │\n"
        f"  │  Windows:          {len(results):<8d} │\n"
        "  ├─────────────────────────────┤\n"
        "  │   BENCHMARKS                │\n"
        "  ├─────────────────────────────┤\n"
        f"  │  Precision:  {bench.get('precision', 0):.3f}         │\n"
        f"  │  Recall:     {bench.get('recall', 0):.3f}         │\n"
        f"  │  F1 Score:   {bench.get('f1', 0):.3f}         │\n"
        f"  │  AUC-ROC:    {bench.get('auc_roc', 0):.3f}         │\n"
        f"  │  Lead Time:  {bench.get('lead_time_days', 0):.0f}d           │\n"
        "  ├─────────────────────────────┤\n"
        "  │   TIMING                    │\n"
        "  ├─────────────────────────────┤\n"
        f"  │  Avg PMFG:   {t_pmfg.mean():.1f}s          │\n"
        f"  │  Total PMFG: {t_pmfg.sum():.0f}s         │\n"
        "  └─────────────────────────────┘"
    )
    fig.text(
        0.78, 0.50, sidebar,
        fontsize=7.5, color=TEXT, fontfamily="monospace",
        va="center", ha="left",
        bbox=dict(boxstyle="round,pad=0.5", fc=PANEL, ec=GRID, alpha=0.95),
    )

    crisis_patch = mpatches.Patch(
        color=SHADE, alpha=0.35, label="Crash Periods (>15% drawdown)")
    fig.legend(
        handles=[crisis_patch], loc="lower center",
        bbox_to_anchor=(0.42, 0.002), facecolor=PANEL, edgecolor=GRID,
        labelcolor=TEXT, fontsize=8,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    print(f"[Plot] Saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sep = "═" * 62
    print(f"\n{sep}")
    print("  Phase 1: Market Crash Detection via Classical Random Walker")
    print("  Pipeline: S&P 500 → MP Denoise → PMFG → Random Walker")
    print(f"  Period: {CFG['START']} to {CFG['END']}")
    print(sep)

    total_t0 = time.time()

    # ── 1. Data ──────────────────────────────────────────────────────────────
    log_ret, spx = load_sp500_data(
        CFG["TICKERS"], CFG["START"], CFG["END"])

    N = log_ret.shape[1]
    Q = CFG["T_WINDOW"] / N
    print(f"\n[Config] N={N}, Q={CFG['T_WINDOW']}/{N}={Q:.4f}")
    print(f"[Config] MP λ+ = {(1 + np.sqrt(1/Q))**2:.4f}")

    # ── 2. Rolling pipeline ──────────────────────────────────────────────────
    results = rolling_analysis(log_ret, CFG["T_WINDOW"], CFG["STEP"])

    # ── 3. Verification ─────────────────────────────────────────────────────
    verify(results, N)

    # ── 4. Benchmarks ────────────────────────────────────────────────────────
    print("\n[Benchmarks] Computing crash prediction metrics...")

    # PMFG classical walker (primary — mixing time DECREASES in crash)
    bench_pmfg = compute_benchmarks(
        results, spx,
        mixing_time_col="mixing_time_pmfg",
        model_name="Classical Walker (PMFG)",
        direction="below",
    )

    # Correlation-based (reference — mixing time INCREASES in crash)
    bench_corr = compute_benchmarks(
        results, spx,
        mixing_time_col="tau_C",
        model_name="Correlation τ_C (ref)",
        direction="above",
    )

    # Placeholder row for quantum Szegedy walker
    bench_szegedy = dict(
        model="Quantum Szegedy Walker",
        precision=float('nan'), recall=float('nan'),
        f1=float('nan'), lead_time_days=float('nan'),
        auc_roc=float('nan'),
        threshold=float('nan'),
        n_alerts=0, n_crash_days=0, n_total_days=0,
    )

    all_benchmarks = [bench_pmfg, bench_corr, bench_szegedy]
    print_benchmark_table(all_benchmarks)

    # ── 5. Save outputs ─────────────────────────────────────────────────────
    results.to_csv(CFG["OUTPUT_CSV"])
    print(f"[CSV] Results → {CFG['OUTPUT_CSV']}")

    bench_df = pd.DataFrame(all_benchmarks)
    bench_df.to_csv(CFG["BENCH_CSV"], index=False)
    print(f"[CSV] Benchmarks → {CFG['BENCH_CSV']}")

    # ── 6. Plot ──────────────────────────────────────────────────────────────
    plot_results(results, spx, N, bench_pmfg, CFG["OUTPUT_PNG"])

    total_elapsed = time.time() - total_t0
    print(f"\n[Done] Phase 1 complete in {total_elapsed:.0f}s "
          f"({total_elapsed/60:.1f} min)\n")

    return results


if __name__ == "__main__":
    results = main()
