# extend module for microstate.py
import math
import numpy as np
from typing import Iterable, Any, Optional, List, Tuple, Dict


# ---------------- Label-sequence metrics (transition / entropy / Hurst) ----------------

def _as_labels(labels: Iterable[Any]) -> np.ndarray:
    """Return labels as a 1-D numpy array; allow ints/strs."""
    lab = np.asarray(list(labels))
    return lab if lab.ndim == 1 else lab.ravel()


def _unique_states(lab: np.ndarray, states: Optional[List[Any]] = None) -> List[Any]:
    """Order-preserving unique states; allow user-specified ordering via `states`."""
    if states is not None and len(states) > 0:
        return list(states)
    # preserve first-appearance order
    seen, order = set(), []
    for x in lab.tolist():
        if x not in seen:
            seen.add(x)
            order.append(x)
    return order


def transition_matrix(labels: Iterable[Any],
                      states: Optional[List[Any]] = None) -> Tuple[np.ndarray, List[Any]]:
    """First-order Markov transition matrix (row-normalized)."""
    lab = _as_labels(labels)
    st = _unique_states(lab, states)
    idx = {s: i for i, s in enumerate(st)}
    L = np.vectorize(lambda x: idx[x])(lab) if lab.size else np.array([], dtype=int)

    k = len(st)
    P = np.zeros((k, k), dtype=float)
    if L.size >= 2:
        for a, b in zip(L[:-1], L[1:]):
            P[a, b] += 1.0
        row = P.sum(axis=1, keepdims=True)
        # normalize rows (leave all-zero rows as zeros)
        np.divide(P, row, out=P, where=row > 0)
    return P, st


def transition_frequency(labels: Iterable[Any],
                         sfreq: Optional[float] = None) -> float:
    """
    Number of state switches per unit time.
    - Without sfreq: return fraction of switch events per sample (in [0,1]).
    - With sfreq:    return switches per second (Hz).
    """
    lab = _as_labels(labels)
    if lab.size <= 1:
        return 0.0
    n_switch = float(np.sum(lab[1:] != lab[:-1]))
    if sfreq and sfreq > 0:
        duration_sec = len(lab) / float(sfreq)
        return n_switch / duration_sec if duration_sec > 0 else np.nan
    else:
        return n_switch / (len(lab) - 1)


def microstate_coverage_from_labels(labels: Iterable[Any],
                                    states: Optional[List[Any]] = None) -> Dict[str, float]:
    """Coverage per state: fraction of time spent in each state."""
    lab = _as_labels(labels)
    st = _unique_states(lab, states)
    n = len(lab)
    out: Dict[str, float] = {}
    for s in st:
        out[f"coverage_{s}"] = (np.count_nonzero(lab == s) / n) if n > 0 else 0.0
    return out


def microstate_duration_from_labels(labels: Iterable[Any],
                                    sfreq: Optional[float] = None,
                                    states: Optional[List[Any]] = None,
                                    include_std: bool = True,
                                    include_median: bool = True) -> Dict[str, float]:
    """
    Mean duration (and optional std/median) per state computed from contiguous runs.
    If sfreq is given, durations are returned in seconds; otherwise in samples.
    Keys:
      duration_mean_<s>, (optional) duration_std_<s>, duration_median_<s>, segments_<s>
    """
    lab = _as_labels(labels)
    st = _unique_states(lab, states)
    out: Dict[str, float] = {}

    if lab.size == 0:
        for s in st:
            out[f"duration_mean_{s}"] = np.nan
            if include_std: out[f"duration_std_{s}"] = np.nan
            if include_median: out[f"duration_median_{s}"] = np.nan
            out[f"segments_{s}"] = 0
        return out

    # run-length encoding to find contiguous segments
    changes = np.diff(lab, prepend=lab[:1])
    boundaries = np.flatnonzero(changes != 0)
    # boundaries marks the first index of each new run
    run_starts = boundaries
    run_ends = np.append(boundaries[1:], lab.size)  # exclusive

    # collect durations per state
    per_state_durs = {s: [] for s in st}
    for a, b in zip(run_starts, run_ends):
        s = lab[a]
        per_state_durs[s].append(b - a)

    # aggregate
    for s in st:
        arr = np.asarray(per_state_durs[s], dtype=float)
        if sfreq and sfreq > 0:
            arr = arr / float(sfreq)
        if arr.size == 0:
            out[f"duration_mean_{s}"] = np.nan
            if include_std: out[f"duration_std_{s}"] = np.nan
            if include_median: out[f"duration_median_{s}"] = np.nan
            out[f"segments_{s}"] = 0
        else:
            out[f"duration_mean_{s}"] = float(np.mean(arr))
            if include_std: out[f"duration_std_{s}"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            if include_median: out[f"duration_median_{s}"] = float(np.median(arr))
            out[f"segments_{s}"] = int(arr.size)

    return out


def entropy_rate(labels: Iterable[Any],
                 states: Optional[List[Any]] = None,
                 log_base: float = math.e) -> float:
    """Entropy rate H = - sum_i pi_i * sum_j P_ij * log(P_ij) with empirical pi and row-stochastic P."""
    lab = _as_labels(labels)
    if lab.size <= 1:
        return 0.0
    P, st = transition_matrix(lab, states)
    cov = microstate_coverage_from_labels(lab, st)  # empirical stationary distribution
    pi = np.array([cov[f"coverage_{s}"] for s in st], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0, np.log(P) / np.log(log_base), 0.0)
        H = -float((pi[:, None] * P * logP).sum())
    return H


def hurst_rs(series: np.ndarray,
             min_chunk: int = 16,
             max_chunk: Optional[int] = None) -> float:
    """R/S estimator for Hurst exponent on a binary/real-valued series."""
    x = np.asarray(series, dtype=float)
    N = len(x)
    if N < min_chunk * 4:
        return np.nan
    if max_chunk is None:
        max_chunk = max(min(N // 4, 1024), min_chunk)

    sizes, rs_vals = [], []
    scale = min_chunk
    while scale <= max_chunk:
        n_blocks = N // scale
        if n_blocks < 4:
            break
        rs_per_block = []
        for b in range(n_blocks):
            seg = x[b * scale:(b + 1) * scale]
            if seg.size < 2:
                continue
            y = seg - seg.mean()
            z = np.cumsum(y)
            R = z.max() - z.min()
            S = seg.std(ddof=1)
            if S > 0:
                rs_per_block.append(R / S)
        if rs_per_block:
            sizes.append(scale)
            rs_vals.append(np.mean(rs_per_block))
        scale *= 2

    if len(sizes) < 2:
        return np.nan
    sizes = np.array(sizes, dtype=float)
    rs_vals = np.array(rs_vals, dtype=float)
    H = np.polyfit(np.log(sizes), np.log(rs_vals), 1)[0]
    return float(H)


def hurst_from_labels(labels: Iterable[Any],
                      states: Optional[List[Any]] = None,
                      aggregate: bool = True) -> Dict[str, float]:
    """Hurst per state (from 0/1 indicator sequences); optionally return mean only."""
    lab = _as_labels(labels)
    st = _unique_states(lab, states)
    Hs: Dict[str, float] = {}
    for s in st:
        x = (lab == s).astype(float)
        Hs[f"hurst_{s}"] = hurst_rs(x)
    if aggregate:
        vals = [v for v in Hs.values() if not np.isnan(v)]
        Hs["hurst_mean"] = float(np.mean(vals)) if len(vals) else np.nan
    return Hs


def metrics_for_labels(labels: Iterable[Any],
                       sfreq: Optional[float] = None,
                       states: Optional[List[Any]] = None,
                       log_base: float = math.e,
                       include_duration: bool = True,
                       include_state_hurst: bool = True) -> Dict[str, float]:
    """
    One-shot metrics from a single label sequence.
    Returns a flat dict containing:
      - coverage_<s>
      - (optional) duration_mean/std/median_<s>, segments_<s>
      - transition_frequency
      - entropy_rate
      - hurst_<s> (or hurst_mean if aggregate)
    """
    lab = _as_labels(labels)
    st = _unique_states(lab, states)
    out: Dict[str, float] = {}

    out.update(microstate_coverage_from_labels(lab, st))

    if include_duration:
        out.update(microstate_duration_from_labels(lab, sfreq=sfreq, states=st,
                                                   include_std=True, include_median=True))

    out["transition_frequency"] = transition_frequency(lab, sfreq=sfreq)
    out["entropy_rate"] = entropy_rate(lab, states=st, log_base=log_base)

    hurst_dict = hurst_from_labels(lab, states=st, aggregate=not include_state_hurst)
    out.update(hurst_dict)
    if not include_state_hurst and "hurst_mean" not in out:
        vals = [v for k, v in hurst_dict.items() if k.startswith("hurst_") and not np.isnan(v)]
        out["hurst_mean"] = float(np.mean(vals)) if len(vals) else np.nan

    return out
