import numpy as _np

try:
    import cupy as _cp

    _HAS_CUPY = True
except Exception as e:  # pragma: no cover
    print(f"Failed import cupy: {e}")
    _cp = None
    _HAS_CUPY = False

from scipy.signal import find_peaks as _np_find_peaks
import os, sys, glob
import ctypes
from pathlib import Path
from typing import Iterable, Set


def _xp(use_gpu: bool):
    """Return numerical backend (cupy or numpy)."""
    if use_gpu and _HAS_CUPY:
        return _cp
    return _np


def _to_xp(x, xp):
    """Move array to target backend (NumPy/CuPy)."""
    if xp is _np:
        if isinstance(x, _np.ndarray):
            return x
        # cupy to numpy
        return _cp.asnumpy(x) if _HAS_CUPY and isinstance(x, _cp.ndarray) else _np.asarray(x)
    else:
        # to CuPy
        return x if (xp is _cp and isinstance(x, _cp.ndarray)) else _cp.asarray(x)


def _to_numpy(x):
    """Ensure NumPy array output."""
    if _HAS_CUPY and isinstance(x, _cp.ndarray):
        return _cp.asnumpy(x)
    return _np.asarray(x)


def zero_mean(x, axis=1, xp=_np):
    """Zero-mean along given axis. Compatible with NumPy/CuPy backends."""
    mean = xp.mean(x, axis=axis, keepdims=True)
    return x - mean


def _ensure_cuda_dlls(preload: bool = True) -> None:
    """
    Ensure CUDA/NVIDIA runtime DLLs shipped via pip wheels are discoverable on Windows.

    Why this is needed:
      - On Windows with Python 3.8+, native DLL lookup does NOT automatically include
        site-packages subfolders. NVIDIA CUDA wheels (e.g., nvidia-cublas-cu12,
        nvidia-cuda-nvrtc-cu12, etc.) place DLLs under:
          <venv>\Lib\site-packages\nvidia\<pkg>\bin\*.dll
      - If these folders are not added to the DLL search path BEFORE importing GPU
        libraries (CuPy, cuBLAS, NVRTC), `ImportError: DLL load failed` will occur.
      - In multiprocessing with "spawn", child processes may import this module before
        any pool initializer runs. Therefore this function must be called at module
        import time, i.e., before `import cupy`.

    What it does:
      1) Locate all `<site-packages>/nvidia/**/bin/` directories that contain `*.dll`.
      2) Add each directory to the Windows DLL search path via `os.add_dll_directory`,
         and prepend to process PATH as a fallback.
      3) If `CUPY_NVRTC_SONAME` is not set, auto-pick a detected `nvrtc64_*.dll`
         so CuPy can load the correct NVRTC SONAME even when versions differ (12.0/12.4/12.6).
      4) (Optional) Preload some critical DLLs early to fail fast with a clearer error
         if anything is still missing: cublas, cublasLt, cudart, and the chosen nvrtc.

    Parameters
    ----------
    preload : bool
        If True, attempt to `ctypes.WinDLL(...)` a few critical CUDA DLLs right away
        to expose missing dependencies early. Default: True.

    Notes
    -----
    - This is a no-op on non-Windows platforms.
    - Keep this function at the top of GPU modules and call it BEFORE `import cupy`.
    - For multiprocessing on Windows, still consider passing a similar path-setup
      function as the Pool `initializer` for double safety.
    """
    # Only needed on Windows
    if not sys.platform.startswith("win"):
        return

    # Locate NVIDIA CUDA wheels' bin directories inside the active interpreter prefix
    site_base = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not site_base.exists():
        # Nothing to add; likely using system-wide CUDA or not on a venv.
        return

    # Collect every .../nvidia/**/bin/*.dll directory
    dll_glob = str(site_base / "**" / "bin" / "*.dll")
    bin_dirs: Set[str] = {str(Path(p).parent) for p in glob.glob(dll_glob, recursive=True)}

    # Add discovered bin dirs to the DLL search path first, then PATH as a fallback.
    for d in bin_dirs:
        # Python 3.8+ on Windows: preferred API for per-process DLL directories
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(d)
        # Also prepend to PATH so child tools inheriting env can resolve these DLLs
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")

    # If user hasn't pinned NVRTC SONAME, try to auto-select one we found.
    os.environ.setdefault(
        "CUPY_NVRTC_SONAME",
        _pick_first_nvrtc_name(bin_dirs) or os.environ.get("CUPY_NVRTC_SONAME", "")
    )

    if preload:
        # Preload a few critical DLLs to surface missing MSVC runtime or other deps early.
        # Order is not strict; failures will raise OSError with a specific message.
        to_try = [
            "cublas64_12.dll",
            "cublasLt64_12.dll",
            "cudart64_12.dll",
        ]
        nvrtc_soname = os.environ.get("CUPY_NVRTC_SONAME")
        if nvrtc_soname:
            to_try.insert(0, nvrtc_soname)

        for name in to_try:
            try:
                ctypes.WinDLL(name)
            except OSError as e:
                # Do not hard-crash here; let the caller decide.
                # You can replace print with your logger if available.
                print(f"[WARN] Preload failed for {name}: {e}")


def _pick_first_nvrtc_name(bin_dirs: Iterable[str]) -> str | None:
    """
    Return the first `nvrtc64_*.dll` file name found under provided bin directories.

    Examples of valid names:
      - nvrtc64_120_0.dll   (CUDA 12.0)
      - nvrtc64_124_0.dll   (CUDA 12.4)
      - nvrtc64_126_0.dll   (CUDA 12.6)

    Parameters
    ----------
    bin_dirs : Iterable[str]
        Candidate bin directories to scan.

    Returns
    -------
    str | None
        The file name (not full path) of the first matching NVRTC DLL, or None if not found.
    """
    for d in bin_dirs:
        matches = list(Path(d).glob("nvrtc64_*.dll"))
        if matches:
            return matches[0].name
    return None


#_ensure_cuda_dlls()


def _as_xp_index(idx, xp):
    """
    Convert `idx` into a 1-D integer array that lives on `xp` (numpy or cupy).
    - If xp is cupy and idx is numpy/list: move to device.
    - If xp is numpy and idx is cupy: bring to host.
    Ensures dtype is integer and shape is (N,).
    """
    # Move to the right backend
    if xp is _np:
        # Bring from CuPy to NumPy if needed
        if _cp is not None and isinstance(idx, _cp.ndarray):
            idx = _cp.asnumpy(idx)
        idx = _np.asarray(idx)
    else:
        # xp is CuPy: push to device if needed
        if not (hasattr(idx, "ndim") and getattr(idx, "__array_priority__", None) == 1000):
            # Heuristic: non-cupy → to device
            idx = xp.asarray(idx)

    # Make it 1-D integer
    if idx.ndim != 1:
        idx = idx.ravel()
    if idx.dtype.kind not in "iu":
        idx = idx.astype(xp.int64, copy=False)
    return idx


def _ensure_xp_array(a, xp, dtype=None, copy=False):
    """
    Ensure `a` is an array on backend `xp` with optional dtype.
    - If xp is cupy and `a` is numpy: moves to device.
    - If xp is numpy and `a` is cupy: brings to host.
    """
    if xp is _np:
        if _cp is not None and isinstance(a, _cp.ndarray):
            a = _cp.asnumpy(a)
        a = _np.asarray(a)
    else:
        a = xp.asarray(a)  # cupy: host→device if needed
    if dtype is not None and a.dtype != dtype:
        a = a.astype(dtype, copy=copy)
    return a


class MicrostateGPU:
    """
    GPU-accelerated variant of Microstate clustering (K-means-like) using CuPy.
    Falls back to NumPy if GPU is unavailable.
    API mirrors the original Microstate where possible.
    """

    def __init__(self, data, use_gpu: bool = True):
        # choose backend
        self.use_gpu = bool(use_gpu and _HAS_CUPY)
        self.xp = _xp(self.use_gpu)

        # keep a NumPy copy for CPU-only operations like find_peaks
        self.data_np = _np.asarray(data, dtype=_np.float32)  # shape (n_t, n_ch) after transpose to match original
        self.data_np = zero_mean(self.data_np, 1, _np).astype(_cp.float32 if self.use_gpu else _np.float32, copy=False)

        # GPU (or CPU) working copy
        self.data = _to_xp(self.data_np, self.xp)

        self.n_t = int(self.data.shape[0])
        self.n_ch = int(self.data.shape[1])

        self.gfp = None  # NumPy 1D
        self.peaks = None  # NumPy 1D

        # For opt_microstate outputs (kept NumPy-compatible for downstream code)
        self.cv = None
        self.gev = None
        self.maps = None
        self.label = None
        self.opt_k = -1
        self.opt_k_index = -1

        self.cv_list = []
        self.gev_list = []
        self.maps_list = []
        self.label_list = []

    # ---- static helpers (backend-aware) ----
    @staticmethod
    def normalization(v, xp, axis=1):
        norm = xp.linalg.norm(v, axis=axis, keepdims=True)
        # guard against zero division
        norm = xp.where(norm == 0, 1.0, norm)
        return v / norm

    @staticmethod
    def orthogonal_dist(v, eeg_map, xp):
        # sum(v^2) - sum( (eeg_map • v)^2 ), row-wise
        return xp.sum(v ** 2) - xp.sum(xp.sum((eeg_map * v), axis=1) ** 2)

    @staticmethod
    def max_evec(v, xp, axis):
        # principal eigenvector of (v^T v)
        data = v.T @ v
        # eigh is more stable/symmetric than eig
        evals, evecs = xp.linalg.eigh(data)
        # pick eigenvector with largest absolute eigenvalue
        idx = xp.argmax(xp.abs(evals))
        c = xp.real(evecs[:, idx])
        c = MicrostateGPU.normalization(c, xp, axis)
        return c

    @staticmethod
    def spatial_correlation(averaged_v1, averaged_v2, std_v1, std_v2, n_ch, xp):
        # averaged_v1: (N, C), averaged_v2: (M, C). We need (N, M)
        # correlation = (v1 @ v2^T) / (n_ch * outer(std_v1, std_v2))
        return (averaged_v1 @ averaged_v2) / (n_ch * xp.outer(std_v1, std_v2))

    @staticmethod
    def global_explained_variance(n_maps, correlation, label, data_std, xp):
        gev = xp.zeros(n_maps)
        denom = xp.sum(data_std[label != -1] ** 2)
        for k in range(n_maps):
            mask = (label == k)
            if xp.any(mask):
                gev[k] = xp.sum((data_std[mask] ** 2) * (correlation[mask, k] ** 2)) / denom
            else:
                gev[k] = 0.0
        return gev

    def cross_validation(self, var, n_maps):
        return var * (self.n_ch - 1) ** 2 / (self.n_ch - n_maps - 1.) ** 2

    def variance(self, label, maps, data):
        xp = self.xp
        var = xp.sum(data ** 2) - xp.sum(xp.sum(maps[label, :] * data, axis=1) ** 2)
        var = var / (self.n_t * (self.n_ch - 1))
        return var

    def gfp_peaks(self, distance=10, n_std=3):
        gfp_np = self.data_np.std(axis=1)  # CPU NumPy
        lo = float(gfp_np.mean() - n_std * gfp_np.std())
        hi = float(gfp_np.mean() + n_std * gfp_np.std())
        peaks_np, _ = _np_find_peaks(gfp_np, distance=distance, height=(lo, hi))
        self.gfp = gfp_np
        self.peaks = peaks_np

    # ---- core KMeans-like loop on GPU/CPU backend ----
    def kmeans_modified(self, data, data_std, n_runs=100, n_maps=4,
                        maxerr=1e-6, maxiter=1000, polarity=False):
        xp = self.xp
        n_gfp = data_std.shape[0]

        best_cv = float("inf")
        best_result = None

        for run in range(n_runs):
            # if run == 0 or run == n_runs - 1:
            #     print(f"[GPU kmeans] run {run + 1}/{n_runs}")

            # --- init maps ---
            rndi = xp.random.permutation(n_gfp)[:n_maps]
            maps = MicrostateGPU.normalization(data[rndi, :], xp, axis=1)

            var0, var1 = 1.0, 0.0
            n_iter = 0

            while (abs(var0 - var1) / var0 > maxerr) and (n_iter < maxiter):
                n_iter += 1

                # --- assignment step ---
                label = xp.argmax((data @ maps.T) ** 2, axis=1)

                # --- update step ---
                for k in range(n_maps):
                    mask = (label == k)
                    if not xp.any(mask):
                        continue
                    Xk = data[mask, :]
                    cov = Xk.T @ Xk
                    # use power iteration instead of eigh
                    v = xp.random.randn(self.n_ch, 1).astype(self.data.dtype)
                    for _ in range(10):  # 10 iters usually enough
                        v = cov @ v
                        v /= xp.linalg.norm(v)
                    maps[k, :] = v.ravel()

                var1 = var0
                var0 = MicrostateGPU.orthogonal_dist(data, maps[label, :], xp) / (n_gfp * (self.n_ch - 1))

            # --- evaluate ---
            label_opt, corr, cv, gev = self.optimize_k(maps=maps, data=data, data_std=data_std, polarity=polarity)
            if cv < best_cv:
                best_cv = cv
                best_result = (cv, gev, maps, label_opt)

        # return only once (convert at the very end)
        cv, gev, maps, label = best_result
        return _to_numpy(cv), _to_numpy(gev), _to_numpy(maps), _to_numpy(label)

    def optimize_k(self, maps, data=None, data_std=None, polarity=False):
        xp = self.xp
        if data is None or data_std is None:
            data = self.data
            data_std = _to_xp(self.gfp, xp)

        n_maps = int(maps.shape[0])
        maps_zn = zero_mean(maps, 1, xp)
        # correlation over all time points
        correlation = MicrostateGPU.spatial_correlation(
            data, maps_zn.T, data_std, maps.std(axis=1), self.n_ch, xp
        )
        if not polarity:
            correlation = xp.abs(correlation)

        label = xp.argmax(correlation, axis=1)
        var = self.variance(label=label, maps=maps, data=data)
        cv = self.cross_validation(var, n_maps)
        gev = MicrostateGPU.global_explained_variance(n_maps, correlation, label, data_std, xp)

        # return NumPy to keep parity with CPU code elsewhere
        return _to_numpy(label), _to_numpy(correlation), float(_to_numpy(cv)), _to_numpy(gev)

    def opt_microstate(self, min_maps=2, max_maps=10, distance=10, n_std=3, n_runs=10, maxerr=1e-6, maxiter=1000,
                       polarity=False, peaks_only=True, method='kmeans_modified', opt_k=None):
        # Find peaks on CPU
        self.gfp_peaks(distance=distance, n_std=n_std)

        # Ensure peaks lives on xp as 1-D integer indices (no implicit host/device conversions)
        has_peaks = (self.peaks is not None)
        if has_peaks:
            # .size is a Python int for both numpy/cupy; cheap and safe
            try:
                peaks_size = int(getattr(self.peaks, "size", len(self.peaks)))
            except TypeError:
                # In case len(self.peaks) is not supported (unlikely), fall back to xp.asarray
                peaks_size = int(_as_xp_index(self.peaks, self.xp).size)
        else:
            peaks_size = 0
        if peaks_only and has_peaks and peaks_size > 0:
            peaks_xp = _as_xp_index(self.peaks, self.xp)  # 1-D int array on xp
            # Slice directly on the device/XP to stay "all-xp"
            # data shape is (T, C); peaks indexes time axis (axis=0)
            temp_data = self.data[peaks_xp, :]  # xp array
            temp_data_std = self.gfp[peaks_xp]  # xp array of shape (T_sel,)
            # Max maps is bounded by number of selected rows (timepoints)
            temp_max_maps = int(min(int(temp_data.shape[0]), max_maps))
        else:
            # No peak sub-sampling: just use full xp arrays
            temp_data = self.data  # xp array
            temp_data_std = self.gfp  # xp array
            temp_max_maps = int(min(int(temp_data.shape[0]), max_maps))

        cv_list, gev_list, maps_list = [], [], []

        if method == 'kmeans_modified':
            best_cv = _np.inf
            best = None
            for n_maps in range(min_maps, int(temp_max_maps) + 1):
                cv, gev, maps, label = self.kmeans_modified(
                    data=temp_data, data_std=temp_data_std, n_runs=n_runs,
                    n_maps=int(n_maps), maxerr=maxerr, maxiter=maxiter, polarity=polarity
                )
                cv_list.append(cv)
                gev_list.append(gev)
                maps_list.append(maps)
                if cv < best_cv:
                    best_cv = cv
                    best = (cv, gev, maps, label, n_maps)
                del maps, label, gev, cv
                if self.use_gpu:
                    _cp.get_default_memory_pool().free_all_blocks()
                    _cp.get_default_pinned_memory_pool().free_all_blocks()
        else:
            raise NotImplementedError("GPU AAHC is not implemented in this module.")

        self.cv, self.gev, self.maps, self.label, k_star = best
        self.opt_k = int(k_star)
        self.opt_k_index = int(opt_k - min_maps)
        # keep full lists JSON-serializable
        self.cv_list = [float(c) for c in cv_list]
        self.gev_list = [g.tolist() for g in gev_list]
        self.maps_list = [m.tolist() for m in maps_list]
        # self.label_list = [l.tolist() for l in label_list]
