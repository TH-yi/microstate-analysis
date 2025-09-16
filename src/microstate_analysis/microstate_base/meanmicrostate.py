
import numpy as np
from scipy import stats  # kept for backward-compat imports; not strictly required
from scipy.optimize import linear_sum_assignment
from operator import itemgetter  # kept for backward-compat imports
import itertools  # kept for backward-compat imports


# ---- Optional GPU delegate (CuPy-based) ----
_MM_GPU_AVAILABLE = False
try:
    from microstate_analysis.microstate_base.meanmicrostate_gpu import MeanMicrostateGPU as _MeanMicrostateGPU
    _MM_GPU_AVAILABLE = True
except Exception:
    _MM_GPU_AVAILABLE = False


def _zero_mean_unit_norm(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """Zero-mean along axis and L2-normalize each row (axis=1) to unit norm."""
    x = x - np.mean(x, axis=axis, keepdims=True)
    denom = np.linalg.norm(x, axis=axis, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


class MeanMicrostate:
    """
    Mean microstate estimator with optional GPU acceleration.
    Public API mirrors the original implementation:
      - label_two_microstates(microstates, mean_microstates, polarity=True)
      - label_microstates(mul_microstates, mean_microstates, polarity=True)
      - reorder_microstates(mul_microstates, mean_microstates, polarity=True)
      - update_mean_microstates(label, sign, polarity=True)
      - mean_microstates(n_runs=10, maxiter=100)

    When constructed with use_gpu=True (and CuPy available), the methods delegate
    to MeanMicrostateGPU. Otherwise, CPU/NumPy implementation is used.
    """

    def __init__(self, data, n_k, n_ch, n_condition, use_gpu: bool = False):
        """
        Parameters
        ----------
        data : list/array-like of length n_condition
            Each item is an array of shape (n_k, n_ch) for that condition.
        n_k : int
            Number of microstates per condition.
        n_ch : int
            Number of channels.
        n_condition : int
            Number of conditions (or subjects/blocks) to aggregate.
        use_gpu : bool, optional
            If True and GPU implementation is available, delegate heavy math to GPU.
        """
        self.data = data
        self.n_k = int(n_k)
        self.n_ch = int(n_ch)
        self.n_condition = int(n_condition)

        # Build concatenated view for backward-compat (as in original code)
        data_concatenate = np.zeros((1, self.n_ch), dtype=float)
        for i in range(self.n_condition):
            data_concatenate = np.concatenate((data_concatenate, np.asarray(data[i])), axis=0)
        self.data_concatenate = data_concatenate[1:, :]

        # Optional GPU delegate
        self._use_gpu = bool(use_gpu and _MM_GPU_AVAILABLE)
        self._gpu = _MeanMicrostateGPU(data, self.n_k, self.n_ch, self.n_condition, use_gpu=True) if self._use_gpu else None

    # ---------------- CPU implementation ----------------

    def _corr_matrix(self, A: np.ndarray, B: np.ndarray, polarity: bool = True) -> np.ndarray:
        """Compute (n_k x n_k) spatial correlation matrix between two sets of maps."""
        A = _zero_mean_unit_norm(np.asarray(A), axis=1)
        B = _zero_mean_unit_norm(np.asarray(B), axis=1)
        C = A @ B.T  # cosine correlation due to unit norm
        if not polarity:
            C = np.abs(C)
        return C

    def label_two_microstates(self, microstates: np.ndarray, mean_microstates: np.ndarray, polarity: bool = True):
        """
        Align one set of microstates to a mean template using linear assignment.
        Returns:
            label: np.ndarray shape (n_k,), label[i] = assigned index in mean_microstates
            sign:  np.ndarray shape (n_k,), each in {+1, -1}, chosen to maximize similarity
            mean_similarity: float, mean of |corr| over matched pairs
            std_similarity:  float, std  of |corr| over matched pairs
        """
        if self._gpu is not None:
            return self._gpu.label_two_microstates(microstates, mean_microstates, polarity=polarity)

        C = self._corr_matrix(microstates, mean_microstates, polarity=True)  # keep signs for polarity
        row_ind, col_ind = linear_sum_assignment(-C)  # maximize correlation

        label = np.empty(self.n_k, dtype=np.int64)
        sign = np.ones(self.n_k, dtype=np.int8)
        sims = []

        for i, j in zip(row_ind, col_ind):
            val = C[i, j]
            label[i] = int(j)
            if polarity:
                sign[i] = 1 if val >= 0 else -1
                sims.append(abs(val))
            else:
                sign[i] = 1
                sims.append(abs(val))

        sims = np.asarray(sims, dtype=float)
        return label, sign, float(sims.mean()), float(sims.std())

    def label_microstates(self, mul_microstates, mean_microstates: np.ndarray, polarity: bool = True):
        """
        Align multiple sets of microstates to the current mean template.
        Returns:
            label_list: list of length n_condition, each array shape (n_k,)
            sign_list:  list of length n_condition, each array shape (n_k,)
            mean_similarity: float (mean |corr| over all matched pairs across conditions)
            std_similarity:  float
        """
        if self._gpu is not None:
            return self._gpu.label_microstates(mul_microstates, mean_microstates, polarity=polarity)

        label_list, sign_list, all_sims = [], [], []

        for cond in range(self.n_condition):
            lbl, sgn, m, _ = self.label_two_microstates(mul_microstates[cond], mean_microstates, polarity=polarity)
            label_list.append(lbl)
            sign_list.append(sgn)

            # accumulate similarities per matched pair for stats
            C = self._corr_matrix(mul_microstates[cond], mean_microstates, polarity=True)
            sims = [abs(C[i, lbl[i]]) for i in range(self.n_k)]
            all_sims.extend(sims)

        all_sims = np.asarray(all_sims, dtype=float)
        return label_list, sign_list, float(all_sims.mean()), float(all_sims.std())

    def reorder_microstates(self, mul_microstates, mean_microstates: np.ndarray, polarity: bool = True):
        """Alias to label_microstates to preserve original API semantics."""
        if self._gpu is not None:
            return self._gpu.reorder_microstates(mul_microstates, mean_microstates, polarity=polarity)
        return self.label_microstates(mul_microstates, mean_microstates, polarity=polarity)

    def update_mean_microstates(self, label, sign, polarity: bool = True):
        """
        Average aligned microstates across conditions into a new mean template.
        Returns:
            mean_microstate_updated: np.ndarray shape (n_k, n_ch)
        """
        if self._gpu is not None:
            return self._gpu.update_mean_microstates(label, sign, polarity=polarity)

        mean_out = np.zeros((self.n_k, self.n_ch), dtype=float)
        counts = np.zeros((self.n_k,), dtype=float)

        for cond in range(self.n_condition):
            X = _zero_mean_unit_norm(np.asarray(self.data[cond]), axis=1)
            for i in range(self.n_k):
                k = int(label[cond][i])
                s = int(sign[cond][i])
                mean_out[k, :] += s * X[i, :]
                counts[k] += 1.0

        counts[counts == 0] = 1.0
        mean_out = mean_out / counts[:, None]
        mean_out = _zero_mean_unit_norm(mean_out, axis=1)
        return mean_out

    def mean_microstates(self, n_runs: int = 10, maxiter: int = 100):
        """
        Multi-start iterative alignment to compute stable mean microstates.
        Returns:
            maps_best:   list-list shape (n_k, n_ch)
            label_best:  list length n_condition, each array shape (n_k,)
            mean_sim:    float
            std_sim:     float
        """
        if self._gpu is not None:
            return self._gpu.mean_microstates(n_runs=n_runs, maxiter=maxiter)

        best_mean, best_labels = None, None
        best_ms, best_ss = -np.inf, np.inf

        data_list = [np.asarray(d) for d in self.data]

        for _ in range(int(n_runs)):
            # initialize from a random condition
            init_idx = np.random.randint(0, self.n_condition)
            mean_maps = _zero_mean_unit_norm(np.asarray(self.data[init_idx]), axis=1)

            last_labels = None
            for _it in range(int(maxiter)):
                labels, signs, mean_sim, std_sim = self.reorder_microstates(data_list, mean_maps, polarity=True)
                mean_maps_updated = self.update_mean_microstates(labels, signs, polarity=True)

                # check convergence by labels stability
                if last_labels is not None:
                    unchanged = True
                    for a, b in zip(labels, last_labels):
                        if not np.array_equal(a, b):
                            unchanged = False
                            break
                    if unchanged:
                        break
                last_labels = [np.asarray(x).copy() for x in labels]
                mean_maps = mean_maps_updated

            # keep the best run (higher mean_sim; tie-breaker lower std_sim)
            if (mean_sim > best_ms) or (np.isclose(mean_sim, best_ms) and std_sim < best_ss):
                best_ms, best_ss = float(mean_sim), float(std_sim)
                best_labels = [np.asarray(x) for x in labels]
                best_mean = np.asarray(mean_maps)
        maps_py = [self._to_list(map) for map in best_mean]
        labels_py = [self._to_list(lbl) for lbl in best_labels]
        return maps_py, labels_py, float(best_ms), float(best_ss)


    # --- list/JSON-friendly path ---
    @staticmethod
    def _to_list(x):
        """Prefer list-like conversion for arrays, otherwise return as-is."""
        try:
            import cupy as cp
            if isinstance(x, cp.ndarray):
                x = cp.asnumpy(x)
        except Exception:
            pass
        try:
            import numpy as np
            return x.tolist() if isinstance(x, np.ndarray) else x
        except Exception:
            return x

