
import numpy as _np
try:
    import cupy as _cp
    _HAS_CUPY = True
except Exception:
    _cp = None
    _HAS_CUPY = False

from scipy.optimize import linear_sum_assignment as _hungarian


def _xp(use_gpu: bool):
    return _cp if (use_gpu and _HAS_CUPY) else _np


def _to_xp(x, xp):
    if xp is _np:
        if isinstance(x, _np.ndarray):
            return x
        return _cp.asnumpy(x) if (_HAS_CUPY and isinstance(x, _cp.ndarray)) else _np.asarray(x)
    else:
        return x if isinstance(x, _cp.ndarray) else _cp.asarray(x)


def _to_numpy(x):
    if _HAS_CUPY and isinstance(x, _cp.ndarray):
        return _cp.asnumpy(x)
    return _np.asarray(x)


def _zero_mean_unit_norm(x, xp, axis=1):
    x = x - xp.mean(x, axis=axis, keepdims=True)
    norm = xp.linalg.norm(x, axis=axis, keepdims=True)
    norm = xp.where(norm == 0, 1.0, norm)
    return x / norm


class MeanMicrostateGPU:
    """
    GPU-accelerated mean microstate alignment/averaging.
    Assumes input data is a list/array of shape [n_condition] where each item is (n_k, n_ch).
    """

    def __init__(self, data, n_k, n_ch, n_condition, use_gpu: bool = True):
        self.use_gpu = bool(use_gpu and _HAS_CUPY)
        self.xp = _xp(self.use_gpu)

        # normalize and store per-condition microstates
        self.n_k = int(n_k)
        self.n_ch = int(n_ch)
        self.n_condition = int(n_condition)

        # Ensure a canonical structure: list of (n_k, n_ch) arrays
        self.data = []
        for i in range(n_condition):
            arr = _to_xp(data[i], self.xp)
            arr = _zero_mean_unit_norm(arr, self.xp, axis=1)
            self.data.append(arr)

        # also keep concatenated NumPy for compatibility if needed
        self.data_concatenate = _to_numpy(self.xp.concatenate(self.data, axis=0))

    # ----------------- Core utilities -----------------
    def _correlation_matrix(self, A, B, polarity=True):
        """
        Compute (n_k x n_k) spatial correlation between two microstate sets A and B.
        A, B: (n_k, n_ch), assumed zero-mean unit-norm along channels.
        Returns xp array (n_k, n_k). If polarity=False, take absolute value.
        """
        xp = self.xp
        C = A @ B.T  # cosine correlation because of unit norm
        if not polarity:
            C = xp.abs(C)
        return C

    def _assign(self, corr):
        """
        Solve one-to-one assignment maximizing total correlation using Hungarian algorithm.
        corr: (n_k, n_k) xp array.
        Returns: perm indices array (len n_k), and signed indicator (+1/-1) computed by corr sign.
        """
        # Hungarian works on CPU; move matrix to NumPy, solve, and bring indices back.
        C_np = _to_numpy(corr)
        # maximize corr -> minimize -corr
        row_ind, col_ind = _hungarian(-C_np)
        return row_ind, col_ind

    def label_two_microstates(self, microstates, mean_microstates, polarity=True):
        """
        Given two sets of microstates (n_k, n_ch), return label & sign that align microstates to mean_microstates.
        Returns a tuple (label, sign, mean_similarity, std_similarity).
        """
        xp = self.xp
        A = _zero_mean_unit_norm(_to_xp(microstates, xp), xp, axis=1)
        B = _zero_mean_unit_norm(_to_xp(mean_microstates, xp), xp, axis=1)
        corr = self._correlation_matrix(A, B, polarity=True)  # compute signed corr
        r_ind, c_ind = self._assign(corr)
        # label[i] = assigned mean index for microstate i
        label = _np.empty(self.n_k, dtype=_np.int64)
        sign = _np.ones(self.n_k, dtype=_np.int8)

        # Collect per-pair similarity (abs corr if polarity False else corr with sign)
        sims = []
        for i, j in zip(r_ind, c_ind):
            val = float(_to_numpy(corr[i, j]))
            label[i] = int(j)
            if polarity:
                # choose sign to maximize similarity
                s = 1 if val >= 0 else -1
                sign[i] = s
                sims.append(abs(val))
            else:
                sign[i] = 1
                sims.append(abs(val))

        sims = _np.asarray(sims, dtype=_np.float64)
        return label, sign, float(sims.mean()), float(sims.std())

    def label_microstates(self, mul_microstates, mean_microstates, polarity=True):
        """
        Align multiple conditions against the current mean templates.
        mul_microstates: list/array of length n_condition, each (n_k, n_ch)
        Returns (label_list, sign_list, mean_similarity, std_similarity)
        where label_list[k] has shape (n_k,), sign_list[k] has shape (n_k,).
        """
        labels, signs, all_sims = [], [], []
        for i in range(self.n_condition):
            lbl, sgn, m, _ = self.label_two_microstates(mul_microstates[i], mean_microstates, polarity=polarity)
            labels.append(lbl)
            signs.append(sgn)
            # For aggregation, recompute per-pair sims
            A = _zero_mean_unit_norm(_to_xp(mul_microstates[i], self.xp), self.xp, axis=1)
            B = _zero_mean_unit_norm(_to_xp(mean_microstates, self.xp), self.xp, axis=1)
            corr = self._correlation_matrix(A, B, polarity=True)
            sims = [_to_numpy(corr[idx, jdx]) for idx, jdx in zip(range(self.n_k), lbl)]
            all_sims.extend([abs(float(v)) for v in sims])

        all_sims = _np.asarray(all_sims, dtype=_np.float64)
        return labels, signs, float(all_sims.mean()), float(all_sims.std())

    def reorder_microstates(self, mul_microstates, mean_microstates, polarity=True):
        """Wrapper to match MeanMicrostate API. Returns (labels, signs, mean_sim, std_sim)."""
        return self.label_microstates(mul_microstates, mean_microstates, polarity=polarity)

    def update_mean_microstates(self, label, sign, polarity=True):
        """
        Update mean microstates by averaging aligned inputs with sign correction.
        label: list of length n_condition; each array shape (n_k,)
        sign:  list of length n_condition; each array shape (n_k,)
        """
        xp = self.xp
        mean_out = xp.zeros((self.n_k, self.n_ch), dtype=_to_xp(_np.zeros(1), xp).dtype)
        counts = xp.zeros((self.n_k,), dtype=_to_xp(_np.zeros(1), xp).dtype)

        for cond in range(self.n_condition):
            X = _to_xp(self.data[cond], xp)
            for i in range(self.n_k):
                k = int(label[cond][i])
                s = int(sign[cond][i])
                mean_out[k, :] += s * X[i, :]
                counts[k] += 1

        counts = xp.where(counts == 0, 1.0, counts)
        mean_out = mean_out / counts[:, None]
        mean_out = _zero_mean_unit_norm(mean_out, xp, axis=1)
        return _to_numpy(mean_out)

    def mean_microstates(self, n_runs=10, maxiter=100):
        """
        Run multi-start alignment to find stable mean microstates.
        Returns (maps_best, label_best, mean_similarity_best, std_similarity_best)
        where label_best is a list length n_condition.
        """
        best_mean, best_label, best_ms, best_ss = None, None, -_np.inf, _np.inf

        # prepare data bundle
        data_list = [ _to_xp(x, self.xp) for x in self.data ]

        for _ in range(int(n_runs)):
            # init from a random condition or the average of all first maps
            init_idx = _np.random.randint(0, self.n_condition)
            mean_maps = _zero_mean_unit_norm(_to_xp(self.data[init_idx], self.xp), self.xp, axis=1)

            last_labels = None
            for _it in range(int(maxiter)):
                labels, signs, mean_sim, std_sim = self.reorder_microstates(data_list, mean_maps, polarity=True)
                mean_maps_new = _to_xp(self.update_mean_microstates(labels, signs, polarity=True), self.xp)

                # check convergence: labels unchanged
                if last_labels is not None:
                    unchanged = True
                    for a, b in zip(labels, last_labels):
                        if not _np.array_equal(_to_numpy(a), _to_numpy(b)):
                            unchanged = False
                            break
                    if unchanged:
                        break
                last_labels = labels
                mean_maps = mean_maps_new

            # evaluate and keep best
            if mean_sim > best_ms or (mean_sim == best_ms and std_sim < best_ss):
                best_ms, best_ss = mean_sim, std_sim
                best_label = [ _to_numpy(x) for x in labels ]
                best_mean  = _to_numpy(mean_maps)

        return best_mean, best_label, float(best_ms), float(best_ss)
