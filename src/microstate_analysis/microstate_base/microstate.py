from scipy import signal

from microstate_analysis.eeg_tool.math_utilis import zero_mean
from microstate_analysis.microstate_metrics.microstate_label_squence_metrics import *
from typing import Set

# ---- GPU optional backend (CuPy) adapter ----
_MS_GPU_AVAILABLE = False
try:
    from microstate_analysis.microstate_base.microstate_gpu import MicrostateGPU as _MicrostateGPU
    import os, sys, glob
    from pathlib import Path

    _MS_GPU_AVAILABLE = True

except Exception as e:
    print(f"Error import microstate_gpu module: {e}")
    _MS_GPU_AVAILABLE = False


class Microstate:
    def __init__(self, data, use_gpu: bool = False, reconstruct_pca_maps_info = None):
        # reconstruct pca maps info only for pca_individual_run reconstruct original dimensions maps
        self.reconstruct_pca_maps_info = reconstruct_pca_maps_info
        self.maps_list_original_dim = [] # serve for pca_individual_run
        self.label_list_original_dim = [] # serve for pca_individual_run
        # GPU delegation setup
        self._use_gpu = bool(use_gpu and _MS_GPU_AVAILABLE)
        if self._use_gpu:
            self._gpu = _MicrostateGPU(data, use_gpu=True)
        else:
            self.normalize_data(data)
            self._gpu = None
        self.gfp = None
        self.peaks = None

        self.cv = None
        self.maps = None
        self.label = None

        # All data
        self.label = None
        self.gev = None
        self.correlation = None
        self.duration = None
        self.coverage = None
        self.opt_k = -1
        self.opt_k_index = -1

        # All data from min_maps to max_maps
        self.cv_list = []
        self.gev_list = []
        self.maps_list = []
        self.label_list = []

    def normalize_data(self, data):
        self.data = data.T
        self.data = zero_mean(self.data, 1)
        self.n_t = self.data.shape[0]
        self.n_ch = self.data.shape[1]

    @staticmethod
    def orthogonal_dist(v, eeg_map):
        return np.sum(v ** 2) - np.sum(np.sum((eeg_map * v), axis=1) ** 2)

    @staticmethod
    def normalization(v, axis=1):
        return v / np.linalg.norm(v, axis=axis, keepdims=True)

    @staticmethod
    def global_dissimilarity(spatial_correlation):
        return np.sqrt(2 * (1 - spatial_correlation))

    def global_explained_variance(self, n_maps, correlation, label, data_std):
        gev = np.zeros(n_maps)
        for k in range(n_maps):
            gev[k] = np.sum(data_std[label == k] ** 2 * correlation[label == k, k] ** 2) / np.sum(
                data_std[label != -1] ** 2)
        return gev

    @staticmethod
    def global_explained_variance_sum(data, maps, distance=10, n_std=3, polarity=False):
        n_maps = maps.shape[0]
        n_ch = maps.shape[1]
        gev_peaks = np.zeros(n_maps)
        gev_raw = np.zeros(n_maps)
        gfp = data.std(axis=1)
        peaks, _ = signal.find_peaks(gfp, distance=distance,
                                     height=(gfp.mean() - n_std * gfp.std(), gfp.mean() + n_std * gfp.std()))
        correlation_peaks = Microstate.spatial_correlation(data[peaks], zero_mean(maps, 1).T, gfp[peaks],
                                                           maps.std(axis=1), n_ch)
        correlation_raw = Microstate.spatial_correlation(data, zero_mean(maps, 1).T, gfp, maps.std(axis=1), n_ch)
        correlation_peaks = correlation_peaks if polarity else abs(correlation_peaks)
        correlation_raw = correlation_raw if polarity else abs(correlation_raw)
        label_peaks = np.argmax(correlation_peaks, axis=1)
        label_raw = np.argmax(correlation_raw, axis=1)
        for k in range(n_maps):
            gev_peaks[k] = np.sum(
                gfp[peaks][label_peaks == k] ** 2 * correlation_peaks[label_peaks == k, k] ** 2) / np.sum(
                gfp[peaks] ** 2)
            gev_raw[k] = np.sum(gfp[label_raw == k] ** 2 * correlation_raw[label_raw == k, k] ** 2) / np.sum(gfp ** 2)
        return np.sum(gev_peaks), np.sum(gev_raw)

    @staticmethod
    def opt_microstate_criteria(cvs):
        k_opt = np.argmin(cvs)
        return k_opt

    @staticmethod
    def max_evec(v, axis):
        data = np.dot(v.T, v)
        evals, evecs = np.linalg.eig(data)
        c = evecs[:, np.argmax(np.abs(evals))]
        c = np.real(c)
        return Microstate.normalization(c, axis)

    @staticmethod
    def spatial_correlation(averaged_v1, averaged_v2, std_v1, std_v2, n_ch):
        correlation = np.dot(averaged_v1, averaged_v2) / (n_ch * np.outer(std_v1, std_v2))
        return correlation

    @staticmethod
    def fit_back(data, maps, distance=10, n_std=3, polarity=False):
        gfp = data.std(axis=1)
        peaks, _ = signal.find_peaks(gfp, distance=distance,
                                     height=(gfp.mean() - n_std * gfp.std(), gfp.mean() + n_std * gfp.std()))
        label = np.full(data.shape[0], -1)
        correlation = Microstate.spatial_correlation(data[peaks], zero_mean(maps, 1).T, gfp[peaks], maps.std(axis=1),
                                                     data.shape[0])
        correlation = correlation if polarity else abs(correlation)
        label_peaks = np.argmax(correlation, axis=1)
        for i in range(len(peaks)):
            if i == 0:
                previous_middle = 0
                next_middle = int((peaks[i] + peaks[i + 1]) / 2)
            elif i == len(peaks) - 1:
                previous_middle = int((peaks[i] + peaks[i - 1]) / 2)
                next_middle = data.shape[0]
            else:
                previous_middle = int((peaks[i] + peaks[i - 1]) / 2)
                next_middle = int((peaks[i] + peaks[i + 1]) / 2)
            label[previous_middle:next_middle] = label_peaks[i]
        return label

    @staticmethod
    def microstates_duration(label, n_maps):
        duration = [[] for _ in range(n_maps)]
        j = label[0]
        count = 1
        for i in range(1, len(label)):
            if j != label[i]:
                duration[j].append(count)
                j = label[i]
                count = 1
            elif j == label[i]:
                count += 1
        for i, temp in enumerate(duration):
            if i == label[0]:
                duration[i] = np.asarray(temp[1::]).mean() if len(temp[1::]) > 0 else 0
            elif i == label[-1]:
                duration[i] = np.asarray(temp[0:-1]).mean() if len(temp[0:-1]) > 0 else 0
            else:
                duration[i] = np.asarray(temp).mean() if len(temp) > 0 else 0

        return duration

    @staticmethod
    def microstates_coverage(label, n_maps):
        coverage = []
        n_label = len(label)
        for i in range(n_maps):
            coverage.append(np.argwhere(label == i).shape[0] / n_label)
        return coverage

    @staticmethod
    def microstates_parameters(data, maps, distance=10, n_std=3, polarity=False, sfreq=500, epoch=2):
        n_maps = len(maps)
        res = {'duration': [], 'coverage': []}
        data = zero_mean(data.T, 1)
        for i in range(0, data.shape[0], sfreq * epoch):
            data_epoch = data[i: i + sfreq * epoch]
            label = Microstate.fit_back(data_epoch, maps, distance, n_std, polarity)
            res['duration'].append(Microstate.microstates_duration(label, n_maps))
            res['coverage'].append(Microstate.microstates_coverage(label, n_maps))
        return res

    @staticmethod
    def microstates_parameters_selective(
            data,
            maps,
            distance: int = 10,
            n_std: int = 3,
            polarity: bool = False,
            sfreq: float = 500,
            epoch: float = 2.0,
            parameters: Optional[Set[str]] = None,
            log_base: float = math.e,
            states: Optional[List[Any]] = None,
            include_duration_seconds: bool = False,
    ) -> Dict[str, List[Any]]:
        """
        Compute selected microstate metrics per epoch from raw data and a fixed set of maps.

        This function is a *new* API that does not change the legacy `microstates_parameters`.
        It returns only the requested metrics as lists aligned by epoch.

        Args:
            data: EEG array (n_times, n_channels) or (n_channels, n_times) depending on your zero_mean usage.
            maps: Fixed microstate maps used for backfitting (len = n_maps).
            distance, n_std, polarity: Parameters passed to `fit_back`.
            sfreq: Sampling frequency in Hz. If <= 0, duration_seconds/transition_frequency in seconds are disabled.
            epoch: Window size in seconds (split the time-series into chunks of size `epoch`).
            parameters: A set of metric keys to compute. If None, a sensible default set is used.
                Supported keys:
                  - 'coverage'                -> coverage per state (list[list[float]])
                  - 'duration'                -> mean duration per state in *samples* (compat style) (list[list[float]])
                  - 'segments'                -> number of segments per state (list[list[int]])
                  - 'duration_seconds'        -> mean duration per state in *seconds* (list[dict or list])
                  - 'transition_frequency'    -> switches/sec if sfreq>0 else fraction per sample (list[float])
                  - 'entropy_rate'            -> (list[float])
                  - 'hurst_mean'              -> mean H across states (list[float])
                  - 'hurst_states'            -> per-state H (list[dict])
                  - 'transition_matrix'       -> per-epoch row-stochastic matrix (list[np.ndarray])
            log_base: Log base for entropy rate (default: e).
            states: Optional explicit state order. If None, defaults to [0..n_maps-1] for stability.
            include_duration_seconds: If True and 'duration_seconds' is requested, return second-based duration stats.

        Returns:
            Dict[str, list] where each value is a list with one entry per epoch. Keys match requested `parameters`.
        """
        # Default selection
        default_params = {
            'coverage',
            'duration',
            'transition_frequency',
            'entropy_rate',
            'hurst_mean',
        }
        params: Set[str] = set(default_params if parameters is None else parameters)

        # Prepare outputs
        out: Dict[str, List[Any]] = {k: [] for k in params}
        n_maps = len(maps)
        st = states if (states is not None and len(states) > 0) else list(range(n_maps))

        # Zero-mean like your legacy function (keeps legacy orientation)
        data = zero_mean(data.T, 1)  # shape -> (T, C)
        hop = int(sfreq * epoch) if sfreq and sfreq > 0 else int(epoch)  # graceful fallback

        # Iterate epochs
        for i in range(0, data.shape[0], hop):
            data_epoch = data[i: i + hop]
            if data_epoch.shape[0] < 2:
                # Not enough samples: push NaNs/empty per requested metric
                for k in params:
                    if k in ('coverage', 'duration', 'segments', 'duration_seconds'):
                        # per-state lists
                        if k == 'coverage':
                            out[k].append([np.nan] * n_maps)
                        elif k == 'duration':
                            out[k].append([np.nan] * n_maps)
                        elif k == 'segments':
                            out[k].append([0] * n_maps)
                        elif k == 'duration_seconds':
                            out[k].append([np.nan] * n_maps)
                    elif k in ('hurst_states', 'transition_matrix'):
                        out[k].append({} if k == 'hurst_states' else np.full((n_maps, n_maps), np.nan))
                    else:
                        out[k].append(np.nan)
                continue

            # 1) Back-fit labels for this epoch
            label = Microstate.fit_back(data_epoch, maps, distance, n_std, polarity)

            # 2) Legacy-style, per-state duration/coverage (in *samples*)
            if 'duration' in params or 'segments' in params or 'coverage' in params:
                # Duration (mean per state) in samples - preserves old semantics
                if 'duration' in params:
                    out['duration'].append(Microstate.microstates_duration(label, n_maps))

                # Coverage (fraction per state) in order 0..n_maps-1
                if 'coverage' in params:
                    out['coverage'].append(Microstate.microstates_coverage(label, n_maps))

                # Segments (run counts per state)
                if 'segments' in params:
                    # derive segments from run-length encoding
                    # faster path: reuse per-state durations length
                    durs = Microstate.microstates_duration(label, n_maps)  # uses contiguous runs internally
                    # microstates_duration returns means; we still need counts -> recompute RLE quickly:
                    # (moderately cheap, keeps clarity)
                    lab = np.asarray(label)
                    boundaries = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
                    run_labels = lab[boundaries]
                    seg_counts = [int(np.sum(run_labels == s)) for s in st]
                    out['segments'].append(seg_counts)

            # 3) Transition-based & information metrics (and Hurst)
            if any(p in params for p in
                   ('transition_frequency', 'entropy_rate', 'hurst_mean', 'hurst_states', 'duration_seconds',
                    'transition_matrix')):
                # Use helper that already aligns with explicit `states=st`
                met = metrics_for_labels(
                    label,
                    sfreq=sfreq,
                    states=st,
                    log_base=log_base,
                    include_duration=bool('duration_seconds' in params),  # compute second-based stats only if requested
                    include_state_hurst=bool('hurst_states' in params)
                )

                if 'transition_frequency' in params:
                    out['transition_frequency'].append(met['transition_frequency'])

                if 'entropy_rate' in params:
                    out['entropy_rate'].append(met['entropy_rate'])

                if 'hurst_mean' in params:
                    out['hurst_mean'].append(met.get('hurst_mean', np.nan))

                if 'hurst_states' in params:
                    out['hurst_states'].append({k: v for k, v in met.items() if k.startswith('hurst_')})

                if 'duration_seconds' in params and include_duration_seconds:
                    # We will produce three aligned lists per epoch:
                    #   duration_seconds_mean    : list[float] per state order 'st'
                    #   duration_seconds_median  : list[float]
                    #   duration_seconds_std     : list[float]

                    # 1) Mean from metrics_for_labels if available
                    dur_keys = [f"duration_mean_{s}" for s in st]
                    mean_row: List[float]
                    if all(k in met for k in dur_keys):
                        mean_row = [float(met[k]) for k in dur_keys]
                    else:
                        mean_row = [np.nan] * n_maps

                    # 2) Median / Std from run-lengths (convert samples -> seconds)
                    if sfreq and sfreq > 0:
                        per_state_runs = Microstate._state_run_lengths(np.asarray(label), st)
                        # convert each run length (samples) to seconds
                        sec_lists = {s: (np.asarray(per_state_runs.get(s, []), dtype=float) / float(sfreq))
                                     for s in st}

                        def safe_median(x: np.ndarray) -> float:
                            return float(np.median(x)) if x.size > 0 else float('nan')

                        def safe_std(x: np.ndarray) -> float:
                            # population std (ddof=0) for stability with few segments
                            return float(np.std(x, ddof=0)) if x.size > 0 else float('nan')

                        median_row = [safe_median(sec_lists[s]) for s in st]
                        std_row = [safe_std(sec_lists[s]) for s in st]
                    else:
                        # No valid sampling rate -> cannot report seconds
                        median_row = [np.nan] * n_maps
                        std_row = [np.nan] * n_maps

                    # Backward-compatible key rename: user requested 'duration_seconds',
                    # we expose explicit suffixed keys.
                    if 'duration_seconds' in out:
                        out['duration_seconds_mean'] = out.pop('duration_seconds')
                    if 'duration_seconds_mean' not in out:
                        out['duration_seconds_mean'] = []
                    if 'duration_seconds_median' not in out:
                        out['duration_seconds_median'] = []
                    if 'duration_seconds_std' not in out:
                        out['duration_seconds_std'] = []

                    out['duration_seconds_mean'].append(mean_row)
                    out['duration_seconds_median'].append(median_row)
                    out['duration_seconds_std'].append(std_row)


                if 'transition_matrix' in params:
                    P, _ = transition_matrix(label, states=st)
                    out['transition_matrix'].append(P)

        return out

    def cross_validation(self, var, n_maps):
        return var * (self.n_ch - 1) ** 2 / (self.n_ch - n_maps - 1.) ** 2

    def variance(self, label, maps, data):
        var = np.sum(data ** 2) - np.sum(np.sum(maps[label, :] * data, axis=1) ** 2)
        var /= (self.n_t * (self.n_ch - 1))
        return var

    def gfp_peaks(self, distance=10, n_std=3):
        gfp = self.data.std(axis=1)
        peaks, _ = signal.find_peaks(gfp, distance=distance,
                                     height=(gfp.mean() - n_std * gfp.std(), gfp.mean() + n_std * gfp.std()))
        self.peaks = peaks
        self.gfp = gfp

    def kmeans_modified(self, data, data_std, n_runs=100, n_maps=4, maxerr=1e-6, maxiter=1000, polarity=False):

        # GPU delegation when available
        if getattr(self, "_gpu", None) is not None:
            cv, gev, maps, label = self._gpu.kmeans_modified(
                data=data, data_std=self._gpu.xp.asarray(data_std) if hasattr(self._gpu, "xp") else data_std,
                n_runs=n_runs, n_maps=n_maps, maxerr=maxerr, maxiter=maxiter, polarity=polarity)
        else:
            n_gfp = data_std.shape[0]
            cv_list = []
            gev_list = []
            maps_list = []
            label_list = []
            for run in range(n_runs):
                # if run == 0 or run == n_runs - 1:
                #     print(f"{run + 1}/{n_runs}")
                rndi = np.random.permutation(n_gfp)[:n_maps]
                maps = Microstate.normalization(data[rndi, :], axis=1)
                n_iter = 0
                var0 = 1.0
                var1 = 0.0
                while ((np.abs(var0 - var1) / var0 > maxerr) & (n_iter < maxiter)):
                    n_iter += 1

                    label = np.argmax(np.dot(data, maps.T) ** 2, axis=1)
                    for k in range(n_maps):
                        data_k = data[label == k, :]
                        maps[k, :] = Microstate.max_evec(data_k, 0)
                    var1 = var0
                    var0 = Microstate.orthogonal_dist(data, maps[label, :])
                    var0 /= (n_gfp * (self.n_ch - 1))

                label, cv, gev = self.optimize_k(maps=maps, data=data, data_std=data_std, polarity=polarity)
                cv_list.append(cv)
                gev_list.append(gev)
                maps_list.append(maps)
                label_list.append(label)
            opt = Microstate.opt_microstate_criteria(cv_list)
            cv, gev, maps, label = cv_list[opt], gev_list[opt], maps_list[opt], label_list[opt]
        if self.reconstruct_pca_maps_info is None:
            return cv, gev, maps, label
        else:
            # serve for pca maps reconstruction and cv/gev recalculate
            # Reconstruct maps to original dimension (63/64 channels)
            # task_microstate.maps_list contains maps in reduced PCA space
            # Each element in maps_list is a list of maps for a specific K value
            original_data = zero_mean(self.reconstruct_pca_maps_info['original_data'])  # Original number of channels (typically 63/64)
            self.n_ch = original_data.shape[1]
            reduced_eigenvectors = self.reconstruct_pca_maps_info['reduced_eigenvectors']


            # Convert to numpy array: (n_maps, n_components)
            maps_array = np.array(maps)

            # Reconstruct to original dimension: (n_maps, n_channels)
            maps_reconstructed = np.dot(
                maps_array, reduced_eigenvectors
            )
            original_data_std = original_data.std(axis=1)
            self.maps_list_original_dim.append(maps_reconstructed)
            label_reconstructed_maps, reconstruct_cv, reconstruct_gev = self.optimize_k(maps=maps_reconstructed, data=original_data, data_std=original_data_std, polarity=polarity)
            self.label_list_original_dim.append(label_reconstructed_maps)
            return reconstruct_cv, reconstruct_gev, maps, label

    def opt_microstate(self, min_maps=2, max_maps=10, distance=10, n_std=3, n_runs=10, maxerr=1e-6, maxiter=1000,
                       polarity=False, peaks_only=True, method='kmeans_modified', opt_k=None):
        self.gfp_peaks(distance=distance, n_std=n_std)
        if peaks_only:
            temp_data = self.data[self.peaks]
            temp_data_std = self.gfp[self.peaks]
            temp_max_maps = min(temp_data.shape[0], max_maps)
        else:
            temp_data = self.data
            temp_data_std = self.gfp
            temp_max_maps = min(temp_data.shape[0], max_maps)

        if method == 'kmeans_modified':
            for n_maps in range(min_maps, temp_max_maps + 1):
                # if n_maps == min_maps or n_maps == max_maps or n_maps % 5 == 0:
                #     print(
                #         "kmeans_number:{number}/{maxi} will run {runs} times".format(number=n_maps, maxi=temp_max_maps,
                #                                                                      runs=n_runs))
                cv, gev, maps, label = self.kmeans_modified(data=temp_data, data_std=temp_data_std, n_runs=n_runs,
                                                            n_maps=n_maps, maxerr=maxerr, maxiter=maxiter,
                                                            polarity=polarity)
                self.cv_list.append(cv)
                self.gev_list.append(gev)
                self.maps_list.append(maps)
                self.label_list.append(label)

        elif method == 'aahc':
            self.cv_list, self.gev_list, self.maps_list, self.label_list = self.aahc(data=temp_data,
                                                                                     data_std=temp_data_std,
                                                                                     min_maps=min_maps,
                                                                                     max_maps=temp_max_maps,
                                                                                     polarity=polarity)

        if opt_k:
            self.opt_k_index = opt_k - min_maps
        else:
            self.opt_k_index = Microstate.opt_microstate_criteria(self.cv_list)

        self.cv = self.cv_list[self.opt_k_index]
        self.gev = self.gev_list[self.opt_k_index]
        self.maps = self.maps_list[self.opt_k_index]
        self.label = self.label_list[self.opt_k_index]

        self.opt_k = len(self.gev)

        self.gev_list = [temp.tolist() for temp in self.gev_list]
        self.maps_list = [temp.tolist() for temp in self.maps_list]
        self.label_list = [temp.tolist() for temp in self.label_list]

    def optimize_k(self, maps, data=None, data_std=None, polarity=False):

        # GPU delegation when available
        if getattr(self, "_gpu", None) is not None:
            label, correlation, cv, gev = self._gpu.optimize_k(maps=maps, data=data, data_std=data_std,
                                                               polarity=polarity)
            return label, correlation, cv, gev
        if data is None and data_std is None:
            data = self.data
            data_std = self.gfp
        n_maps = len(maps)
        # calc each time point original data and maps correlation
        correlation = Microstate.spatial_correlation(data, zero_mean(maps, 1).T, data_std, maps.std(axis=1), self.n_ch)
        correlation = correlation if polarity else abs(correlation)
        label = np.argmax(correlation, axis=1)
        var = self.variance(label=label, maps=maps, data=data)
        cv = self.cross_validation(var, n_maps)
        gev = self.global_explained_variance(n_maps=n_maps, correlation=correlation, label=label, data_std=data_std)
        return label, cv, gev

    def aahc(self, data, data_std, min_maps, max_maps, polarity=False):
        maps = data
        n_maps = len(maps)
        label_list = [[k] for k in range(len(maps))]
        cv_list = []
        gev_list = []
        maps_list = []
        res_label_list = []
        while n_maps > (min_maps - 1):
            # print("n_maps:%d" % n_maps)
            correlation = Microstate.spatial_correlation(data, zero_mean(maps, 1).T, data_std, maps.std(axis=1),
                                                         self.n_ch)
            correlation = correlation if polarity else abs(correlation)
            label = np.argmax(correlation, axis=1)
            gev = self.global_explained_variance(n_maps, correlation, label, data_std)
            if max_maps >= n_maps >= min_maps:
                var = self.variance(label, maps, data=data)
                cv = self.cross_validation(var, n_maps)
                cv_list.append(cv)
                maps_list.append(maps)
                gev_list.append(gev)
                res_label_list.append(label)
                if n_maps == min_maps:
                    break
            excluded_k = np.argmin(gev)
            maps = np.vstack((maps[:excluded_k, :], maps[excluded_k + 1:, :]))
            re_label = label_list.pop(excluded_k)
            re_cluster = []
            for k in re_label:
                correlation = Microstate.spatial_correlation(data[k, :], zero_mean(maps, 1).T, data_std[k],
                                                             maps.std(axis=1), self.n_ch)
                correlation = correlation if polarity else abs(correlation)
                new_label = np.argmax(correlation)
                re_cluster.append(new_label)
                label_list[new_label].append(k)

            for i in re_cluster:
                idx = label_list[i]
                maps[i] = Microstate.max_evec(data[idx, :], 0)
            n_maps = len(maps)

        return cv_list[::-1], gev_list[::-1], maps_list[::-1], res_label_list[::-1]

    @staticmethod
    def _state_run_lengths(label: np.ndarray, states: List[int]) -> Dict[int, List[int]]:
        """
        Run-length encode the label sequence and collect per-state run lengths (in samples).
        Returns a dict: state -> list of run lengths (each length is an int in samples).
        """
        lab = np.asarray(label)
        if lab.size == 0:
            return {s: [] for s in states}

        # RLE: boundaries where label changes (True at 0 to start the first run)
        boundaries = np.r_[True, lab[1:] != lab[:-1]]
        idx = np.flatnonzero(boundaries)
        # Run lengths are differences between consecutive boundaries, with last run to end
        lengths = np.diff(np.r_[idx, lab.size])

        run_states = lab[idx]
        per_state = {s: [] for s in states}
        for st, ln in zip(run_states, lengths):
            if st in per_state:
                per_state[st].append(int(ln))
        return per_state

