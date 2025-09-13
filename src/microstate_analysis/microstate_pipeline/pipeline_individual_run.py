"""
Author: Tianhao Ning
Description:
  Individual EEG microstate computation with a single global process pool (pickle-only).

Core behavior:
  1) First, submit the first subject's tasks to fill the pool (up to max_workers).
  2) While running, if the current subject has <=1 not-yet-submitted tasks,
     preload the next subject's data to overlap I/O (but do NOT submit its tasks yet).
  3) Always prioritize submitting tasks from the current subject. Only when the current
     subject has no not-yet-submitted tasks, use idle slots for the next preloaded subject.
  4) As soon as a subject finishes (all its tasks complete), save results and evict
     its arrays from memory.

Stability:
  - Pickle-only transport (no shared memory). This is robust on Windows spawn.
  - Worker normalizes data to float64, C-contiguous, aligned, writeable,
    and ensures (channels, times) layout before calling algorithms.
  - Logger rebuilt in worker for clear log attribution.
"""

import os
import gc
import json
import time
from dataclasses import dataclass
from collections import OrderedDict, deque, defaultdict
from multiprocessing import get_context
from typing import Dict, Deque, Tuple, Optional

import numpy as np

from microstate_analysis.microstate_pipeline.pipeline_base import PipelineBase
from microstate_analysis.logger.dualhandler import DualHandler
from microstate_analysis.microstate_base.microstate_batch_handler import batch_microstate
from microstate_analysis.microstate_base.data_handler import list_to_matrix


# ------------------------- Worker helpers (pickle-only) -------------------------

@dataclass(frozen=True)
class TaskSpec:
    """Descriptor for a single (subject, task) execution."""
    subject: str
    task: str
    # Algorithm hyper-parameters
    peaks_only: bool = True
    min_maps: int = 2
    max_maps: int = 10
    method: str = 'kmeans_modified'
    n_std: int = 3
    n_runs: int = 100
    use_gpu: bool = False


def _rebuild_logger(cfg: dict) -> DualHandler:
    """Rebuild logger in worker processes."""
    return DualHandler(**cfg)


def _ensure_ready_for_algo(arr: np.ndarray) -> np.ndarray:
    """
    Ensure array is float64, C-contiguous, aligned, and writeable.
    If any condition is not met, return a safe copied array.
    """
    x = np.asarray(arr)
    need_copy = (
        x.dtype != np.float64
        or not x.flags['C_CONTIGUOUS']
        or not x.flags['ALIGNED']
        or not x.flags['WRITEABLE']
    )
    return np.array(x, dtype=np.float64, order='C', copy=True) if need_copy else x


def _as_channels_times(x: np.ndarray) -> np.ndarray:
    """
    Normalize layout to (channels, times).
    Heuristic: EEG typically has channels << timepoints; if first dim looks like time, transpose.
    """
    if x.ndim != 2:
        raise ValueError(f"EEG array must be 2D, got shape={x.shape}")
    h, w = x.shape
    return x.T if h > w else x


def _worker_task(spec: TaskSpec, logger_cfg: dict, data_pickle: Optional[np.ndarray] = None) -> dict:
    """
    Worker entry (pickle-only): use the pickled ndarray, normalize it, then run batch_microstate.
    """
    logger = _rebuild_logger(logger_cfg)
    try:
        if data_pickle is None:
            raise ValueError("Worker received no data (pickle payload is None).")
        logger.log_info(
            f'Start task: {spec.subject}, {spec.task}, '
        )
        # 1) Normalize memory flags & dtype
        task_data = _ensure_ready_for_algo(data_pickle)
        # 2) Normalize to (channels, times) so Microstate(...).T -> (times, channels)
        task_data = _as_channels_times(task_data)

        # Optional debug (enable if needed)
        # logger.log_info(
        #     f"Data ready: shape={task_data.shape}, dtype={task_data.dtype}, "
        #     f"C={task_data.flags['C_CONTIGUOUS']}, A={task_data.flags['ALIGNED']}, "
        #     f"W={task_data.flags['WRITEABLE']}, OWN={task_data.flags['OWNDATA']}"
        # )

        batch_params = [
            task_data,
            spec.peaks_only,
            spec.min_maps,
            spec.max_maps,
            None,               # placeholder for prior maps if any
            spec.method,
            spec.n_std,
            spec.n_runs,
            spec.use_gpu,
        ]
        task_microstate = batch_microstate(batch_params)

        logger.log_info(
            f'Finished task: {spec.subject}, {spec.task}, '
            f'opt_k: {task_microstate.opt_k}, opt_k_index: {int(task_microstate.opt_k_index)}'
        )

        return {
            'subject': spec.subject,
            'task': spec.task,
            'cv_list': task_microstate.cv_list,
            'gev_list': task_microstate.gev_list,
            'maps_list': task_microstate.maps_list,
            'opt_k': task_microstate.opt_k,
            'opt_k_index': int(task_microstate.opt_k_index),
            'min_maps': spec.min_maps,
            'max_maps': spec.max_maps,
        }
    except Exception as e:
        logger.log_error(f'Worker failed ({spec.subject}, {spec.task}): {e}')
        raise


# ------------------------- Main pipeline (pickle-only) -------------------------

class PipelineIndividualRun(PipelineBase):
    def __init__(self, input_dir, output_dir, subjects, peaks_only, min_maps, max_maps, task_name,
                 log_dir=None, prefix=None, suffix=None, cluster_method='kmeans_modified',
                 n_std=3, n_runs=100, use_gpu=False):
        """
        Pickle-only pipeline.
        """
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.subjects = list(subjects)
        self.task_name = list(task_name)

        # Microstate parameters
        self.peaks_only = peaks_only
        self.min_maps = min_maps
        self.max_maps = max_maps
        self.cluster_method = cluster_method
        self.n_std = n_std
        self.n_runs = n_runs
        self.use_gpu = use_gpu

        # Logger (and its config to rebuild inside workers)
        self._logger_cfg = dict(log_dir=log_dir, prefix=prefix or '', suffix=suffix or '')
        self.logger = DualHandler(**self._logger_cfg)

    # ---------- I/O helpers ----------

    def _load_subject_data(self, subject: str) -> Dict[str, np.ndarray]:
        """
        Load one subject's JSON and convert each task's list to a C-contiguous float64 ndarray.
        (We keep raw orientation here; worker will normalize to (channels, times) before compute.)
        """
        path = os.path.abspath(os.path.join(self.input_dir, f"{subject}.json"))
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        out = {}
        for t in self.task_name:
            arr = list_to_matrix(raw[t])
            if not isinstance(arr, np.ndarray):
                arr = np.asarray(arr)
            # Enforce a stable memory layout & dtype early (pickle will carry this efficiently)
            out[t] = np.ascontiguousarray(arr, dtype=np.float64)
        return out

    # ---------- Scheduling ----------

    def generate_individual_eeg_maps(self, save_task_map_counts: bool,
                                     task_map_counts_output_dir: Optional[str] = None,
                                     task_map_counts_output_filename: Optional[str] = None,
                                     max_processes: Optional[int] = None,
                                     prefetch_depth: int = 1) -> None:
        """
        Run with a single global pool (pickle-only).

        Step 1: Submit the first subject's tasks to fill the pool (up to max_processes).
        Step 2: While running, when current subject has <=1 not-yet-submitted tasks,
                preload the next subject's data to overlap I/O (do not submit it yet).
        Priority: always submit from the current subject first.

        Eviction: when a subject finishes, dump JSON and evict its arrays from memory.
        """
        cpu = os.cpu_count() or 1
        n_proc = min(max(1, cpu), max_processes or cpu)
        self.logger.log_info(f"[individual] Using {n_proc} processes")

        if not self.subjects:
            self.logger.log_warning("No subjects provided. Nothing to do.")
            return

        # Per-subject bookkeeping
        pending_submit: Dict[str, Deque[str]] = {s: deque(self.task_name) for s in self.subjects}
        pending_count: Dict[str, int] = {s: len(self.task_name) for s in self.subjects}
        results_per_subject: Dict[str, OrderedDict] = {s: OrderedDict() for s in self.subjects}
        optk_per_subject: Dict[str, Dict[str, int]] = {s: {} for s in self.subjects}

        # Cache at most two subjects in memory: current + preloaded next
        data_cache: Dict[str, Dict[str, np.ndarray]] = {}
        preloaded_flags: Dict[str, bool] = defaultdict(bool)

        # Active async results: key=(subject, task) -> AsyncResult
        inflight: Dict[Tuple[str, str], any] = {}

        ctx = get_context("spawn")
        with ctx.Pool(processes=n_proc) as pool:

            def submit_task(subject: str, task: str):
                """Submit one (subject, task) to the pool using pickle (robust)."""
                arr = data_cache[subject][task]  # float64, contiguous
                spec = TaskSpec(
                    subject=subject, task=task,
                    peaks_only=self.peaks_only, min_maps=self.min_maps, max_maps=self.max_maps,
                    method=self.cluster_method, n_std=self.n_std, n_runs=self.n_runs, use_gpu=self.use_gpu
                )
                ar = pool.apply_async(_worker_task, (spec, self._logger_cfg, arr))
                inflight[(subject, task)] = ar

            # ------------------ Phase 1: submit first batch from the first subject ------------------
            current_index = 0
            current_subject = self.subjects[current_index]

            # Load the very first subject
            data_cache[current_subject] = self._load_subject_data(current_subject)
            self.logger.log_info(f"Preloaded subject: {current_subject}")

            # Fill the pool with tasks from the first subject only
            while len(inflight) < n_proc and pending_submit[current_subject]:
                task = pending_submit[current_subject].popleft()
                submit_task(current_subject, task)

            # ------------------ Phase 2: main loop (prefetch + completion + scheduling) --------------
            while inflight or any(pending_submit[s] for s in self.subjects):

                # (A) Prefetch: if current subject has <=1 not-yet-submitted tasks, preload next subject(s)
                remain_to_submit = len(pending_submit[current_subject])
                if remain_to_submit <= 1:
                    for k in range(1, prefetch_depth + 1):
                        nxt_idx = current_index + k
                        if nxt_idx < len(self.subjects):
                            nxt = self.subjects[nxt_idx]
                            if not preloaded_flags[nxt]:
                                try:
                                    data_cache[nxt] = self._load_subject_data(nxt)
                                    preloaded_flags[nxt] = True
                                    self.logger.log_info(f"Preloaded subject: {nxt}")
                                except Exception as e:
                                    self.logger.log_warning(f"Prefetch failed for {nxt}: {e}")

                # (B) Submit: prioritize current subject; if it has no not-yet-submitted tasks,
                #     and there are idle slots, try the next preloaded subject.
                while len(inflight) < n_proc:
                    if pending_submit[current_subject]:
                        task = pending_submit[current_subject].popleft()
                        submit_task(current_subject, task)
                    else:
                        # Current subject has no more tasks to submit; use idle slots for next preloaded subjects
                        advanced = False
                        for nxt_idx in range(current_index + 1, len(self.subjects)):
                            nxt = self.subjects[nxt_idx]
                            if pending_submit[nxt] and preloaded_flags.get(nxt, False):
                                task = pending_submit[nxt].popleft()
                                submit_task(nxt, task)
                                advanced = True
                                break
                        if not advanced:
                            break  # nothing to submit for now

                # (C) Collect ready tasks (non-blocking)
                completed = []
                for key, ar in list(inflight.items()):
                    if ar.ready():
                        completed.append(key)

                for (subj, task) in completed:
                    ar = inflight.pop((subj, task))
                    out = ar.get()  # propagate worker exceptions

                    # Aggregate results
                    results_per_subject[subj][task] = {
                        'cv_list': out['cv_list'],
                        'gev_list': out['gev_list'],
                        'maps_list': out['maps_list'],
                        'opt_k': out['opt_k'],
                        'opt_k_index': out['opt_k_index'],
                        'min_maps': out['min_maps'],
                        'max_maps': out['max_maps'],
                    }
                    optk_per_subject[subj][task] = out['opt_k']

                    # Update pending task count and finalize subject if done
                    pending_count[subj] -= 1
                    if pending_count[subj] == 0:
                        # Persist & evict
                        self.dump_to_json(results_per_subject[subj], self.output_dir, f'{subj}_individual_maps.json')
                        self.logger.log_info(f'Finished subject: {subj} (evicting from memory)')
                        results_per_subject[subj] = None  # free references

                        # Evict data cache
                        if subj in data_cache:
                            del data_cache[subj]
                        gc.collect()

                        # If the finished subject is the current subject, advance the "current"
                        if subj == current_subject:
                            next_cur = None
                            for ii in range(current_index + 1, len(self.subjects)):
                                if pending_count[self.subjects[ii]] > 0:
                                    next_cur = ii
                                    break
                            if next_cur is not None:
                                current_index = next_cur
                                current_subject = self.subjects[current_index]

                # (D) Small cooperative sleep to avoid busy spinning when pool is full and nothing completed
                if not completed and len(inflight) >= n_proc:
                    time.sleep(0.01)

            # End while: all tasks done; pool closed/joined by context manager

        # (E) Save task-wise map counts if requested
        if save_task_map_counts:
            if not task_map_counts_output_dir or not task_map_counts_output_filename:
                self.logger.log_error("Missing parameters for saving task map counts!")
                raise ValueError("task_map_counts_output_dir and task_map_counts_output_filename are required.")
            task_wise_map_counts = [
                {"subject": s, "opt_k": optk_per_subject[s]} for s in self.subjects
            ]
            self.dump_to_json(task_wise_map_counts, task_map_counts_output_dir, task_map_counts_output_filename)

    # ---------- Pickling safety ----------

    def __getstate__(self):
        state = self.__dict__.copy()
        state['logger'] = None  # DualHandler is not picklable; rebuild in worker if needed
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = DualHandler(**self._logger_cfg)


# ------------------------- CLI-like example -------------------------

if __name__ == '__main__':
    subjects = ['sub_01', 'sub_02', 'sub_03', 'sub_04', 'sub_05', 'sub_06', 'sub_07', 'sub_08', 'sub_09', 'sub_10',
                'sub_11', 'sub_12', 'sub_13', 'sub_14', 'sub_15', 'sub_16', 'sub_17', 'sub_18', 'sub_19', 'sub_20',
                'sub_21', 'sub_22', 'sub_23', 'sub_24', 'sub_25', 'sub_26', 'sub_27', 'sub_28']
    task_name = [
        '1_idea generation', '2_idea generation', '3_idea generation',
        '1_idea evolution', '2_idea evolution', '3_idea evolution',
        '1_idea rating', '2_idea rating', '3_idea rating',
        '1_rest', '3_rest'
    ]

    individual_input_dir = '../../../storage/clean_data'
    individual_output_dir = '../../../storage/microstate_output/individual_run'
    peaks_only = True
    min_maps = 2
    max_maps = 10
    save_task_map_counts = True

    # Optional parameters
    individual_log_dir = '../../../storage/log/individual_run'
    individual_log_prefix = 'individual_run'
    individual_log_suffix = ''
    task_map_counts_output_dir = '../../../storage/microstate_output/individual_run'
    task_map_counts_output_filename = 'individual_map_counts'
    max_processes = 4
    cluster_method = 'kmeans_modified'
    n_std = 3
    n_runs = 100
    use_gpu = True

    job = PipelineIndividualRun(
        log_dir=individual_log_dir, prefix=individual_log_prefix, suffix=individual_log_suffix,
        input_dir=individual_input_dir, output_dir=individual_output_dir, subjects=subjects,
        peaks_only=peaks_only, min_maps=min_maps, max_maps=max_maps, task_name=task_name,
        cluster_method=cluster_method, n_std=n_std, n_runs=n_runs, use_gpu=use_gpu
    )

    job.generate_individual_eeg_maps(
        save_task_map_counts=save_task_map_counts,
        task_map_counts_output_dir=task_map_counts_output_dir,
        task_map_counts_output_filename=task_map_counts_output_filename,
        max_processes=max_processes,
        prefetch_depth=1
    )
