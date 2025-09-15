"""
Author: Tianhao Ning
Description:
  Individual EEG microstate computation with a single global process pool (pickle-only),
  with centralized logging: workers send log events to the main process via a
  multiprocessing.Queue; the main process drains the queue and writes logs through
  a single DualHandler. This avoids cross-process file contention and preserves
  strict log ordering.

Core behavior (UPGRADED):
  • Decouple "submission leader" from "completion leader":
      - Submission leader: who gets PRIORITY for submitting tasks.
      - Completion leader: who triggers dump/evict when all tasks finish.
  • As soon as a subject has all tasks SUBMITTED (even if some are still running),
    the submission leader advances to the next subject with pending submissions,
    and prefetching is driven by the submission leader (pipeline-style).
  • Prefetch next K subjects' input arrays to overlap I/O with computation.
  • As soon as a subject has all tasks submitted, its input arrays can be evicted
    from the main-process memory (pickle payload was already sent to workers).
  • When a subject finishes (all tasks complete), persist results and evict results.

Stability:
  - Pickle-only transport (no shared memory). Safe on Windows "spawn".
  - Worker normalizes data to float64, C-contiguous, aligned, writeable,
    and ensures (channels, times) layout before calling algorithms.

Centralized logging:
  - Workers DO NOT write files; they push (level, message) events into a Queue.
  - Main process runs a background thread to drain the Queue and call DualHandler.
  - Messages include PID and subject|task for quick attribution.
  - error_callback is used so worker exceptions are surfaced immediately.

Notes:
  - This file assumes DualHandler has methods: log_info/log_warning/log_error.
  - PipelineBase.dump_to_json(name WITHOUT extension) will append ".json" internally.
"""

import os
import gc
import json
import time
import threading
from dataclasses import dataclass
from collections import OrderedDict, deque, defaultdict
from multiprocessing import get_context
from typing import Dict, Deque, Tuple, Optional, Any

import numpy as np

from microstate_analysis.microstate_pipeline.pipeline_base import PipelineBase
from microstate_analysis.logger.dualhandler import DualHandler
from microstate_analysis.microstate_base.microstate_batch_handler import batch_microstate
from microstate_analysis.microstate_base.data_handler import list_to_matrix

# GPU gating primitives (shared via Manager)
_GPU_SEM: Any = None  # parallel slots, e.g., 2
_GPU_WAIT_Q: Any = None  # waiting queue with capacity = total_slots - parallel_slots
_GPU_WAITERS: Any = None  # mgr.Value('i', count of waiters)
_GPU_THRESHOLD: int = 80_000
_GPU_WAIT_TIMEOUT_S: Optional[float] = None  # None => wait forever

# Module-level queue holder set by pool initializer (spawn-safe)
_LOG_QUEUE: Any = None


# ------------------------- Log event helpers -------------------------

@dataclass(frozen=True)
class LogEvent:
    """A simple cross-process log event."""
    level: str  # 'info' | 'warning' | 'error'
    message: str  # already formatted line


def _pool_initializer(
        log_queue: Any,
        gpu_sem: Any = None,
        gpu_wait_q: Any = None,
        gpu_threshold: int = 80_000,
        gpu_wait_timeout_s: Optional[float] = None,
        gpu_waiters: Any = None
) -> None:
    global _LOG_QUEUE, _GPU_SEM, _GPU_WAIT_Q, _GPU_THRESHOLD, _GPU_WAIT_TIMEOUT_S, _GPU_WAITERS
    _LOG_QUEUE = log_queue
    _GPU_SEM = gpu_sem
    _GPU_WAIT_Q = gpu_wait_q
    _GPU_THRESHOLD = int(gpu_threshold)
    _GPU_WAIT_TIMEOUT_S = gpu_wait_timeout_s
    _GPU_WAITERS = gpu_waiters


def _emit_log(level: str, message: str) -> None:
    """
    Send a log event to the main process queue using the module-level queue.
    Never raise inside logging to avoid masking real errors.
    """
    try:
        q = _LOG_QUEUE
        if q is not None:
            q.put(LogEvent(level=level, message=message))
    except Exception:
        # Last-resort fallback: swallow logging failures
        pass


def _drain_log_queue(q: Any, logger: DualHandler, stop_sentinel: object) -> None:
    """
    Background thread target in the main process:
    read LogEvent from queue and write via DualHandler until sentinel is received.
    """
    while True:
        try:
            evt = q.get()
        except (EOFError, OSError):
            break  # queue is gone
        if evt is stop_sentinel:
            break
        try:
            if isinstance(evt, LogEvent):
                lv = (evt.level or "info").lower()
                if lv == "error":
                    logger.log_error(evt.message)
                elif lv == "warning":
                    logger.log_warning(evt.message)
                else:
                    logger.log_info(evt.message)
            else:
                # Unexpected payload; log as info
                logger.log_info(str(evt))
        except Exception as e:
            # Avoid killing the drain thread on log formatting errors
            logger.log_error(f"[main] Log drain error: {e}")


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


def _worker_task(spec: TaskSpec, data_pickle: Optional[np.ndarray]) -> dict:
    """
    Worker entry (pickle-only): use the pickled ndarray, normalize it, then run batch_microstate.
    The worker does NOT write files; it pushes log events to the main process via _emit_log.
    """
    pid = os.getpid()

    try:
        if data_pickle is None:
            raise ValueError("Worker received no data (pickle payload is None).")

        _emit_log(
            "info",
            (f"[pid={pid}] Start task: {spec.subject} | {spec.task} | "
             f"peaks_only={spec.peaks_only} min_maps={spec.min_maps} max_maps={spec.max_maps} "
             f"method={spec.method} n_std={spec.n_std} n_runs={spec.n_runs} use_gpu={spec.use_gpu}")
        )

        # 1) Normalize memory flags & dtype
        task_data = _ensure_ready_for_algo(data_pickle)
        # 2) Normalize to (channels, times) so Microstate(...).T -> (times, channels)
        task_data = _as_channels_times(task_data)

        # ---------- Decide GPU vs CPU with logging ----------
        use_gpu_for_this_task = False
        acquired_sem = False
        took_wait_slot = False
        inc_waiters = False
        reason_parts = []

        # Basic metrics for the decision
        n_channels, n_points = int(task_data.shape[0]), int(task_data.shape[1])
        is_big = (n_points >= int(_GPU_THRESHOLD)) if _GPU_THRESHOLD is not None else False

        # Whether GPU gating is available at all
        gpu_gating_ready = (spec.use_gpu and (_GPU_SEM is not None))
        if not spec.use_gpu:
            reason_parts.append("GPU disabled by spec.use_gpu=False")
        elif _GPU_SEM is None:
            reason_parts.append("GPU gating primitives not initialized")

        waiters_present = False
        if gpu_gating_ready:
            try:
                waiters_present = (_GPU_WAITERS is not None and int(_GPU_WAITERS.value) > 0)
            except Exception:
                waiters_present = False

        if gpu_gating_ready:
            if not waiters_present:
                # Try immediate GPU (no waiters → allow fast-path)
                try:
                    acquired_sem = _GPU_SEM.acquire(block=False)
                except Exception:
                    acquired_sem = False

                if acquired_sem:
                    use_gpu_for_this_task = True
                    reason_parts.append("GPU: immediate token acquired (no waiters)")
                else:
                    # GPU busy: only BIG tasks may queue (capacity enforced by maxsize)
                    if is_big and (_GPU_WAIT_Q is not None):
                        try:
                            _GPU_WAIT_Q.put_nowait(1)
                            took_wait_slot = True
                            if _GPU_WAITERS is not None:
                                _GPU_WAITERS.value += 1
                                inc_waiters = True
                            reason_parts.append(
                                f"GPU busy → queued big task (points={n_points} ≥ threshold={_GPU_THRESHOLD})")
                        except Exception as e:
                            _emit_log("warning", f"Error check GPU wait queue: {e}")
                            took_wait_slot = False
                            reason_parts.append("GPU busy and waiting queue full → CPU fallback")

                        if took_wait_slot:
                            t0 = time.monotonic()
                            try:
                                if _GPU_WAIT_TIMEOUT_S is None:
                                    acquired_sem = _GPU_SEM.acquire()  # block forever
                                else:
                                    acquired_sem = _GPU_SEM.acquire(timeout=float(_GPU_WAIT_TIMEOUT_S))
                            except Exception:
                                acquired_sem = False

                            waited = time.monotonic() - t0
                            if acquired_sem:
                                if took_wait_slot:
                                    try:
                                        _GPU_WAIT_Q.get_nowait()  # free waiting slot immediately
                                    except Exception:
                                        pass
                                    if inc_waiters and (_GPU_WAITERS is not None):
                                        try:
                                            _GPU_WAITERS.value -= 1
                                        except Exception:
                                            pass
                                    took_wait_slot = False
                                    inc_waiters = False
                                    reason_parts.append("dequeued from waiting queue (slot released upon token)")

                                use_gpu_for_this_task = True
                                reason_parts.append(f"GPU: token acquired after waiting {waited:.3f}s")
                            else:
                                # Only possible if you configured a finite timeout
                                reason_parts.append(f"Timeout after {waited:.3f}s → CPU fallback")
                                try:
                                    _GPU_WAIT_Q.get_nowait()
                                except Exception:
                                    pass
                                took_wait_slot = False
                                if inc_waiters and (_GPU_WAITERS is not None):
                                    _GPU_WAITERS.value -= 1
                                    inc_waiters = False
                    else:
                        if not is_big:
                            reason_parts.append(
                                f"GPU busy and task small (points={n_points} < threshold={_GPU_THRESHOLD}) → CPU fallback")
                        else:
                            reason_parts.append("GPU busy; no waiting queue available → CPU fallback")
            else:
                # There is a waiter → forbid newcomers from immediate grab (fairness)
                reason_parts.append("Waiter present → forbid immediate GPU grab for newcomers")
                if is_big and (_GPU_WAIT_Q is not None):
                    try:
                        _GPU_WAIT_Q.put_nowait(1)
                        took_wait_slot = True
                        if _GPU_WAITERS is not None:
                            _GPU_WAITERS.value += 1
                            inc_waiters = True
                        reason_parts.append(
                            f"Joined waiting queue as big task (points={n_points} ≥ threshold={_GPU_THRESHOLD})")
                    except Exception:
                        took_wait_slot = False
                        reason_parts.append("Waiting queue full while waiter present → CPU fallback")

                    if took_wait_slot:
                        t0 = time.monotonic()
                        try:
                            if _GPU_WAIT_TIMEOUT_S is None:
                                acquired_sem = _GPU_SEM.acquire()
                            else:
                                acquired_sem = _GPU_SEM.acquire(timeout=float(_GPU_WAIT_TIMEOUT_S))
                        except Exception:
                            acquired_sem = False

                        waited = time.monotonic() - t0
                        if acquired_sem:
                            if took_wait_slot:
                                try:
                                    _GPU_WAIT_Q.get_nowait()  # free waiting slot immediately
                                except Exception:
                                    pass
                                if inc_waiters and (_GPU_WAITERS is not None):
                                    try:
                                        _GPU_WAITERS.value -= 1
                                    except Exception:
                                        pass
                                took_wait_slot = False
                                inc_waiters = False
                                reason_parts.append("dequeued from waiting queue (slot released upon token)")
                            use_gpu_for_this_task = True
                            reason_parts.append(f"GPU: token acquired after waiting {waited:.3f}s")
                        else:
                            reason_parts.append(f"Timeout after {waited:.3f}s → CPU fallback")
                            try:
                                _GPU_WAIT_Q.get_nowait()
                            except Exception:
                                pass
                            took_wait_slot = False
                            if inc_waiters and (_GPU_WAITERS is not None):
                                _GPU_WAITERS.value -= 1
                                inc_waiters = False
                else:
                    reason_parts.append(
                        f"Small task while waiter present (points={n_points} < threshold={_GPU_THRESHOLD}) → CPU fallback")

        device = "GPU" if use_gpu_for_this_task else "CPU"
        _emit_log("info",
                  (f"[pid={pid}] Dispatch to {device} | subject={spec.subject} | task={spec.task} | "
                   f"points={n_points} | threshold={_GPU_THRESHOLD} | waiters={'yes' if waiters_present else 'no'} | "
                   f"reasons: " + " ; ".join(reason_parts)))

        # Prepare parameters for microstate batching
        batch_params = [
            task_data,
            spec.peaks_only,
            spec.min_maps,
            spec.max_maps,
            None,  # placeholder for prior maps if any
            spec.method,
            spec.n_std,
            spec.n_runs,
            bool(use_gpu_for_this_task),
        ]

        try:
            task_microstate = batch_microstate(batch_params)
        finally:
            # Always release gating primitives if we used/queued GPU, and do hygiene logging
            if acquired_sem:
                try:
                    _GPU_SEM.release()
                    _emit_log("info", f"[pid={pid}] GPU token released | subject={spec.subject} | task={spec.task}")
                except Exception:
                    pass
            if took_wait_slot:
                try:
                    _GPU_WAIT_Q.get_nowait()
                    _emit_log("info",
                              f"[pid={pid}] GPU waiting-slot cleared | subject={spec.subject} | task={spec.task}")
                except Exception:
                    pass
            if inc_waiters and (_GPU_WAITERS is not None):
                try:
                    _GPU_WAITERS.value -= 1
                except Exception:
                    pass
            if use_gpu_for_this_task:
                # Optional: extra GPU memory hygiene per task
                try:
                    import cupy as cp
                    cp.cuda.Stream.null.synchronize()
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                    _emit_log("info", f"[pid={pid}] GPU memory pools freed | subject={spec.subject} | task={spec.task}")
                except Exception:
                    pass

            # Safe int conversion for opt_k_index (can be None in rare cases)
        oki = task_microstate.opt_k_index
        oki_sanitized = -1 if oki is None else int(oki)

        _emit_log(
            "info",
            (f"[pid={pid}] Finished task: {spec.subject} | {spec.task} | "
             f"opt_k={task_microstate.opt_k}, opt_k_index={oki_sanitized}")
        )

        return {
            'subject': spec.subject,
            'task': spec.task,
            'cv_list': task_microstate.cv_list,
            'gev_list': task_microstate.gev_list,
            'maps_list': task_microstate.maps_list,
            'opt_k': task_microstate.opt_k,
            'opt_k_index': oki_sanitized,
            'min_maps': spec.min_maps,
            'max_maps': spec.max_maps,
        }
    except Exception as e:
        _emit_log("error", f"[pid={pid}] Worker failed ({spec.subject} | {spec.task}): {e}")
        # Re-raise to trigger error_callback in the main process
        raise


# ------------------------- Main pipeline (pickle-only) -------------------------

class PipelineIndividualRun(PipelineBase):
    def __init__(self, input_dir, output_dir, subjects, peaks_only, min_maps, max_maps, task_name,
                 log_dir=None, prefix=None, suffix=None, cluster_method='kmeans_modified',
                 n_std=3, n_runs=100, use_gpu=False):
        """
        Pickle-only pipeline (Windows-spawn safe) with centralized logging.
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

        # Main-process logger (only one writes to files)
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

        out: Dict[str, np.ndarray] = {}
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
        Run with a single global pool (pickle-only) and centralized logging.
        UPGRADED submission strategy:
          - Drive prefetch and priority by "submission leader".
          - As soon as a subject has ALL tasks SUBMITTED, advance submission leader
            and optionally evict its input arrays from memory (data_cache).
          - Completion (dump/evict results) remains per-subject when all tasks finish.
        """
        self.logger.log_info(f"[main] Using {max_processes} processes")

        if not self.subjects:
            self.logger.log_warning("[main] No subjects provided. Nothing to do.")
            return

        # Per-subject bookkeeping
        pending_submit: Dict[str, Deque[str]] = {s: deque(self.task_name) for s in self.subjects}
        pending_count: Dict[str, int] = {s: len(self.task_name) for s in self.subjects}
        results_per_subject: Dict[str, OrderedDict] = {s: OrderedDict() for s in self.subjects}
        optk_per_subject: Dict[str, Dict[str, int]] = {s: {} for s in self.subjects}

        # Cache at most: submission leader + up to prefetch_depth subjects
        data_cache: Dict[str, Dict[str, np.ndarray]] = {}
        preloaded_flags: Dict[str, bool] = defaultdict(bool)

        # Active async results: key=(subject, task) -> AsyncResult
        inflight: Dict[Tuple[str, str], Any] = {}

        # Completion leader (used only for dump/evict results)
        current_index = 0
        current_subject = self.subjects[current_index]

        # Submission leader (drives prefetch & priority of submissions)
        current_submit_index = 0
        current_submit_subject = self.subjects[current_submit_index]

        ctx = get_context("spawn")
        log_queue = ctx.Queue()  # cross-process log queue
        STOP = object()  # sentinel for draining thread

        # Start the centralized log drain thread
        log_thread = threading.Thread(
            target=_drain_log_queue, args=(log_queue, self.logger, STOP), daemon=True
        )
        log_thread.start()
        # ---- GPU gating config (tunable) ----
        parallel_slots = 6  # GPU parallel running
        total_slots = 7  # GPU parallel running + waiting
        waiting_capacity = max(total_slots - parallel_slots, 0)  # = 1
        gpu_threshold = 50000

        gpu_wait_timeout_s = None

        gpu_sem = None
        gpu_wait_q = None
        gpu_waiters = None
        mgr = None

        if self.use_gpu:
            mgr = ctx.Manager()
            gpu_sem = mgr.Semaphore(parallel_slots)
            gpu_wait_q = mgr.Queue(maxsize=waiting_capacity)
            gpu_waiters = mgr.Value('i', 0)
            self.logger.log_info(
                f"[main] GPU gating: parallel={parallel_slots}, "
                f"wait_capacity={waiting_capacity}, threshold={gpu_threshold}, "
                f"timeout={'∞' if gpu_wait_timeout_s is None else gpu_wait_timeout_s}s"
            )
        else:
            gpu_threshold = 0

        def on_error(exc: BaseException, subject: str, task: str):
            # Immediate visibility on worker exceptions
            self.logger.log_error(f"[main] Worker error on {subject} | {task}: {exc}")

        with ctx.Pool(
                processes=max_processes,
                initializer=_pool_initializer,
                initargs=(log_queue, gpu_sem, gpu_wait_q, gpu_threshold, gpu_wait_timeout_s, gpu_waiters)
        ) as pool:

            def submit_task(subject: str, task: str):
                """
                Submit one (subject, task) to the pool using pickle (robust).
                Also write a 'Submit task -> ...' line in the MAIN process log to ensure
                a visible starting point in case a worker crashes before 'Start'.
                """
                self.logger.log_info(f"[main] Submit task -> {subject} | {task}")
                arr = data_cache[subject][task]  # float64, contiguous
                spec = TaskSpec(
                    subject=subject, task=task,
                    peaks_only=self.peaks_only, min_maps=self.min_maps, max_maps=self.max_maps,
                    method=self.cluster_method, n_std=self.n_std, n_runs=self.n_runs, use_gpu=self.use_gpu
                )
                ar = pool.apply_async(
                    _worker_task,
                    (spec, arr),
                    error_callback=lambda e, s=subject, t=task: on_error(e, s, t)
                )
                inflight[(subject, task)] = ar

            # ------------------ Phase 1: preload + fill pool from the FIRST submission leader ------------------
            # Preload submission leader
            data_cache[current_submit_subject] = self._load_subject_data(current_submit_subject)
            preloaded_flags[current_submit_subject] = True
            self.logger.log_info(f"[main] Preloaded subject (submit leader): {current_submit_subject}")

            # Fill the pool with tasks from the submission leader only
            while len(inflight) < max_processes and pending_submit[current_submit_subject]:
                task = pending_submit[current_submit_subject].popleft()
                submit_task(current_submit_subject, task)

            # ------------------ Phase 2: main loop (prefetch + completion + submission) ------------------
            while inflight or any(pending_submit[s] for s in self.subjects):
                did_submit = False  # track whether we submitted anything this iteration

                # (A) Prefetch driven by SUBMISSION LEADER:
                #     If the submission leader has <=2 not-yet-submitted tasks, start preloading next subjects.
                remain_to_submit = len(pending_submit[current_submit_subject])
                if remain_to_submit <= 2:
                    for k in range(1, prefetch_depth + 1):
                        nxt_idx = current_submit_index + k
                        if nxt_idx < len(self.subjects):
                            nxt = self.subjects[nxt_idx]
                            if not preloaded_flags.get(nxt, False):
                                try:
                                    data_cache[nxt] = self._load_subject_data(nxt)
                                    preloaded_flags[nxt] = True
                                    self.logger.log_info(f"[main] Preloaded subject: {nxt}")
                                except Exception as e:
                                    self.logger.log_warning(f"[main] Prefetch failed for {nxt}: {e}")

                # (B0) Advance SUBMISSION LEADER if it has NO pending submissions left.
                if not pending_submit[current_submit_subject]:
                    # The leader has submitted all its tasks → we can evict its INPUT arrays now.
                    if current_submit_subject in data_cache:
                        try:
                            del data_cache[current_submit_subject]
                        except Exception:
                            pass
                        gc.collect()

                    # Move submission leader to the next subject that still has pending submissions.
                    next_submit = None
                    for ii in range(current_submit_index + 1, len(self.subjects)):
                        nxt_s = self.subjects[ii]
                        if pending_submit[nxt_s]:  # still has tasks to submit
                            next_submit = ii
                            break
                    if next_submit is not None:
                        current_submit_index = next_submit
                        current_submit_subject = self.subjects[current_submit_index]
                        # Ensure the new submission leader is preloaded immediately.
                        if not preloaded_flags.get(current_submit_subject, False):
                            try:
                                data_cache[current_submit_subject] = self._load_subject_data(current_submit_subject)
                                preloaded_flags[current_submit_subject] = True
                                self.logger.log_info(
                                    f"[main] Preloaded subject (submit leader): {current_submit_subject}")
                            except Exception as e:
                                self.logger.log_warning(
                                    f"[main] On-lead prefetch failed for {current_submit_subject}: {e}")

                # (B1) SUBMIT: prioritize submission leader; if none, use idle slots for preloaded next subjects.
                while len(inflight) < max_processes:
                    if pending_submit[current_submit_subject]:
                        task = pending_submit[current_submit_subject].popleft()
                        submit_task(current_submit_subject, task)
                        did_submit = True
                    else:
                        advanced = False
                        for nxt_idx in range(current_submit_index + 1, len(self.subjects)):
                            nxt = self.subjects[nxt_idx]
                            if pending_submit[nxt] and preloaded_flags.get(nxt, False):
                                task = pending_submit[nxt].popleft()
                                submit_task(nxt, task)
                                did_submit = True
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
                        # Persist & evict results for this subject
                        # NOTE: pass name WITHOUT ".json"; PipelineBase will append suffix.
                        self.dump_to_json(results_per_subject[subj], self.output_dir, f'{subj}_individual_maps')
                        self.logger.log_info(f"[main] Finished subject: {subj} (evicting results from memory)")
                        results_per_subject[subj] = None  # free references

                        # Evict any stray input cache (in case not removed earlier)
                        if subj in data_cache:
                            del data_cache[subj]
                        gc.collect()

                        # Advance COMPLETION LEADER if this subject was the current one
                        if subj == current_subject:
                            next_cur = None
                            for ii in range(current_index + 1, len(self.subjects)):
                                if pending_count[self.subjects[ii]] > 0:
                                    next_cur = ii
                                    break
                            if next_cur is not None:
                                current_index = next_cur
                                current_subject = self.subjects[current_index]

                # (D) Cooperative sleep to avoid busy spinning
                if not completed and not did_submit:
                    time.sleep(0.01)

            # End while: all tasks done; pool closed/joined by context manager

        # Stop the centralized log drain thread
        try:
            log_queue.put(STOP)
        except Exception:
            pass
        log_thread.join(timeout=5)

        if self.use_gpu:
            try:
                mgr.shutdown()
            except Exception as e:
                self.logger.log_error(f"Failed shutdown GPU queue manager: {e}")

        # (E) Save task-wise map counts if requested
        if save_task_map_counts:
            if not task_map_counts_output_dir or not task_map_counts_output_filename:
                self.logger.log_error("[main] Missing parameters for saving task map counts!")
                raise ValueError("task_map_counts_output_dir and task_map_counts_output_filename are required.")
            task_wise_map_counts = [
                {"subject": s, "opt_k": optk_per_subject[s]} for s in self.subjects
            ]
            # Pass name WITHOUT ".json"; PipelineBase will append suffix.
            self.dump_to_json(task_wise_map_counts, task_map_counts_output_dir, task_map_counts_output_filename)

    # ---------- Pickling safety ----------

    def __getstate__(self):
        state = self.__dict__.copy()
        # DualHandler is not picklable; store its config and rebuild on restore
        state['logger'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = DualHandler(**self._logger_cfg)


# ------------------------- CLI-like example -------------------------

if __name__ == '__main__':
    # Example wiring; adjust paths and flags for your environment
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
    individual_log_suffix = ''  # centralized logger only, worker logs go through queue
    task_map_counts_output_dir = '../../../storage/microstate_output/individual_run'
    task_map_counts_output_filename = 'individual_map_counts'
    max_processes = 30
    cluster_method = 'kmeans_modified'
    n_std = 33
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
