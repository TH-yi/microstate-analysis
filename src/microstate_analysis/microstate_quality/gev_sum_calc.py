"""
Module: ori_gev_sum.py  (parallelized)
Purpose:
    Single-class GEV summarization for microstate analysis with CPU-parallel execution.
    - mode="across_subjects": use condition-specific maps, e.g. map_data[condition]["maps"]
    - mode="across_conditions": use a global maps array, e.g. map_data["maps"]

Parallel policy:
    - If max_processors is None or <= 0: use all logical CPU cores (os.cpu_count()).
    - Otherwise, use min(max_processors, os.cpu_count()).

This file replaces the previous sequential implementation.
"""

from __future__ import annotations
import os
import json
import math
from typing import Dict, List, Literal, Tuple, Optional

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from microstate_analysis.logger.dualhandler import DualHandler
from microstate_analysis.microstate_base.microstate import Microstate


class GEVSumCalc:
    """
    Unified GEV sum calculator for both 'across_subjects' and 'across_conditions',
    now with multi-process parallelism.
    """

    def __init__(
        self,
        log_dir: Optional[str],
        log_prefix: str,
        csv_dir: str,
        maps_path: str,
        subjects: List[str],
        task_names: List[str],
        condition_dict: Dict[str, List[str]],
        mode: str,
    ) -> None:
        """
        Args:
            log_dir: Directory for logs (if None, logs go to stderr only).
            log_prefix: Logger file prefix.
            csv_dir: Root directory for CSV files.
            maps_path: JSON path to microstate maps.
            subjects: List of subject IDs (e.g., ["P01", "P02", ...]).
            task_names: Task names (must align with your CSV filenames).
            condition_dict: Mapping condition -> list of task names (used for aggregation).
            mode: 'across_subjects' or 'across_conditions'.
        """
        self.logger = DualHandler(log_dir, prefix=log_prefix)
        self.csv_dir = csv_dir
        self.maps_path = maps_path
        self.subjects = subjects
        self.task_names = task_names
        self.condition_dict = condition_dict
        self.mode = mode

        # Accumulator per condition: list of (gev_peaks_sum, gev_raw_sum, data_length)
        self._condition_values: Dict[str, List[Tuple[float, float, int]]] = {
            condition: [] for condition in self.condition_dict.keys()
        }
        self.logger.log_info(
            f"[Init] mode={self.mode}, conditions={list(self._condition_values.keys())}"
        )

    # ---------- Helpers ----------

    @staticmethod
    def _find_condition_for_task(
        task_name: str, condition_dict: Dict[str, List[str]]
    ) -> Optional[str]:
        """Return the condition name that contains task_name; None if not found."""
        for cond, tasks in condition_dict.items():
            if task_name in tasks:
                return cond
        return None

    # ---------- Pipeline (Parallel) ----------

    def run(
        self,
        output_dir: str,
        output_name_subjects: str = "gev_sum_ori_across_subjects.json",
        output_name_conditions: str = "gev_sum_ori_across_conditions.json",
        max_processors: Optional[int] = None,
    ) -> None:
        """
        Main entry to perform the GEV summarization in parallel.
        Steps:
          1) Load maps JSON in parent
          2) Build work items for all subject × task pairs:
             (csv_file_path, maps_list_for_this_task, condition_name_for_aggregation)
          3) Parallel compute GEV per file in worker processes
          4) Aggregate weighted stats per condition and dump JSON
        """
        # 1) Load maps JSON (parent)
        try:
            with open(self.maps_path, "r", encoding="utf-8") as f:
                map_data = json.load(f)
            self.logger.log_info(f"[Run] Loaded maps from {self.maps_path}")
        except FileNotFoundError:
            self.logger.log_error(f"[Run] Map file not found: {self.maps_path}")
            return
        except json.JSONDecodeError:
            self.logger.log_error(f"[Run] JSON decode error: {self.maps_path}")
            return

        # 2) Build tasks
        # Each task: (csv_file, maps_list (np.ndarray), condition_name)
        work_items: List[Tuple[str, np.ndarray, str]] = []
        missing_files = 0

        if self.mode == "across_subjects":
            # condition-specific maps
            for subj in self.subjects:
                for task_name in self.task_names:
                    cond = self._find_condition_for_task(task_name, self.condition_dict)
                    if not cond:
                        self.logger.log_error(
                            f"[Run] Cannot resolve condition for task '{task_name}'. Skipping."
                        )
                        continue
                    try:
                        maps_list = np.array(map_data[cond]["maps"])
                    except Exception as e:
                        self.logger.log_error(f"[Run] Bad maps for condition '{cond}': {e}")
                        continue

                    csv_file = os.path.join(self.csv_dir, subj, f"{subj}_{task_name}.csv")
                    if not os.path.isfile(csv_file):
                        missing_files += 1
                        self.logger.log_error(f"[Run] CSV not found: {csv_file}")
                        continue
                    work_items.append((csv_file, maps_list, cond))
        else:
            # across_conditions: global maps array
            try:
                global_maps = np.array(map_data["maps"])
            except Exception as e:
                self.logger.log_error(f"[Run] Bad global 'maps': {e}")
                return

            for subj in self.subjects:
                for task_name in self.task_names:
                    # Still need a condition name for aggregation (based on task_name)
                    cond = self._find_condition_for_task(task_name, self.condition_dict)
                    if not cond:
                        self.logger.log_error(
                            f"[Run] Cannot resolve condition for task '{task_name}'. Skipping."
                        )
                        continue

                    csv_file = os.path.join(self.csv_dir, subj, f"{subj}_{task_name}.csv")
                    if not os.path.isfile(csv_file):
                        missing_files += 1
                        self.logger.log_error(f"[Run] CSV not found: {csv_file}")
                        continue
                    work_items.append((csv_file, global_maps, cond))

        if missing_files:
            self.logger.log_warning(f"[Run] Missing CSV files: {missing_files}")

        if not work_items:
            self.logger.log_warning("[Run] No valid work items; nothing to compute.")
            return

        # Decide worker count
        cpu_total = os.cpu_count() or 1
        if max_processors is None or max_processors <= 0:
            workers = cpu_total
        else:
            workers = min(max_processors, cpu_total)

        self.logger.log_info(
            f"[Run] Parallel execution with {workers} process(es). (Detected CPUs: {cpu_total})"
        )

        # 3) Parallel compute
        # Worker returns: (condition, gev_peaks_sum, gev_raw_sum, data_length) or None
        results_collected = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_one_csv, item) for item in work_items]
            for fut in as_completed(futures):
                try:
                    ret = fut.result()
                    if ret is None:
                        continue
                    condition, peaks_sum, raw_sum, data_len = ret
                    self._condition_values[condition].append(
                        (float(peaks_sum), float(raw_sum), int(data_len))
                    )
                    results_collected += 1
                except Exception as e:
                    self.logger.log_error(f"[Run] Worker failed: {e}")

        self.logger.log_info(f"[Run] Files processed: {results_collected}/{len(work_items)}")

        # 4) Aggregate and write
        json_name = (
            output_name_subjects if self.mode == "across_subjects" else output_name_conditions
        )
        self._write_weighted_stats(output_dir, json_name)

    # ---------- Aggregation & Output ----------

    def _write_weighted_stats(self, output_dir: str, json_name: str) -> None:
        """
        Compute weighted average and weighted std for each condition:
            mean_w = sum(x_i * w_i) / sum(w_i)
            std_w  = sqrt(sum((x_i - mean_w)^2 * w_i) / sum(w_i))
        where weights are data lengths.
        """
        results: Dict[str, Dict[str, object]] = {}
        for condition, triples in self._condition_values.items():
            if not triples:
                self.logger.log_warning(f"[Stats] No data for condition '{condition}'. Skipping.")
                continue

            total_len = 0
            peaks_sum_w = 0.0
            raw_sum_w = 0.0

            for peaks, raw, L in triples:
                peaks_sum_w += peaks * L
                raw_sum_w += raw * L
                total_len += L

            if total_len == 0:
                self.logger.log_warning(f"[Stats] Zero length for '{condition}'. Skipping.")
                continue

            peaks_mean_w = peaks_sum_w / total_len
            raw_mean_w = raw_sum_w / total_len

            peaks_var_w = 0.0
            raw_var_w = 0.0
            for peaks, raw, L in triples:
                peaks_var_w += ((peaks - peaks_mean_w) ** 2) * L
                raw_var_w += ((raw - raw_mean_w) ** 2) * L

            peaks_std_w = math.sqrt(peaks_var_w / total_len)
            raw_std_w = math.sqrt(raw_var_w / total_len)

            results[condition] = {
                "weighted_avg_peaks": float(peaks_mean_w),
                "weighted_avg_raw": float(raw_mean_w),
                "peaks_weighted_std": float(peaks_std_w),
                "raw_weighted_std": float(raw_std_w),
                "condition_value": [(float(p), float(r), int(L)) for (p, r, L) in triples],
            }

            self.logger.log_info(
                f"[Stats] {condition}: mean_w -> Peaks={peaks_mean_w:.6f}, Raw={raw_mean_w:.6f}"
            )
            self.logger.log_info(
                f"[Stats] {condition}: std_w  -> Peaks={peaks_std_w:.6f}, Raw={raw_std_w:.6f}"
            )

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, json_name)
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            self.logger.log_info(f"[Stats] Wrote results to {out_path}")
        except Exception as e:
            self.logger.log_error(f"[Stats] Error writing {out_path}: {e}")


# ---------- Worker (top-level for Windows spawn safety) ----------

def _process_one_csv(item: Tuple[str, np.ndarray, str]) -> Optional[Tuple[str, float, float, int]]:
    """
    Worker to compute GEV for one CSV file.

    Args (packed in `item`):
        - csv_file: absolute path to CSV
        - maps_list: numpy array of microstate maps to use
        - condition_name: the condition bucket to aggregate into

    Returns:
        (condition_name, gev_peaks_sum, gev_raw_sum, data_length)
        or None on failure.
    """
    try:
        csv_file, maps_list, condition_name = item
        # Use fast reader; input CSVs are numeric matrices (rows=timepoints, cols=channels)
        data = np.loadtxt(csv_file, delimiter=",")
        # Compute GEV (peaks/raw sums) with provided maps
        gev_peaks_sum, gev_raw_sum = Microstate.global_explained_variance_sum(data, maps_list)
        return (condition_name, float(gev_peaks_sum), float(gev_raw_sum), int(len(data)))
    except Exception:
        return None
