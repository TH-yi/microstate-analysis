import os
import json
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Literal, Tuple, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

from sklearn.metrics import silhouette_score
from microstate_analysis.logger.dualhandler import DualHandler
from microstate_analysis.microstate_quality.quality_base import QualityBase


class ClusterQualityAnalysis(QualityBase):
    """
    Compute cluster quality metrics (Silhouette Coefficient and WCSS)
    for microstate-labeled CSVs across subjects or across conditions.

    Parallel execution:
        - If max_processors is None or <= 0: use os.cpu_count()
        - Otherwise, use min(max_processors, os.cpu_count())
    Logging:
        - Parent process aggregates and logs summaries.
        - Workers avoid verbose logs to reduce contention on file handles.
    """

    def __init__(self, log_dir: Optional[str], log_prefix: str):
        super().__init__()
        self.logger = DualHandler(log_dir=log_dir, prefix=log_prefix)

    # ---------- Public unified entry ----------

    def run(
        self,
        mode: Literal["across_subjects", "across_conditions"],
        csv_dir: str,
        output_dir: str,
        result_name: str,
        conditions: Optional[List[str]] = None,
        condition_dict: Optional[Dict[str, List[str]]] = None,
        label_column: str = "microstate_label",
        max_processors: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Run the cluster quality analysis (parallel).

        Args:
            mode: 'across_subjects' | 'across_conditions' (flow is the same).
            csv_dir: Root directory containing per-subject subfolders with CSVs.
            output_dir: Directory to save a JSON summary.
            result_name: Output JSON name without suffix.
            conditions: If provided, e.g., ["idea_generation","idea_evolution","idea_rating","rest"].
                        Each condition is matched by substring against CSV filenames.
            condition_dict: If provided, mapping {condition: [task_name_substrings]} for finer control.
            label_column: Column name that contains microstate labels (default: 'microstate_label').
            max_processors: Max number of processes; <=0 or None means use all logical CPUs.
        """
        self.logger.log_info(
            f"[Run] mode={mode}, csv_dir={csv_dir}, output_dir={output_dir}, result={result_name}.json"
        )

        # Build mapping: condition -> substrings
        if condition_dict and len(condition_dict) > 0:
            condition_to_keys = condition_dict
        elif conditions and len(conditions) > 0:
            condition_to_keys = {c: [c] for c in conditions}
        else:
            raise ValueError("You must provide either `conditions` or `condition_dict`.")

        # Collect tasks (file_path + which conditions it contributes to)
        tasks = self._gather_file_tasks(csv_dir, condition_to_keys)
        if not tasks:
            self.logger.log_warning(f"[Run] No CSV files matched under: {csv_dir}")
            return {}

        # Decide workers
        cpu_total = os.cpu_count() or 1
        if max_processors is None or max_processors <= 0:
            workers = cpu_total
        else:
            workers = min(max_processors, cpu_total)

        self.logger.log_info(f"[Run] Parallel execution with {workers} process(es). "
                             f"(Detected CPUs: {cpu_total})")

        # Parallel compute per-file metrics
        cond_sil_list: Dict[str, List[float]] = {c: [] for c in condition_to_keys.keys()}
        cond_wcss_list: Dict[str, List[float]] = {c: [] for c in condition_to_keys.keys()}

        # Prepare immutable args for workers
        worker_args = [
            (file_path, matched_conditions, label_column)
            for (file_path, matched_conditions) in tasks
        ]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_one_file, args) for args in worker_args]
            for fut in as_completed(futures):
                try:
                    results_for_file = fut.result()  # List[Tuple[condition, sil_opt, wcss_opt]]
                    for condition, sil, wcss in results_for_file:
                        if sil is not None:
                            cond_sil_list[condition].append(float(sil))
                        if wcss is not None:
                            cond_wcss_list[condition].append(float(wcss))
                except Exception as e:
                    self.logger.log_error(f"[Run] Worker failed: {e}")

        # Aggregate mean/std for each condition
        results: Dict[str, Dict[str, float]] = {}
        for cond in condition_to_keys.keys():
            sil_arr = np.asarray(cond_sil_list[cond], dtype=float)
            wcss_arr = np.asarray(cond_wcss_list[cond], dtype=float)

            sil_avg = float(np.mean(sil_arr)) if sil_arr.size else float("nan")
            sil_std = float(np.std(sil_arr)) if sil_arr.size else float("nan")
            wcss_avg = float(np.mean(wcss_arr)) if wcss_arr.size else float("nan")
            wcss_std = float(np.std(wcss_arr)) if wcss_arr.size else float("nan")
            nfiles = int(max(len(cond_sil_list[cond]), len(cond_wcss_list[cond])))

            results[cond] = {
                "silhouette_avg": sil_avg,
                "silhouette_std": sil_std,
                "wcss_avg": wcss_avg,
                "wcss_std": wcss_std,
                "num_files": nfiles,
            }

            self.logger.log_info(
                f"[{cond}] silhouette(mean±std)={sil_avg:.6f}±{sil_std:.6f}, "
                f"wcss(mean±std)={wcss_avg:.6f}±{wcss_std:.6f}, files={nfiles}"
            )

        # Save JSON
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{result_name}.json")
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)
            self.logger.log_info(f"[Run] Results saved to {out_path}")
        except Exception as e:
            self.logger.log_error(f"[Run] Error writing {out_path}: {e}")

        return results

    # ---------- Helpers (parent process) ----------

    @staticmethod
    def _file_belongs_to(file_name: str, key_list: List[str]) -> bool:
        """Return True if any key substring appears in file_name."""
        lower = file_name.lower()
        return any(k.lower() in lower for k in key_list)

    def _gather_file_tasks(
        self, csv_dir: str, condition_to_keys: Dict[str, List[str]]
    ) -> List[Tuple[str, List[str]]]:
        """
        Walk csv_dir/{subject}/*.csv and build a list of tasks:
            [(file_path, [matched_conditions]), ...]
        """
        tasks: List[Tuple[str, List[str]]] = []
        if not os.path.isdir(csv_dir):
            self.logger.log_error(f"CSV dir not found: {csv_dir}")
            return tasks

        for participant in os.listdir(csv_dir):
            participant_dir = os.path.join(csv_dir, participant)
            if not os.path.isdir(participant_dir):
                continue

            for file in os.listdir(participant_dir):
                if not file.lower().endswith(".csv"):
                    continue
                file_path = os.path.join(participant_dir, file)
                matched_conditions = [
                    cond
                    for cond, keys in condition_to_keys.items()
                    if self._file_belongs_to(file, keys)
                ]
                if matched_conditions:
                    tasks.append((file_path, matched_conditions))
        return tasks


# ---------- Worker (top-level function for pickling on Windows) ----------

def _process_one_file(args: Tuple[str, List[str], str]) -> List[Tuple[str, Optional[float], Optional[float]]]:
    """
    Worker to compute Silhouette and WCSS for one CSV file.
    Returns a list for all matched conditions of that file:
        [(condition, silhouette or None, wcss or None), ...]
    Minimal logging in workers to avoid I/O contention.
    """
    file_path, matched_conditions, label_column = args
    try:
        # Robust CSV loading with index detection (self-contained)
        df = _load_csv_with_index_check(file_path)

        # Require label column
        if label_column not in df.columns:
            return [(cond, None, None) for cond in matched_conditions]

        labels = np.asarray(df[label_column].tolist())
        features_df = df.drop(columns=[label_column])
        numeric_df = features_df.select_dtypes(include=["number"])
        if numeric_df.shape[1] == 0 or numeric_df.shape[0] < 2:
            return [(cond, None, None) for cond in matched_conditions]

        data = numeric_df.to_numpy(dtype=float)

        # Silhouette (guards)
        unique_labels = np.unique(labels)
        if unique_labels.size < 2 or data.shape[0] <= unique_labels.size:
            silhouette_avg = None
        else:
            silhouette_avg = float(silhouette_score(data, labels))

        # WCSS
        wcss = 0.0
        for lab in unique_labels:
            cluster_points = data[labels == lab]
            if cluster_points.size == 0:
                continue
            centroid = np.mean(cluster_points, axis=0)
            diffs = cluster_points - centroid
            wcss += float(np.sum(np.sum(diffs * diffs, axis=1)))

        return [(cond, silhouette_avg, wcss) for cond in matched_conditions]

    except Exception:
        # On failure, mark all matched conditions for this file as None
        return [(cond, None, None) for cond in matched_conditions]


# ---------- Lightweight, self-contained CSV loader (for workers) ----------

def _load_csv_with_index_check(csv_file: str) -> "pd.DataFrame":
    """
    Load CSV and handle potential index row/column.
    - Peek first few rows to infer index row/column.
    - Heuristics compatible with typical microstate outputs:
        * If first column looks like monotonically increasing small integers, treat as index col.
        * If header row contains many numeric-like tokens, treat as data (no header).
    """
    # Peek
    sample = pd.read_csv(csv_file, nrows=6)
    # Heuristic: first column index-like?
    def _is_index_col(values: Iterable) -> bool:
        ints = []
        for v in values:
            try:
                iv = int(v)
                ints.append(iv)
            except Exception:
                return False
        if not ints:
            return False
        # Monotonic non-negative small ints is a good proxy
        return all(iv >= 0 for iv in ints)

    # Heuristic: header row numeric-like?
    def _is_header_numeric_like(cols: List[str]) -> bool:
        cnt = 0
        for c in cols:
            try:
                float(str(c))
                cnt += 1
            except Exception:
                pass
        # if many columns parse as numbers -> header is not header
        return cnt >= max(2, len(cols) // 2)

    has_index_column = _is_index_col(sample.iloc[1:5, 0].tolist()) if sample.shape[0] >= 5 else False
    # Use original column names to judge header nature (skip the first col for fairness)
    col_names = list(sample.columns)
    has_index_row = _is_header_numeric_like(col_names[1:]) if len(col_names) > 1 else False

    if has_index_row and has_index_column:
        return pd.read_csv(csv_file, index_col=0, header=0)
    if has_index_column:
        return pd.read_csv(csv_file, index_col=0)
    if has_index_row:
        return pd.read_csv(csv_file, header=0)
    # no index row/col detected
    return pd.read_csv(csv_file, header=None)
