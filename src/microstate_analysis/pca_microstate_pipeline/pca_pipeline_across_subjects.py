"""
PCA Across Subjects Pipeline
Aggregates subjects into condition-level maps.
Input: Across-runs JSON files per subject
Output: Single across-subjects JSON with all conditions
"""

import os
from collections import OrderedDict
from multiprocessing import Pool, cpu_count, get_context
from typing import List, Optional, Dict

from microstate_analysis.pca_microstate_pipeline.pca_pipeline_base import PCAPipelineBase
from microstate_analysis.logger.dualhandler import DualHandler
from microstate_analysis.microstate_base.data_handler import load_data
from microstate_analysis.microstate_base.microstate_batch_handler import batch_mean_microstate


class PCAPipelineAcrossSubjects(PCAPipelineBase):
    """
    PCA Across Subjects Pipeline.
    Aggregates per-subject across-runs maps into condition-level maps.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        subjects: List[str],
        condition_names: List[str],
        percentage: float,
        n_k: int = 6,
        n_ch: int = 63,
        data_suffix: str = "_pca_across_runs.json",
        save_name: str = "pca_across_subjects.json",
        log_dir: Optional[str] = None,
        log_prefix: str = "pca_across_subjects",
        log_suffix: str = "",
        use_gpu: bool = False,
    ):
        """
        Initialize PCA Across Subjects Pipeline.

        Args:
            input_dir: Directory containing across-runs JSON files
            output_dir: Output directory
            subjects: List of subject IDs
            condition_names: List of condition names
            percentage: PCA percentage
            n_k: Number of microstates
            n_ch: Number of channels
            data_suffix: Input filename suffix
            save_name: Output filename
            log_dir: Optional directory for log files
            log_prefix: Log file prefix
            log_suffix: Log file suffix
        """
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.subjects = subjects
        self.condition_names = condition_names
        self.percentage = percentage
        self.n_k = n_k
        self.n_ch = n_ch
        self.data_suffix = data_suffix
        self.save_name = save_name
        self.use_gpu = use_gpu

        # Logger config for safe reconstruction in child processes
        self._logger_cfg = dict(log_dir=log_dir, prefix=log_prefix or "", suffix=log_suffix or "")
        self.logger = DualHandler(**self._logger_cfg)

    # ---------- pickling-safe logger ----------

    def __getstate__(self):
        """Exclude logger from pickle state (contains thread locks)."""
        state = self.__dict__.copy()
        state["logger"] = None
        return state

    def __setstate__(self, state):
        """Reconstruct logger in child process."""
        self.__dict__.update(state)
        self.logger = DualHandler(**self._logger_cfg)

    @staticmethod
    def process_condition(
        condition_name: str,
        input_dir: str,
        subjects: List[str],
        data_suffix: str,
        n_k: int,
        n_ch: int,
        use_gpu: bool = False,
    ) -> Dict:
        """
        Process a single condition across all subjects.

        Args:
            condition_name: Condition name
            input_dir: Input directory
            subjects: List of subject IDs
            data_suffix: Input filename suffix
            n_k: Number of microstates
            n_ch: Number of channels

        Returns:
            Condition result dictionary
        """
        maps = []
        for subject in subjects:
            data_path = os.path.join(input_dir, f"{subject}{data_suffix}")
            if os.path.exists(data_path):
                data = load_data(data_path)
                if condition_name in data:
                    maps.append(data[condition_name]['maps'])

        if not maps:
            return {
                'condition': condition_name,
                'maps': [],
                'label': [],
                'mean_similarity': 0.0,
                'std_similarity': 0.0
            }

        # Aggregate maps across subjects
        result = batch_mean_microstate([maps, n_k, n_ch, len(maps), use_gpu])

        return {
            'condition': condition_name,
            'maps': result["maps"].tolist(),
            'label': result["label"],
            'mean_similarity': result['mean_similarity'],
            'std_similarity': result['std_similarity']
        }

    def run(self, max_processes: Optional[int] = None):
        """
        Run across-subjects processing.

        Args:
            max_processes: Maximum number of worker processes
        """
        if max_processes is None:
            max_processes = min(len(self.condition_names), cpu_count())

        self.logger.log_info(
            f"Starting PCA across-subjects processing: {len(self.subjects)} subjects, "
            f"{len(self.condition_names)} conditions, {max_processes} processes, "
            f"percentage: {self.percentage}"
        )

        res = OrderedDict()

        # Prepare arguments for each condition
        args_list = [
            (
                condition_name,
                self.input_dir,
                self.subjects,
                self.data_suffix,
                self.n_k,
                self.n_ch,
                self.use_gpu,
            )
            for condition_name in self.condition_names
        ]

        # Process conditions in parallel using spawn context for cross-platform compatibility
        ctx = get_context("spawn")
        with ctx.Pool(processes=max_processes) as pool:
            results = pool.starmap(self.process_condition, args_list)

        # Build result dictionary
        for result in results:
            condition_name = result['condition']
            res[condition_name] = {
                'maps': result['maps'],
                'label': result['label'],
                'mean_similarity': result['mean_similarity'],
                'std_similarity': result['std_similarity']
            }

        # Save result
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, self.save_name)
        self.dump_to_json_path(res, output_path)

        self.logger.log_info("PCA across-subjects processing completed")

