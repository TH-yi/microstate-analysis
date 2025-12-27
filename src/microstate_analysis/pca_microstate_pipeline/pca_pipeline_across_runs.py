"""
PCA Across Runs Pipeline
Aggregates runs/tasks into conditions per subject.
Input: Individual maps JSON files
Output: Across-runs maps per subject
"""

import os
import json
from collections import OrderedDict
from multiprocessing import Pool, cpu_count, get_context
from typing import List, Optional, Dict

from microstate_analysis.pca_microstate_pipeline.pca_pipeline_base import PCAPipelineBase
from microstate_analysis.logger.dualhandler import DualHandler
from microstate_analysis.microstate_base.data_handler import load_data
from microstate_analysis.microstate_base.microstate_batch_handler import batch_mean_microstate


class PCAPipelineAcrossRuns(PCAPipelineBase):
    """
    PCA Across Runs Pipeline.
    Aggregates each subject's runs/tasks into conditions.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        subjects: List[str],
        condition_dict: Dict[str, List[str]],
        percentage: float,
        n_k: int = 6,
        n_k_index: int = 1,
        n_ch: int = 63,
        data_suffix: str = "_pca_individual_maps.json",
        save_suffix: str = "_pca_across_runs.json",
        log_dir: Optional[str] = None,
        log_prefix: str = "pca_across_runs",
        log_suffix: str = "",
        use_gpu: bool = False,
    ):
        """
        Initialize PCA Across Runs Pipeline.

        Args:
            input_dir: Directory containing individual maps JSON files
            output_dir: Output directory for across-runs maps
            subjects: List of subject IDs
            condition_dict: Mapping from condition names to task lists
            percentage: PCA percentage
            n_k: Number of microstates
            n_k_index: Index into maps_list_original_dim (or maps_list as fallback)
            n_ch: Number of channels (default, will use original_n_channels from data if available)
            data_suffix: Input filename suffix
            save_suffix: Output filename suffix
            log_dir: Optional directory for log files
            log_prefix: Log file prefix
            log_suffix: Log file suffix
        """
        super().__init__()
        self.input_dir = os.path.join(input_dir, f"pca_{int(percentage*100)}")
        self.output_dir = output_dir
        self.subjects = subjects
        self.condition_dict = condition_dict
        self.condition_names = list(condition_dict.keys())
        self.percentage = percentage
        self.n_k = n_k
        self.n_k_index = n_k_index
        self.n_ch = n_ch
        self.data_suffix = data_suffix
        self.save_suffix = save_suffix
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
    def process_subject(
        subject: str,
        input_dir: str,
        output_dir: str,
        condition_dict: Dict[str, List[str]],
        condition_names: List[str],
        n_k: int,
        n_k_index: int,
        n_ch: int,
        data_suffix: str,
        save_suffix: str,
        use_gpu: bool = False,
    ) -> str:
        """
        Process a single subject's across-runs aggregation.

        Args:
            subject: Subject ID
            input_dir: Input directory
            output_dir: Output directory
            condition_dict: Condition to tasks mapping
            condition_names: List of condition names
            n_k: Number of microstates
            n_k_index: Index into maps_list_original_dim (or maps_list as fallback)
            n_ch: Number of channels (default, will use original_n_channels from data if available)
            data_suffix: Input filename suffix
            save_suffix: Output filename suffix

        Returns:
            Status message
        """
        res = OrderedDict()
        data_path = os.path.join(input_dir, f"{subject}{data_suffix}")

        if not os.path.exists(data_path):
            return f"Subject {subject}: Input file not found: {data_path}"

        data = load_data(data_path)
        condition_res = []

        # Aggregate maps for each condition
        # Use maps_list_original_dim (reconstructed to original 64D) instead of maps_list (reduced PCA space)
        for condition_name in condition_names:
            task_names = condition_dict[condition_name]
            maps = []
            # Get original_n_channels from data (default to n_ch if not available for backward compatibility)
            original_n_channels = n_ch
            for task_name in task_names:
                if task_name in data:
                    # Get original_n_channels from first available task
                    if original_n_channels == n_ch and 'original_n_channels' in data[task_name]:
                        original_n_channels = data[task_name]['original_n_channels']
                    # Get maps from maps_list_original_dim (reconstructed to original dimension)
                    maps_list_original = data[task_name].get('maps_list_original_dim', [])
                    if maps_list_original and len(maps_list_original) > n_k_index:
                        maps.append(maps_list_original[n_k_index])
                    else:
                        # Fallback 1: use opt_k_index if available
                        opt_k_index = data[task_name].get('opt_k_index', n_k_index)
                        if maps_list_original and len(maps_list_original) > opt_k_index:
                            maps.append(maps_list_original[opt_k_index])
                        else:
                            # Fallback 2: use maps_list (reduced PCA space) for backward compatibility
                            maps_list = data[task_name].get('maps_list', [])
                            if maps_list and len(maps_list) > n_k_index:
                                maps.append(maps_list[n_k_index])
                            elif maps_list and len(maps_list) > opt_k_index:
                                maps.append(maps_list[opt_k_index])

            if maps:
                # Use original_n_channels for mean microstate calculation
                condition_res.append(batch_mean_microstate([maps, n_k, original_n_channels, len(maps), use_gpu]))

        # Build result dictionary
        for idx, condition_name in enumerate(condition_names):
            if idx < len(condition_res):
                temp = condition_res[idx]
                if not isinstance(temp["maps"], list):
                    temp["maps"] = temp["maps"].tolist()
                res[condition_name] = {
                    'maps': temp["maps"],
                    'label': temp["label"],
                    'mean_similarity': temp['mean_similarity'],
                    'std_similarity': temp['std_similarity']
                }

        # Save result
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{subject}{save_suffix}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)

        return f"Subject {subject} processing complete."

    def run(self, max_processes: Optional[int] = None):
        """
        Run across-runs processing for all subjects.

        Args:
            max_processes: Maximum number of worker processes
        """
        if max_processes is None:
            max_processes = min(len(self.subjects), cpu_count())

        self.logger.log_info(
            f"Starting PCA across-runs processing: {len(self.subjects)} subjects, "
            f"{max_processes} processes, percentage: {self.percentage}"
        )

        # Prepare arguments for each subject
        args_list = [
            (
                subject,
                self.input_dir,
                self.output_dir,
                self.condition_dict,
                self.condition_names,
                self.n_k,
                self.n_k_index,
                self.n_ch,
                self.data_suffix,
                self.save_suffix,
                self.use_gpu,
            )
            for subject in self.subjects
        ]

        # Process subjects in parallel using spawn context for cross-platform compatibility
        ctx = get_context("spawn")
        with ctx.Pool(processes=max_processes) as pool:
            results = pool.starmap(self.process_subject, args_list)

        for result in results:
            self.logger.log_info(result)

        self.logger.log_info("PCA across-runs processing completed")

