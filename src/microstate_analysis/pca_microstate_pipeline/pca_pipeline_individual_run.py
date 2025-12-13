"""
PCA Individual Run Pipeline
Processes PCA-transformed CSV files to generate individual microstate maps.
Input: PCA final_matrix CSV files
Output: Individual microstate maps per subject per task
"""

import os
from collections import OrderedDict
from multiprocessing import Pool, cpu_count, get_context
from typing import List, Optional
import numpy as np

from microstate_analysis.pca_microstate_pipeline.pca_pipeline_base import PCAPipelineBase
from microstate_analysis.logger.dualhandler import DualHandler
from microstate_analysis.microstate_base.microstate_batch_handler import batch_microstate


class PCAPipelineIndividualRun(PCAPipelineBase):
    """
    PCA Individual Run Pipeline.
    Processes PCA-transformed CSV files to generate individual microstate maps.
    """

    def __init__(
        self,
        pca_data_dir: str,
        output_dir: str,
        subjects: List[str],
        task_names: List[str],
        percentage: float,
        peaks_only: bool = False,
        min_maps: int = 2,
        max_maps: int = 10,
        opt_k: Optional[int] = None,
        cluster_method: str = "kmeans_modified",
        n_std: int = 3,
        n_runs: int = 100,
        log_dir: Optional[str] = None,
        log_prefix: str = "pca_individual_run",
        log_suffix: str = "",
        use_gpu: bool = False,
    ):
        """
        Initialize PCA Individual Run Pipeline.

        Args:
            pca_data_dir: Base directory containing PCA final_matrix data
                         (structure: {pca_data_dir}/pca_{percentage}/final_matrix/{subject}/*.csv)
            output_dir: Output directory for individual maps
            subjects: List of subject IDs
            task_names: List of task names
            percentage: PCA percentage (e.g., 0.95, 0.98, 0.99)
            peaks_only: Use peaks-only logic
            min_maps: Minimum number of maps
            max_maps: Maximum number of maps
            opt_k: Optional fixed K value
            cluster_method: Clustering method
            n_std: Threshold std
            n_runs: Clustering restarts
            log_dir: Optional directory for log files
            log_prefix: Log file prefix
            log_suffix: Log file suffix
            use_gpu: Enable GPU acceleration
        """
        super().__init__()
        self.pca_data_dir = pca_data_dir
        self.output_dir = output_dir
        self.subjects = subjects
        self.task_names = task_names
        self.percentage = percentage
        self.peaks_only = peaks_only
        self.min_maps = min_maps
        self.max_maps = max_maps
        self.opt_k = opt_k
        self.cluster_method = cluster_method
        self.n_std = n_std
        self.n_runs = n_runs
        self.use_gpu = use_gpu

        # Construct PCA data path
        percentage_str = f"{int(percentage * 100)}"
        self.pca_final_matrix_dir = os.path.join(
            pca_data_dir, f"pca_{percentage_str}", "final_matrix"
        )

        # Setup logger
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

    def process_subject(self, subject: str) -> dict:
        """
        Process a single subject's PCA-transformed data.

        Args:
            subject: Subject ID

        Returns:
            Dictionary mapping task names to opt_k values
        """
        self.logger.log_info(f"Processing subject: {subject}")
        subj_maps_counts = {}
        res = OrderedDict()

        # Construct subject data directory
        subject_dir = os.path.join(self.pca_final_matrix_dir, subject)
        if not os.path.exists(subject_dir):
            self.logger.log_warning(f"Subject directory not found: {subject_dir}")
            return subj_maps_counts

        # Get all CSV files
        csv_files = [f for f in os.listdir(subject_dir) if f.endswith('.csv')]

        for task in self.task_names:
            task_csv_found = False
            task_csv_file = None

            # Find matching CSV file for this task
            task_search = task.replace(" ", "_")
            for csv_file in csv_files:
                if task_search in csv_file:
                    task_csv_file = os.path.join(subject_dir, csv_file)
                    task_csv_found = True
                    break

            if task_csv_found:
                try:
                    # Load CSV file and transpose (channels x time -> time x channels)
                    task_data_df = self.load_csv_with_index_check(task_csv_file)
                    task_data = task_data_df.to_numpy().T

                    # Ensure proper data format (channels, time)
                    if task_data.shape[0] > task_data.shape[1]:
                        # If time > channels, transpose
                        task_data = task_data.T

                    # Normalize data
                    task_data = np.ascontiguousarray(task_data, dtype=np.float64)

                    # Run microstate clustering
                    batch_params = [
                        task_data,
                        self.peaks_only,
                        self.min_maps,
                        self.max_maps,
                        self.opt_k,
                        self.cluster_method,
                        self.n_std,
                        self.n_runs,
                        self.use_gpu,
                    ]
                    task_microstate = batch_microstate(batch_params)

                    res[task] = {
                        'cv_list': task_microstate.cv_list,
                        'gev_list': task_microstate.gev_list,
                        'maps_list': task_microstate.maps_list,
                        'opt_k': task_microstate.opt_k,
                        'opt_k_index': int(task_microstate.opt_k_index),
                        'min_maps': self.min_maps,
                        'max_maps': self.max_maps
                    }
                    subj_maps_counts[task] = task_microstate.opt_k
                    self.logger.log_info(
                        f'Finished task: {subject}, {task}, opt_k={task_microstate.opt_k}, '
                        f'opt_k_index={task_microstate.opt_k_index}'
                    )
                except Exception as e:
                    self.logger.log_error(f"Error processing {subject}/{task}: {e}")
                    import traceback
                    self.logger.log_error(traceback.format_exc())
            else:
                self.logger.log_warning(f"No CSV file found for task: {task} in subject: {subject}")

        # Save results
        output_file = f'{subject}_pca_individual_maps.json'
        self.dump_to_json(res, self.output_dir, output_file.replace('.json', ''))

        return subj_maps_counts

    def generate_individual_eeg_maps(self, max_processes: Optional[int] = None, save_task_map_counts: bool = True):
        """
        Generate individual EEG maps for all subjects using multiprocessing.

        Args:
            max_processes: Maximum number of worker processes
            save_task_map_counts: Whether to save per-task opt_k counts
        """
        if max_processes is None:
            max_processes = min(len(self.subjects), cpu_count())

        self.logger.log_info(f"Starting PCA individual-run processing with {max_processes} processes")
        self.logger.log_info(f"PCA percentage: {self.percentage}, Subjects: {len(self.subjects)}")

        # Process subjects in parallel using spawn context for cross-platform compatibility
        ctx = get_context("spawn")
        with ctx.Pool(processes=max_processes) as pool:
            results = pool.map(self._process_subject_wrapper, self.subjects)

        # Save task map counts if requested
        if save_task_map_counts:
            task_wise_map_counts = results
            self.dump_to_json(
                task_wise_map_counts,
                self.output_dir,
                'pca_individual_map_counts'
            )

        self.logger.log_info("PCA individual-run processing completed")

    def _process_subject_wrapper(self, subject: str) -> dict:
        """
        Wrapper for process_subject to work with multiprocessing.
        """
        return self.process_subject(subject)

