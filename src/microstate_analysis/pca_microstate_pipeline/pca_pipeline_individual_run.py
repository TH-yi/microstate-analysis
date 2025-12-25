"""
PCA Individual Run Pipeline
Complete end-to-end pipeline: Raw JSON → GFP peaks → PCA → Microstate clustering.
Input: Raw JSON files (e.g., sub_01.json)
Output: Individual microstate maps per subject per task
"""

import os
import json
import numpy as np
import pandas as pd
from collections import OrderedDict
from multiprocessing import Pool, cpu_count, get_context
from typing import List, Optional, Tuple
from scipy import signal
from sklearn.decomposition import PCA

from microstate_analysis.pca_microstate_pipeline.pca_pipeline_base import PCAPipelineBase
from microstate_analysis.logger.dualhandler import DualHandler
from microstate_analysis.microstate_base.data_handler import list_to_matrix
from microstate_analysis.microstate_base.microstate_batch_handler import batch_microstate


class PCAPipelineIndividualRun(PCAPipelineBase):
    """
    PCA Individual Run Pipeline.
    Complete end-to-end pipeline: Raw JSON → GFP peaks → PCA → Microstate clustering.
    """

    def __init__(
        self,
        input_dir: str,
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
        gfp_distance: int = 10,
        gfp_n_std: int = 3,
        pca_output_dir: Optional[str] = None,
        save_pca_intermediate: bool = True,
        log_dir: Optional[str] = None,
        log_prefix: str = "pca_individual_run",
        log_suffix: str = "",
        use_gpu: bool = False,
    ):
        """
        Initialize PCA Individual Run Pipeline.

        Args:
            input_dir: Directory containing raw per-subject JSON files (e.g., sub_01.json)
            output_dir: Output directory for individual maps
            subjects: List of subject IDs
            task_names: List of task names
            percentage: PCA variance retention ratio (e.g., 0.95, 0.98, 0.99)
            peaks_only: Use peaks-only logic for microstate clustering
            min_maps: Minimum number of maps
            max_maps: Maximum number of maps
            opt_k: Optional fixed K value
            cluster_method: Clustering method
            n_std: Threshold std for microstate clustering
            n_runs: Clustering restarts
            gfp_distance: Minimum distance between GFP peaks
            gfp_n_std: Number of standard deviations for GFP peak thresholding
            pca_output_dir: Optional directory to save PCA intermediate results
            save_pca_intermediate: Whether to save PCA intermediate results
            log_dir: Optional directory for log files
            log_prefix: Log file prefix
            log_suffix: Log file suffix
            use_gpu: Enable GPU acceleration
        """
        super().__init__()
        self.input_dir = input_dir
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
        self.gfp_distance = gfp_distance
        self.gfp_n_std = gfp_n_std
        self.pca_output_dir = pca_output_dir
        self.save_pca_intermediate = save_pca_intermediate
        self.use_gpu = use_gpu

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

    def calculate_gfp_peaks(self, data: np.ndarray, distance: int = 10, n_std: int = 3):
        """
        Calculate GFP and find peaks.

        Args:
            data: 2D numpy array (time, channels)
            distance: Minimum distance between peaks
            n_std: Number of standard deviations for thresholding

        Returns:
            peaks: Indices of GFP peaks
            gfp: GFP values
        """
        # Calculate GFP: sqrt of mean of squared values across channels
        gfp = np.sqrt(np.mean(data ** 2, axis=1))

        # Find peaks
        height_low = gfp.mean() - n_std * gfp.std()
        height_high = gfp.mean() + n_std * gfp.std()
        peaks, _ = signal.find_peaks(gfp, distance=distance, height=(height_low, height_high))
        return peaks, gfp

    def apply_pca(self, data: np.ndarray, percentage: float, save_path: Optional[str] = None) -> Tuple[np.ndarray, int, PCA]:
        """
        Apply PCA dimensionality reduction.

        Args:
            data: 2D numpy array (time, channels)
            percentage: Variance retention ratio
            save_path: Optional path to save PCA results (without extension)

        Returns:
            transformed_data: PCA-transformed data (time, n_components)
            n_components: Number of components retained
            pca_model: Fitted PCA model
        """
        pca = PCA()
        pca.fit(data)
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_
        explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        n_components = np.argmax(cumulative_explained_variance >= percentage) + 1

        if n_components == 0:
            n_components = len(eigenvalues)

        # Transform data
        pca_reduced = PCA(n_components=n_components)
        transformed_data = pca_reduced.fit_transform(data)

        # Save intermediate results if requested
        if save_path and self.save_pca_intermediate:
            percentage_str = f"{int(percentage * 100)}"
            try:
                # Extract base directory and filename from save_path
                save_dir = os.path.dirname(save_path)
                save_filename = os.path.basename(save_path)
                
                # Get parent directories for eigenvalues and eigenvectors
                final_matrix_dir = save_dir
                eigenvalues_dir = final_matrix_dir.replace("final_matrix", "eigenvalues")
                eigenvectors_dir = final_matrix_dir.replace("final_matrix", "eigenvectors")
                
                # Ensure directories exist
                os.makedirs(eigenvalues_dir, exist_ok=True)
                os.makedirs(eigenvectors_dir, exist_ok=True)
                
                # Save eigenvalues
                eigenvalues_df = pd.DataFrame(eigenvalues[:n_components], columns=['Eigenvalue'])
                eigenvalues_path = os.path.join(
                    eigenvalues_dir,
                    f"{save_filename}_reduced_eigenvalues_gfp_pca{percentage_str}.csv"
                )
                eigenvalues_df.to_csv(eigenvalues_path, index=False)

                # Save eigenvectors
                eigenvectors_df = pd.DataFrame(
                    eigenvectors[:n_components],
                    columns=[f'Eigenvector_{i + 1}' for i in range(eigenvectors.shape[1])]
                )
                eigenvectors_path = os.path.join(
                    eigenvectors_dir,
                    f"{save_filename}_reduced_eigenvectors_gfp_pca{percentage_str}.csv"
                )
                eigenvectors_df.to_csv(eigenvectors_path, index=False)

                # Save final matrix (PCA-transformed data)
                final_matrix_df = pd.DataFrame(
                    transformed_data,
                    columns=[f'PC{i + 1}' for i in range(n_components)]
                )
                # Add index starting from 1
                final_matrix_df.index = final_matrix_df.index + 1
                final_matrix_path = os.path.join(
                    final_matrix_dir,
                    f"{save_filename}_final_matrix_gfp_pca{percentage_str}.csv"
                )
                final_matrix_df.to_csv(final_matrix_path, index=True)
            except Exception as e:
                self.logger.log_warning(f"Failed to save PCA intermediate results to {save_path}: {e}")
                import traceback
                self.logger.log_warning(traceback.format_exc())

        return transformed_data, n_components, pca

    def process_subject(self, subject: str) -> dict:
        """
        Process a single subject: load JSON → GFP peaks → PCA → Microstate clustering.

        Args:
            subject: Subject ID

        Returns:
            Dictionary mapping task names to opt_k values
        """
        self.logger.log_info(f"Processing subject: {subject}")
        subj_maps_counts = {}
        res = OrderedDict()

        # Load raw JSON data
        json_path = os.path.join(self.input_dir, f"{subject}.json")
        if not os.path.exists(json_path):
            self.logger.log_warning(f"Subject JSON file not found: {json_path}")
            return subj_maps_counts

        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        # Process each task
        for task in self.task_names:
            if task not in raw_data:
                self.logger.log_warning(f"Task {task} not found in {subject}.json")
                continue

            try:
                # Step 1: Load task data and convert to numpy array
                task_data = list_to_matrix(raw_data[task])
                # Ensure shape is (time, channels)
                if task_data.shape[0] < task_data.shape[1]:
                    # If channels > time, transpose
                    task_data = task_data.T
                task_data = np.ascontiguousarray(task_data, dtype=np.float64)

                self.logger.log_info(f"{subject}/{task}: Original shape: {task_data.shape}")

                # Step 2: Calculate GFP and find peaks
                peaks, gfp_values = self.calculate_gfp_peaks(
                    task_data, distance=self.gfp_distance, n_std=self.gfp_n_std
                )
                if len(peaks) == 0:
                    self.logger.log_warning(f"{subject}/{task}: No GFP peaks found")
                    continue

                # Extract peaks data
                peaks_data = task_data[peaks, :]
                self.logger.log_info(
                    f"{subject}/{task}: GFP peaks found: {len(peaks)}/{len(task_data)} "
                    f"({100*len(peaks)/len(task_data):.1f}%)"
                )

                # Step 3: Apply PCA dimensionality reduction
                # Prepare save path for intermediate results
                save_path = None
                if self.save_pca_intermediate:
                    if self.pca_output_dir:
                        pca_base_dir = self.pca_output_dir
                    else:
                        # Use default structure: output_dir/../pca_output
                        pca_base_dir = os.path.join(os.path.dirname(self.output_dir), "pca_output")
                    
                    percentage_str = f"{int(self.percentage * 100)}"
                    final_matrix_dir = os.path.join(
                        pca_base_dir, f"pca_{percentage_str}", "final_matrix", subject
                    )
                    os.makedirs(final_matrix_dir, exist_ok=True)
                    
                    # Create filename from task name (replace spaces with underscores)
                    # Format should match: {subject}_{task}_gfp_final_matrix_gfp_pca{percentage}.csv
                    task_filename = task.replace(" ", "_")
                    full_filename = f"{subject}_{task_filename}_gfp_final_matrix_gfp_pca{percentage_str}"
                    save_path = os.path.join(final_matrix_dir, full_filename)

                pca_data, n_components, pca_model = self.apply_pca(
                    peaks_data, self.percentage, save_path=save_path
                )
                self.logger.log_info(
                    f"{subject}/{task}: PCA reduction: {peaks_data.shape[1]} → {n_components} components "
                    f"({self.percentage*100:.0f}% variance)"
                )
                if save_path:
                    self.logger.log_info(
                        f"{subject}/{task}: PCA intermediate results saved to {save_path}_*.csv"
                    )

                # Step 4: Transpose for microstate clustering (channels, time)
                # Microstate expects (channels, time) format
                microstate_data = pca_data.T
                microstate_data = np.ascontiguousarray(microstate_data, dtype=np.float64)

                # Step 5: Run microstate clustering
                batch_params = [
                    microstate_data,
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
                    'max_maps': self.max_maps,
                    'pca_n_components': int(n_components),
                }
                subj_maps_counts[task] = task_microstate.opt_k
                self.logger.log_info(
                    f'Finished task: {subject}, {task}, opt_k={task_microstate.opt_k}, '
                    f'opt_k_index={task_microstate.opt_k_index}, PCA components={n_components}'
                )
            except Exception as e:
                self.logger.log_error(f"Error processing {subject}/{task}: {e}")
                import traceback
                self.logger.log_error(traceback.format_exc())

        # Save results
        output_file = f'{subject}_pca_individual_maps.json'
        self.dump_to_json(res, self.output_dir, output_file.replace('.json', ''))

        return subj_maps_counts

    def run(self, max_processes: Optional[int] = None, save_task_map_counts: bool = True):
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

