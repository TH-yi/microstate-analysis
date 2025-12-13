"""
PCA GFP Processing Module
Performs PCA dimensionality reduction on GFP CSV files.
Based on pca-eeg-pipeline/PCA/PCA_GFP.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from multiprocessing import Pool, cpu_count
from typing import List, Optional
from microstate_analysis.logger.dualhandler import DualHandler
from microstate_analysis.microstate_pipeline.pipeline_base import PipelineBase


class PCAGFP(PipelineBase):
    """
    PCA processing for GFP CSV files.
    Performs PCA dimensionality reduction with specified variance retention ratios.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        subjects: List[str],
        percentages: List[float] = [0.95, 0.98, 0.99],
        log_dir: Optional[str] = None,
        log_prefix: str = "pca_gfp",
        log_suffix: str = "",
    ):
        """
        Initialize PCA GFP processor.

        Args:
            input_dir: Directory containing GFP CSV files (structure: {input_dir}/{subject}/*.csv)
            output_dir: Base output directory for PCA results
            subjects: List of subject IDs (e.g., ['P01', 'P02', ...])
            percentages: List of variance retention ratios (e.g., [0.95, 0.98, 0.99])
            log_dir: Optional directory for log files
            log_prefix: Log file prefix
            log_suffix: Log file suffix
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.subjects = subjects
        self.percentages = percentages
        self.logger = DualHandler(log_dir=log_dir, prefix=log_prefix, suffix=log_suffix)

    @staticmethod
    def is_integer(x):
        """Check if a number is an integer."""
        return np.equal(np.mod(x, 1), 0)

    @staticmethod
    def explained_variance_ratio(eigenvalues):
        """Calculate the explained variance ratio from eigenvalues."""
        total_variance = np.sum(eigenvalues)
        return eigenvalues / total_variance

    @staticmethod
    def check_if_header_exists(file_path):
        """Check if the first row contains a header by looking at the 2nd to 4th columns."""
        df = pd.read_csv(file_path, nrows=1, header=None)
        row = df.iloc[0, 1:4].values
        has_header = any(str(x).isupper() for x in row) or all(not isinstance(x, float) for x in row)
        return has_header, row

    def _process_file_worker(self, args):
        """
        Worker function for multiprocessing. Unpacks args and calls process_file.
        """
        subject, file, percentage, input_dir, output_dir = args
        return self._process_file_single(subject, file, percentage, input_dir, output_dir)

    def _process_file_single(self, subject, file, percentage, input_dir, output_dir):
        """
        Process a single CSV file for PCA analysis.

        Args:
            subject: Subject ID
            file: CSV filename
            percentage: Variance retention ratio
            input_dir: Input directory
            output_dir: Output directory
        """
        try:
            # Define output directories for this percentage
            percentage_str = f"{percentage * 100:.0f}"
            target_eigenvalues_folder = os.path.join(
                output_dir, f"pca_{percentage_str}", "eigenvalues", subject
            )
            target_eigenvectors_folder = os.path.join(
                output_dir, f"pca_{percentage_str}", "eigenvectors", subject
            )
            target_final_matrix_folder = os.path.join(
                output_dir, f"pca_{percentage_str}", "final_matrix", subject
            )

            os.makedirs(target_eigenvalues_folder, exist_ok=True)
            os.makedirs(target_eigenvectors_folder, exist_ok=True)
            os.makedirs(target_final_matrix_folder, exist_ok=True)

            # Construct file path
            subject_dir = os.path.join(input_dir, subject)
            file_path = os.path.join(subject_dir, file)

            if not os.path.exists(file_path):
                self.logger.log_warning(f"File not found: {file_path}")
                return

            # Determine if the file has a header
            has_header, row = self.check_if_header_exists(file_path)
            self.logger.log_info(f"{subject}/{file}: Header detected: {has_header}")

            # Load the CSV file
            df = pd.read_csv(file_path, header=0 if has_header else None)

            # Check if the first column is an index by examining values 2-6
            if len(df) > 6:
                first_column_is_index = all(
                    self.is_integer(value) for value in df.iloc[1:6, 0]
                )
            else:
                first_column_is_index = False

            self.logger.log_info(f"{subject}/{file}: First column is index: {first_column_is_index}")

            # Temporarily save the index column
            index_column = None
            if first_column_is_index:
                index_column = df.iloc[:, 0].copy()
                df = df.iloc[:, 1:]

            # Check if the last column is 'GFP' or contains 'GFP'
            last_column_name = df.columns[-1] if hasattr(df.columns[-1], '__str__') else str(df.columns[-1])
            last_column_is_gfp = 'GFP' in str(last_column_name).upper()
            self.logger.log_info(f"{subject}/{file}: Last column is GFP: {last_column_is_gfp}")

            # Remove the last column if it's 'GFP'
            if last_column_is_gfp:
                df = df.iloc[:, :-1]

            # Perform PCA analysis
            pca = PCA()
            pca.fit(df)
            eigenvalues = pca.explained_variance_
            eigenvectors = pca.components_
            cumulative_explained_variance = np.cumsum(self.explained_variance_ratio(eigenvalues))
            n_components = np.argmax(cumulative_explained_variance >= percentage) + 1

            if n_components == 0:
                n_components = len(eigenvalues)  # Use all components if threshold not reached

            reduced_eigenvalues = eigenvalues[:n_components]
            reduced_eigenvectors = eigenvectors[:n_components]

            # Save eigenvalues
            reduced_eigenvalues_df = pd.DataFrame(reduced_eigenvalues, columns=['Eigenvalue'])
            eigenvalues_file_path = os.path.join(
                target_eigenvalues_folder,
                f"{os.path.splitext(file)[0]}_reduced_eigenvalues_gfp_pca{percentage_str}.csv"
            )
            reduced_eigenvalues_df.to_csv(eigenvalues_file_path, index=False)

            # Save eigenvectors
            reduced_eigenvectors_df = pd.DataFrame(
                reduced_eigenvectors,
                columns=[f'Eigenvector_{i + 1}' for i in range(reduced_eigenvectors.shape[1])]
            )
            eigenvectors_file_path = os.path.join(
                target_eigenvectors_folder,
                f"{os.path.splitext(file)[0]}_reduced_eigenvectors_gfp_pca{percentage_str}.csv"
            )
            reduced_eigenvectors_df.to_csv(eigenvectors_file_path, index=False)

            # Compute the final matrix
            final_matrix = np.dot(df.values, reduced_eigenvectors.T)

            # Restore the index column if it existed
            if first_column_is_index and index_column is not None:
                final_matrix_df = pd.DataFrame(
                    final_matrix,
                    index=index_column,
                    columns=[f'PC{i + 1}' for i in range(n_components)]
                )
            else:
                final_matrix_df = pd.DataFrame(
                    final_matrix,
                    columns=[f'PC{i + 1}' for i in range(n_components)]
                )

            # Save the final matrix
            final_matrix_file_path = os.path.join(
                target_final_matrix_folder,
                f"{os.path.splitext(file)[0]}_final_matrix_gfp_pca{percentage_str}.csv"
            )
            final_matrix_df.to_csv(final_matrix_file_path, index=True)

            self.logger.log_info(
                f"Processed {subject}/{file} successfully. "
                f"Components: {n_components}, Variance retained: {cumulative_explained_variance[n_components-1]:.4f}"
            )

        except Exception as e:
            self.logger.log_error(f"Error processing {subject}/{file}: {e}")
            import traceback
            self.logger.log_error(traceback.format_exc())

    def run(self, max_processes: Optional[int] = None):
        """
        Run PCA processing on all files.

        Args:
            max_processes: Maximum number of worker processes. If None, uses cpu_count().
        """
        if max_processes is None:
            max_processes = cpu_count()

        # Collect all files to process
        files_to_process = []
        for subject in self.subjects:
            subject_dir = os.path.join(self.input_dir, subject)
            if not os.path.exists(subject_dir):
                self.logger.log_warning(f"Subject directory not found: {subject_dir}")
                continue

            csv_files = [f for f in os.listdir(subject_dir) if f.endswith('.csv')]
            if not csv_files:
                self.logger.log_warning(f"No CSV files found in {subject_dir}")
                continue

            for file in csv_files:
                for percentage in self.percentages:
                    files_to_process.append((subject, file, percentage, self.input_dir, self.output_dir))

        if not files_to_process:
            self.logger.log_error("No files to process!")
            return

        self.logger.log_info(
            f"Starting PCA processing: {len(files_to_process)} tasks, "
            f"{max_processes} processes, percentages: {self.percentages}"
        )

        # Process files in parallel
        # Use a lambda to bind the instance method
        with Pool(processes=max_processes) as pool:
            pool.map(self._process_file_worker, files_to_process)

        self.logger.log_info("PCA processing completed.")

