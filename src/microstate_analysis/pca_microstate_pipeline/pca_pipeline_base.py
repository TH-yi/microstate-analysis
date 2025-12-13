"""
PCA Microstate Pipeline Base Class
Provides common functionality for PCA-based microstate pipelines.
"""

import os
import pandas as pd
import numpy as np
from microstate_analysis.microstate_pipeline.pipeline_base import PipelineBase


class PCAPipelineBase(PipelineBase):
    """
    Base class for PCA microstate pipelines.
    Provides CSV loading utilities for PCA-processed data.
    """

    @staticmethod
    def is_index(lst):
        """Check if a list contains index-like values (integers or alphabetic strings)."""
        if all(isinstance(x, int) for x in lst):
            return True
        if all(isinstance(x, str) and x.isalpha() for x in lst):
            return True
        return False

    def load_csv_with_index_check(self, csv_file):
        """
        Load CSV file and handle index row/column detection.
        Based on pca-eeg-pipeline/pca_pipeline_base.py
        """
        # Read first few rows to detect index row/column
        sample_data = pd.read_csv(csv_file, nrows=6)

        # Check if first column is index column
        if len(sample_data) > 5:
            first_column = sample_data.iloc[1:5, 0].tolist()
            has_index_column = self.is_index(first_column)
        else:
            has_index_column = False

        # Check if first row is index row
        if len(sample_data.columns) > 1:
            first_row = sample_data.columns.tolist()[1:]
            has_index_row = self.is_index(first_row)
        else:
            has_index_row = False

        if has_index_row and has_index_column:
            return pd.read_csv(csv_file, index_col=0, header=0)
        elif has_index_column:
            return pd.read_csv(csv_file, index_col=0)
        elif has_index_row:
            return pd.read_csv(csv_file, header=0)
        else:
            return pd.read_csv(csv_file, header=None)

