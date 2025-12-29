"""
PCA Across Conditions Pipeline
Aggregates conditions into global maps.
Input: Across-subjects JSON file
Output: Single across-conditions JSON
"""

import os
from collections import OrderedDict
from typing import List, Optional

from microstate_analysis.pca_microstate_pipeline.pca_pipeline_base import PCAPipelineBase
from microstate_analysis.logger.dualhandler import DualHandler
from microstate_analysis.microstate_base.data_handler import load_data
from microstate_analysis.microstate_base.meanmicrostate import MeanMicrostate


class PCAPipelineAcrossConditions(PCAPipelineBase):
    """
    PCA Across Conditions Pipeline.
    Aggregates conditions into global microstate maps.
    """

    def __init__(
        self,
        input_dir: str,
        input_name: str,
        output_dir: str,
        output_name: str,
        condition_names: List[str],
        percentage: float,
        n_k: int = 6,
        n_ch: int = 63,
        log_dir: Optional[str] = None,
        log_prefix: str = "pca_across_conditions",
        log_suffix: str = "",
        use_gpu: bool = False,
    ):
        """
        Initialize PCA Across Conditions Pipeline.

        Args:
            input_dir: Directory containing across-subjects JSON
            input_name: Input filename
            output_dir: Output directory
            output_name: Output filename
            condition_names: List of condition names
            percentage: PCA percentage
            n_k: Number of microstates
            n_ch: Number of channels
            log_dir: Optional directory for log files
            log_prefix: Log file prefix
            log_suffix: Log file suffix
        """
        super().__init__()
        self.input_dir = os.path.join(input_dir, f"pca_{int(percentage*100)}")
        self.input_name = input_name
        self.output_dir = os.path.join(output_dir, f"pca_{int(percentage*100)}")
        self.output_name = output_name
        self.condition_names = condition_names
        self.percentage = percentage
        self.n_k = n_k
        self.n_ch = n_ch
        self.use_gpu = use_gpu

        self.logger = DualHandler(log_dir=log_dir, prefix=log_prefix, suffix=log_suffix)

    def run(self):
        """
        Run across-conditions processing.
        """
        self.logger.log_info(
            f"Starting PCA across-conditions processing: {len(self.condition_names)} conditions, "
            f"percentage: {self.percentage}"
        )

        # Load across-subjects data
        input_path = os.path.join(self.input_dir, self.input_name)
        if not os.path.exists(input_path):
            self.logger.log_error(f"Input file not found: {input_path}")
            return

        data = load_data(input_path)

        # Collect maps for each condition
        maps = []
        for condition_name in self.condition_names:
            self.logger.log_info(f"Processing condition: {condition_name}")
            if condition_name in data:
                maps.append(data[condition_name]['maps'])
            else:
                self.logger.log_warning(f"Condition {condition_name} not found in input data")

        if not maps:
            self.logger.log_error("No maps found for any condition")
            return

        # Aggregate maps across conditions
        microstate = MeanMicrostate(maps, self.n_k, self.n_ch, len(maps), use_gpu=self.use_gpu)
        temp = microstate.mean_microstates()

        if not isinstance(temp[0], list):
            temp[0] = temp[0].tolist()
        res = {
            'maps': temp[0],
            'label': temp[1],
            'mean_similarity': temp[2],
            'std_similarity': temp[3]
        }

        # Save result
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, self.output_name)
        self.dump_to_json_path(res, output_path)

        self.logger.log_info("PCA across-conditions processing completed")

