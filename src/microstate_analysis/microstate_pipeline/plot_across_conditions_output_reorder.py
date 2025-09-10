"""
Author: Tianhao Ning
Description:
  Plot topographies for ACROSS-CONDITIONS output and optionally reorder maps,
  with all parameters explicit (no config file). Robust logging and I/O helpers.

Expected input JSON (across-conditions):
{
  "idea_generation": { "maps": [...], "label": ..., "mean_similarity": ..., "std_similarity": ... },
  "idea_evolution": { "maps": [...], ... },
  "idea_rating":    { "maps": [...], ... },
  "rest":           { "maps": [...], ... }
}

This script will:
  1) Load the JSON (or accept a 'data' dict),
  2) Plot and save figures,
  3) Compute a new order from plot_eegmaps,
  4) Reorder maps via reorder_maps and dump the reordered JSON.
"""

import os
import json
from typing import Dict, List, Any

from microstate_analysis.microstate_pipeline.pipeline_base import PipelineBase
from microstate_analysis.logger.dualhandler import DualHandler
from microstate_analysis.microstate_base.plot_gen import plot_eegmaps, plot_eegmap_conditions, plot_eegmap_one_row
from microstate_analysis.reorder_maps.reorder_maps import reorder_maps


class PlotAcrossConditionsOutput(PipelineBase):
    """
    All parameters explicit; no config object needed.
    """

    def __init__(
        self,
        input_json_path: str,
        output_img_dir: str,
        reordered_json_path: str,
        conditions: List[str],
        first_row_order=None,
        log_dir=None,
        log_prefix: str = "plot_across_conditions",
        log_suffix: str = "",
    ):
        super().__init__()
        self.input_json_path = input_json_path
        self.output_img_dir = output_img_dir
        self.reordered_json_path = reordered_json_path
        self.conditions = conditions
        self.first_row_order = first_row_order  # e.g., [3,5,0,4,2,1]; None = use default inside plotter
        # Logger config & instance (rebuildable in child processes if needed)
        self._logger_cfg = dict(log_dir=log_dir, prefix=log_prefix or "", suffix=log_suffix or "")
        self.logger = DualHandler(**self._logger_cfg)

    # ----- pickling-safe logger hooks (for consistency with your other stages) -----

    def __getstate__(self):
        state = self.__dict__.copy()
        state["logger"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = DualHandler(**self._logger_cfg)

    # ----- I/O helpers -----

    def _ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    def _load_data(self) -> Dict[str, Any]:
        self.logger.log_info(f"Loading input JSON: {self.input_json_path}")
        with open(self.input_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_json(self, data_obj: Dict[str, Any], json_path: str):
        out_dir = os.path.dirname(json_path)
        if out_dir:
            self._ensure_dir(out_dir)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_obj, f, ensure_ascii=False, indent=4)
        self.logger.log_info(f"Saved JSON: {json_path}")

    # ----- main -----

    def plot_and_reorder(self):
        """
        Plot condition maps (grid figure) and reorder them by the indices returned from plot_eegmaps.
        If 'data' is None, loads from self.input_json_path.
        """
        data = self._load_data()
        # Basic validation (ensure expected keys exist)
        if "maps" not in data:
            self.logger.log_error(f"Missing maps in data")
            raise KeyError(f"Missing maps in data")

        # Ensure output image directory exists
        self._ensure_dir(self.output_img_dir)

        # 1) Plot and get order
        self.logger.log_info("Plotting EEG maps across conditions...")
        order = plot_eegmaps(
            data,
            ["maps"],
            first_row_order=self.first_row_order if self.first_row_order is not None else [3, 5, 0, 4, 2, 1],
            savepath=self.output_img_dir,
        )
        self.logger.log_info(f"Computed order from plot_eegmaps: {order}")

        # 2) Reorder maps and dump JSON
        self.logger.log_info("Reordering EEG maps based on computed indices...")
        reordered_data = reorder_maps(data, order, one_task=True)
        self._save_json(reordered_data, self.reordered_json_path)
        self.logger.log_info("Plot & reorder pipeline finished.")


if __name__ == "__main__":
    # ----------------- EDIT YOUR PARAMETERS HERE (no config file) -----------------

    # I/O paths
    input_json_path = "../../../storage/microstate_output/across_conditions/across_conditions.json"
    output_img_dir = "../../../storage/microstate_output/across_conditions/plots"  # folder to save images
    reordered_json_path = "../../../storage/microstate_output/across_conditions/across_conditions_reordered.json"

    # Conditions (the keys expected in the input JSON)
    conditions = ["idea_generation", "idea_evolution", "idea_rating", "rest"]

    # Plot options
    # If you want to keep the exact behavior from your original script, use [3,5,0,4,2,1].
    # Set to None to defer to the plotting function's default if it has one.
    first_row_order = [3, 5, 0, 4, 2, 1]

    # Logging
    log_dir = "../../../storage/log/plot_across_conditions"
    log_prefix = "plot_across_conditions"
    log_suffix = ""

    # ----------------- RUN -----------------
    job = PlotAcrossConditionsOutput(
        input_json_path=input_json_path,
        output_img_dir=output_img_dir,
        reordered_json_path=reordered_json_path,
        conditions=conditions,
        first_row_order=first_row_order,
        log_dir=log_dir,
        log_prefix=log_prefix,
        log_suffix=log_suffix,
    )
    job.logger.log_info("Start plot_across_conditions pipeline")
    job.plot_and_reorder()
    job.logger.log_info("Finish plot_across_conditions pipeline")
