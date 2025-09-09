"""
Author: Tianhao Ning
Description:
  Plot topographies for ACROSS-SUBJECTS output and reorder maps. All parameters are
  explicit (no config file). Includes robust logging and I/O helpers.

Expected input JSON (across-subjects; one file containing all conditions):
{
  "idea_generation": { "maps": [...], "label": ..., "mean_similarity": ..., "std_similarity": ... },
  "idea_evolution": { "maps": [...], ... },
  "idea_rating":    { "maps": [...], ... },
  "rest":           { "maps": [...], ... }
}

Pipeline:
  1) Load input JSON.
  2) Plot grid via plot_eegmaps (saving images to output_img_dir).
  3) Use returned order indices to reorder maps.
  4) Dump reordered JSON.
"""

import os
import json
from typing import List, Dict, Any

from microstate_analysis.microstate_pipeline.pipeline_base import PipelineBase
from microstate_analysis.logger.dualhandler import DualHandler
from microstate_analysis.microstate_base.plot_gen import plot_eegmaps
from microstate_analysis.reorder_maps.reorder_maps import reorder_maps


class PlotAcrossSubjectsOutput(PipelineBase):
    """
    Across-subjects plotting & reordering with explicit parameters.
    """

    def __init__(
        self,
        input_json_path: str,
        output_img_dir: str,
        reordered_json_path: str,
        conditions: List[str],
        map_condition_name: List[str] = None,   # e.g., ["maps"]
        first_row_order=None,   # e.g., [3,5,4,1,0,2]
        log_dir=None,
        log_prefix: str = "plot_across_subjects",
        log_suffix: str = "",
    ):
        super().__init__()
        self.input_json_path = input_json_path
        self.output_img_dir = output_img_dir
        self.reordered_json_path = reordered_json_path
        self.conditions = conditions
        self.map_condition_name = map_condition_name or ["maps"]
        self.first_row_order = first_row_order or [3, 5, 4, 1, 0, 2]

        # Logger config & instance (rebuildable in child processes if later extended)
        self._logger_cfg = dict(log_dir=log_dir, prefix=log_prefix or "", suffix=log_suffix or "")
        self.logger = DualHandler(**self._logger_cfg)

    # ----- pickling-safe logger hooks (for parity with other stages) -----

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

    def _load_json(self, path: str) -> Dict[str, Any]:
        self.logger.log_info(f"Loading input JSON: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_json(self, data_obj: Dict[str, Any], path: str):
        out_dir = os.path.dirname(path)
        if out_dir:
            self._ensure_dir(out_dir)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_obj, f, ensure_ascii=False, indent=4)
        self.logger.log_info(f"Saved JSON: {path}")

    # ----- validation -----

    def _validate(self, data: Dict[str, Any]):
        for cond in self.conditions:
            if cond not in data:
                self.logger.log_error(f"Condition '{cond}' is missing in input JSON.")
                raise KeyError(f"Missing condition: {cond}")
            if "maps" not in data[cond]:
                self.logger.log_error(f"Condition '{cond}' has no 'maps' field.")
                raise KeyError(f"Missing 'maps' in condition: {cond}")

    # ----- main -----

    def plot_and_reorder(self):
        """
        Plot across-subjects maps and reorder them by indices returned from plot_eegmaps.
        If 'data' is None, it loads from self.input_json_path.
        """
        data = self._load_json(self.input_json_path)

        self._validate(data)
        self._ensure_dir(self.output_img_dir)

        # 1) Plot grid & compute order
        self.logger.log_info("Plotting EEG maps across subjects...")
        order = plot_eegmaps(
            data,
            self.map_condition_name,                   # usually ["maps"]
            first_row_order=self.first_row_order,      # default [3,5,4,1,0,2]
            savepath=self.output_img_dir,
        )
        self.logger.log_info(f"Computed order from plot_eegmaps: {order}")

        # 2) Reorder maps and dump JSON
        self.logger.log_info("Reordering EEG maps based on computed indices...")
        reordered_data = reorder_maps(data, order)     # across-subjects: default usage (no one_task flag)
        self._save_json(reordered_data, self.reordered_json_path)

        self.logger.log_info("Across-subjects plotting & reordering finished.")


if __name__ == "__main__":
    # ----------------- EDIT YOUR PARAMETERS HERE (no config file) -----------------

    # I/O
    input_json_path = "../../../storage/microstate_output/across_subjects/across_subjects.json"
    output_img_dir = "../../../storage/microstate_output/across_subjects/plots"
    reordered_json_path = "../../../storage/microstate_output/across_subjects/across_subjects_reordered.json"

    # Conditions as they appear in the input JSON
    conditions = ["idea_generation", "idea_evolution", "idea_rating", "rest"]

    # Plotting options
    map_condition_name = ["maps"]            # key(s) used by plot_eegmaps
    first_row_order = [3, 5, 4, 1, 0, 2]     # preserve your original choice

    # Logging
    log_dir = "../../../storage/log/plot_across_subjects"
    log_prefix = "plot_across_subjects"
    log_suffix = ""

    # ----------------- RUN -----------------
    job = PlotAcrossSubjectsOutput(
        input_json_path=input_json_path,
        output_img_dir=output_img_dir,
        reordered_json_path=reordered_json_path,
        conditions=conditions,
        map_condition_name=map_condition_name,
        first_row_order=first_row_order,
        log_dir=log_dir,
        log_prefix=log_prefix,
        log_suffix=log_suffix,
    )
    job.logger.log_info("Start plot_across_subjects pipeline")
    job.plot_and_reorder()
    job.logger.log_info("Finish plot_across_subjects pipeline")
