"""
Author: Tianhao Ning
Description:
  Aggregate microstate maps across CONDITIONS (single input JSON that already
  contains condition-level maps, e.g., produced by the Across-Subjects stage).
  All parameters are explicit (no config file). Robust logging and I/O helpers.

Usage:
  - Edit parameters in the __main__ block and run directly.
  - Expected input JSON structure:
      {
        "idea_generation": { "maps": [...], "label": ..., "mean_similarity": ..., "std_similarity": ... },
        "idea_evolution": { "maps": [...] , ... },
        "idea_rating":    { "maps": [...] , ... },
        "rest":           { "maps": [...] , ... }
      }
  - The script will read each condition's "maps" and aggregate them to produce a
    single across-conditions result JSON.
"""

import os
import json
from typing import List, Dict

from microstate_analysis.microstate_pipeline.pipeline_base import PipelineBase
from microstate_analysis.logger.dualhandler import DualHandler
from microstate_analysis.microstate_base.data_handler import load_data
from microstate_analysis.microstate_base.meanmicrostate import MeanMicrostate


class PipelineAcrossConditions(PipelineBase):
    """
    Compute a single across-conditions microstate result from one input JSON file
    that already includes condition-level maps. Parameters are explicit.
    """

    def __init__(
        self,
        input_dir: str,
        input_name: str,
        output_dir: str,
        output_name: str,
        condition_names: List[str],
        n_k: int = 6,
        n_ch: int = 63,
        log_dir=None,
        log_prefix: str = "across_conditions",
        log_suffix: str = "",
    ):
        super().__init__()
        # Paths & params
        self.input_dir = input_dir
        self.input_name = input_name          # e.g., "across_subjects.json"
        self.output_dir = output_dir
        self.output_name = output_name        # e.g., "across_conditions.json"
        self.condition_names = condition_names
        self.n_k = n_k
        self.n_ch = n_ch

        # Logger (reconstructable in child if ever refactored to MP)
        self._logger_cfg = dict(log_dir=log_dir, prefix=log_prefix or "", suffix=log_suffix or "")
        self.logger = DualHandler(**self._logger_cfg)

    # ---------- pickling-safe logger hooks (for consistency with other stages) ----------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["logger"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = DualHandler(**self._logger_cfg)

    # ---------- IO helpers ----------
    def _dump_json(self, data_obj: Dict, out_dir: str, filename: str):
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_obj, f, ensure_ascii=False, indent=4)
        self.logger.log_info(f"Saved JSON: {path}")

    def _load_input(self) -> Dict:
        in_path = os.path.abspath(os.path.join(self.input_dir, self.input_name))
        self.logger.log_info(f"Loading across-subjects input: {in_path}")
        return load_data(in_path)

    # ---------- main compute ----------
    def run(self):
        """
        Read the per-condition maps from a single input JSON and aggregate
        them into one across-conditions microstate result.
        """
        data = self._load_input()

        # Collect maps per condition (order defined by self.condition_names)
        maps = []
        for cond in self.condition_names:
            self.logger.log_info(f"Collect maps for condition: {cond}")
            try:
                maps.append(data[cond]["maps"])
            except KeyError as e:
                self.logger.log_error(f"Condition '{cond}' missing or malformed in input JSON: {e}")
                raise

        micro = MeanMicrostate(maps, self.n_k, self.n_ch, len(self.condition_names))
        m_maps, m_label, m_mean, m_std = micro.mean_microstates()

        result = {
            "maps": m_maps.tolist(),
            "label": m_label,
            "mean_similarity": m_mean,
            "std_similarity": m_std,
        }

        # from eeg_code.microstate_base.microstate_batch_handler import batch_mean_microstate
        # agg = batch_mean_microstate([maps, self.n_k, self.n_ch, len(self.condition_names)])
        # result = {
        #     "maps": agg["maps"].tolist(),
        #     "label": agg["label"],
        #     "mean_similarity": agg["mean_similarity"],
        #     "std_similarity": agg["std_similarity"],
        # }

        # Save
        self._dump_json(result, self.output_dir, self.output_name)
        self.logger.log_info("Across-conditions aggregation finished.")


if __name__ == "__main__":
    # ----------------- EDIT YOUR PARAMETERS HERE (no config file) -----------------

    # I/O
    input_dir = "../../../storage/microstate_output/across_subjects"
    input_name = "across_subjects.json"
    output_dir = "../../../storage/microstate_output/across_conditions"
    output_name = "across_conditions.json"

    # Conditions
    condition_names = ["idea_generation", "idea_evolution", "idea_rating", "rest"]

    # Microstate params
    n_k = 6
    n_ch = 63

    # Logging
    log_dir = "../../../storage/log/across_conditions"
    log_prefix = "across_conditions"
    log_suffix = ""

    # ----------------- RUN -----------------
    job = PipelineAcrossConditions(
        input_dir=input_dir,
        input_name=input_name,
        output_dir=output_dir,
        output_name=output_name,
        condition_names=condition_names,
        n_k=n_k,
        n_ch=n_ch,
        log_dir=log_dir,
        log_prefix=log_prefix,
        log_suffix=log_suffix,
    )
    job.logger.log_info("Start across_conditions pipeline")
    job.run()
    job.logger.log_info("Finish across_conditions pipeline")
