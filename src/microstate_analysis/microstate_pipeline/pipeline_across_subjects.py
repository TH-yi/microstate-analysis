"""
Author: Tianhao Ning
Description:
  Aggregate microstate maps across subjects for each condition with fully explicit
  parameters (no config file). Uses smart multiprocessing (spawn, capped workers)
  and rebuilds logger in child processes.

Usage:
  - Edit parameters in the __main__ block and run directly.
  - Each subject JSON should contain condition-level 'maps' produced earlier, e.g.:
      data[condition_name]['maps']  # array-like (n_ch x n_k)
"""

import os
import json
from collections import OrderedDict
from multiprocessing import get_context

from typing import List, Tuple

from microstate_analysis.microstate_pipeline.pipeline_base import PipelineBase
from microstate_analysis.logger.dualhandler import DualHandler
from microstate_analysis.microstate_base.data_handler import load_data
from microstate_analysis.microstate_base.microstate_batch_handler import batch_mean_microstate


class PipelineAcrossSubjects(PipelineBase):
    """
    Computes condition-level mean microstate maps across subjects.
    All parameters are explicit; no config object is required.
    """

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        subjects: List[str],
        data_suffix: str,
        save_name: str,
        condition_names: List[str],
        n_k: int = 6,
        n_ch: int = 63,
        log_dir=None,
        log_prefix: str = "across_subjects",
        log_suffix: str = "",
        use_gpu: bool = False
    ):
        super().__init__()
        # Core params
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.subjects = subjects
        self.data_suffix = data_suffix            # subject + data_suffix is input file
        self.save_name = save_name                # a single JSON file for all conditions
        self.condition_names = condition_names
        self.n_k = n_k
        self.n_ch = n_ch
        self.use_gpu = use_gpu

        # Logger config for safe reconstruction in child processes
        self._logger_cfg = dict(log_dir=log_dir, prefix=log_prefix or "", suffix=log_suffix or "")
        self.logger = DualHandler(**self._logger_cfg)


    # ---------- pickling-safe logger ----------

    def __getstate__(self):
        state = self.__dict__.copy()
        state["logger"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = DualHandler(**self._logger_cfg)

    # ---------- per-condition worker (runs in child) ----------

    def _process_one_condition(self, cond: str) -> Tuple[str, dict]:
        """
        Aggregate maps across subjects for one condition.
        Expects each subject JSON to have data[cond]['maps'].
        Returns (condition_name, result_dict).
        """
        self.logger.log_info(f"Start condition: {cond}")

        maps = []
        for subj in self.subjects:
            self.logger.log_info(f"Read subject: {subj} for condition: {cond}")
            data_path = os.path.abspath(os.path.join(self.input_dir, subj + self.data_suffix))
            data = load_data(data_path)
            maps.append(data[cond]["maps"])

        # Aggregate across subjects
        agg = batch_mean_microstate([maps, self.n_k, self.n_ch, len(self.subjects), self.use_gpu])
        result = {
            "maps": agg["maps"],
            "label": agg["label"],
            "mean_similarity": agg["mean_similarity"],
            "std_similarity": agg["std_similarity"],
        }
        self.logger.log_info(f"Done condition: {cond}")
        return cond, result

    # ---------- main run ----------

    def run(self, max_processes=None):
        """
        Smart multiprocessing driver:
          - processes = min(#conditions, CPU, max_processes if provided)
          - spawn context for cross-platform stability
          - imap_unordered for streaming progress
        """
        cpu = os.cpu_count() or 1
        n_workers = min(len(self.condition_names), cpu if max_processes is None else min(cpu, max_processes))
        self.logger.log_info(f"Using {n_workers} processes for across-subjects.")

        results = OrderedDict()
        ctx = get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for cond, res in pool.imap_unordered(self._process_one_condition, self.condition_names):
                results[cond] = res
                self.logger.log_info(f"Collected result: {cond}")

        # Save a single JSON containing all conditions
        out_path = os.path.abspath(os.path.join(self.output_dir, self.save_name))
        self.dump_to_json_path(results, out_path)


if __name__ == "__main__":
    # ----------------- EDIT YOUR PARAMETERS HERE (no config file) -----------------

    # Subjects & I/O
    subjects = [f"sub_{i:02d}" for i in range(1, 29)]
    input_dir = "../../../storage/microstate_output/across_runs"         # where each subject's per-condition JSON is
    output_dir = "../../../storage/microstate_output/across_subjects"    # where to save the final JSON
    data_suffix = "_across_runs.json"                       # input filename suffix per subject
    save_name = "across_subjects.json"                      # single output file for all conditions

    # Conditions
    condition_names = ["idea_generation", "idea_evolution", "idea_rating", "rest"]

    # Microstate params
    n_k = 6
    n_ch = 63

    # Logging
    log_dir = "../../../storage/log/across_subjects"
    log_prefix = "across_subjects"
    log_suffix = ""

    # Multiprocessing cap (None = up to CPU count)
    max_processes = None

    # Use gpu (cupy)
    use_gpu = True

    # ----------------- RUN -----------------
    job = PipelineAcrossSubjects(
        input_dir=input_dir,
        output_dir=output_dir,
        subjects=subjects,
        data_suffix=data_suffix,
        save_name=save_name,
        condition_names=condition_names,
        n_k=n_k,
        n_ch=n_ch,
        log_dir=log_dir,
        log_prefix=log_prefix,
        log_suffix=log_suffix,
        use_gpu=use_gpu
    )
    job.logger.log_info("Start across_subjects pipeline")
    job.run(max_processes=max_processes)
    job.logger.log_info("Finish across_subjects pipeline")
