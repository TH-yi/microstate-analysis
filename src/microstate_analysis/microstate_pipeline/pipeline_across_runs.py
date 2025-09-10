"""
Author: Tianhao Ning
Description:
  Compute across-run microstate maps per subject and per condition with
  fully explicit parameters (no config file). Uses smart multiprocessing
  (spawn context, capped workers) and robust logging reconstruction in
  child processes, inspired by the IndividualRun pattern.

Usage:
  - Edit the parameters in the __main__ block and run this file directly.
  - Input JSON per subject must contain each task's data produced by
    the individual run stage, e.g.:
      data[task]['maps_list'][n_k_index]
"""

import os
import json
from collections import OrderedDict
from multiprocessing import get_context

from typing import List, Dict

from microstate_analysis.microstate_pipeline.pipeline_base import PipelineBase
from microstate_analysis.logger.dualhandler import DualHandler
from microstate_analysis.microstate_base.data_handler import load_data
from microstate_analysis.microstate_base.microstate_batch_handler import batch_mean_microstate


class PipelineAcrossRuns(PipelineBase):
    """
    Compute condition-level mean microstate maps across runs (tasks) for each subject.
    All parameters are explicit; no config object is required.
    """

    def __init__(
            self,
            input_dir: str,
            output_dir: str,
            subjects: List[str],
            data_suffix: str,
            save_suffix: str,
            condition_dict: Dict[str, List[str]],
            condition_names: List[str],
            n_k: int = 6,
            n_k_index: int = 4,
            n_ch: int = 63,
            log_dir=None,
            log_prefix: str = "across_runs",
            log_suffix: str = "",
            use_gpu: bool = False
    ):
        super().__init__()
        # Core paths/params
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.subjects = subjects
        self.data_suffix = data_suffix  # e.g. ".json" or "_individual_maps.json"
        self.save_suffix = save_suffix  # e.g. "_across_runs.json"
        self.condition_dict = condition_dict  # {condition_name: [task1, task2, ...]}
        self.condition_names = condition_names  # [cond1, cond2, ...]
        self.n_k = n_k
        self.n_k_index = n_k_index
        self.n_ch = n_ch
        self.use_gpu = use_gpu

        # Logger (store args for child-process reconstruction)
        self._logger_cfg = dict(log_dir=log_dir, prefix=log_prefix or "", suffix=log_suffix or "")
        self.logger = DualHandler(**self._logger_cfg)

    # -------------- helpers --------------

    def dump_to_json(self, data_obj, out_dir: str, filename_wo_ext: str):
        """Save dict/list as pretty JSON under out_dir/filename_wo_ext.json"""
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"{filename_wo_ext}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data_obj, f, ensure_ascii=False, indent=4)
        self.logger.log_info(f"Saved JSON: {path}")

    # -------------- pickling-safe logger --------------

    def __getstate__(self):
        state = self.__dict__.copy()
        state["logger"] = None  # Drop handler before pickling
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Rebuild logger in child
        self.logger = DualHandler(**self._logger_cfg)

    # -------------- per-subject work (called in child) --------------

    def _process_one_subject(self, subject: str) -> str:
        """
        Build condition-level mean microstate maps across runs for one subject.
        Reads subject JSON, collects task maps per condition, aggregates with batch_mean_microstate.
        """
        self.logger.log_info(f"Start processing subject: {subject}")

        # Read subject data
        data_path = os.path.abspath(os.path.join(self.input_dir, subject + self.data_suffix))
        data = load_data(data_path)  # your helper already handles JSON loading

        # Compute condition-wise aggregation
        res = OrderedDict()
        for cond in self.condition_names:
            task_list = self.condition_dict[cond]
            # Collect the run-wise maps for the chosen k (opt index)
            # Each element is a 63xK (or 63xopt_k) topography map array, taken at n_k_index
            maps = [data[task]["maps_list"][self.n_k_index] for task in task_list]

            # batch_mean_microstate expects: [maps, n_k, n_ch, n_runs]
            agg = batch_mean_microstate([maps, self.n_k, self.n_ch, len(task_list), self.use_gpu])
            res[cond] = {
                "maps": agg["maps"].tolist(),
                "label": agg["label"],
                "mean_similarity": agg["mean_similarity"],
                "std_similarity": agg["std_similarity"],
            }

        # Save subject-level across-runs result
        filename_wo_ext = f"{subject}{self.save_suffix}".replace(".json", "")
        self.dump_to_json(res, self.output_dir, filename_wo_ext)

        msg = f"Subject {subject} processing complete."
        self.logger.log_info(msg)
        return msg

    # -------------- multiprocessing entry --------------

    def run(self, max_processes=None):
        """
        Smart multiprocessing runner:
          - Caps workers to min(len(subjects), CPU count, max_processes if provided)
          - Uses spawn context for compatibility across platforms
          - Rebuilds logger in children
        """
        cpu = os.cpu_count() or 1
        n_workers = min(len(self.subjects), cpu if max_processes is None else min(cpu, max_processes))
        self.logger.log_info(f"Using {n_workers} processes for across-runs.")

        ctx = get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            for msg in pool.imap_unordered(self._process_one_subject, self.subjects):
                # Stream results as they finish
                self.logger.log_info(msg)


if __name__ == "__main__":
    # ----------------- EDIT YOUR PARAMETERS HERE (no config file) -----------------

    # Subjects and I/O
    subjects = [f"sub_{i:02d}" for i in range(1, 29)]
    input_dir = "../../../storage/microstate_output/individual_run"  # where each subject JSON (per-run) is
    output_dir = "../../../storage/microstate_output/across_runs"  # where to save across-runs JSON

    # File naming
    data_suffix = "_individual_maps.json"  # subject + data_suffix is the input file
    save_suffix = "_across_runs.json"  # subject + save_suffix is the output file

    # Conditions and tasks (example matches your earlier structure)
    condition_dict = {
        "idea_generation": ["1_idea generation", "2_idea generation", "3_idea generation"],
        "idea_evolution": ["1_idea evolution", "2_idea evolution", "3_idea evolution"],
        "idea_rating": ["1_idea rating", "2_idea rating", "3_idea rating"],
        "rest": ["1_rest", "3_rest"],
    }
    condition_names = ["idea_generation", "idea_evolution", "idea_rating", "rest"]

    # Microstate parameters
    n_k = 6  # number of microstates to compute for aggregation
    n_k_index = 4  # index into maps_list (e.g., opt_k_index from individual stage)
    n_ch = 63  # EEG channels

    # Logging (child processes will rebuild the same logger settings)
    log_dir = "../../../storage/log/across_runs"
    log_prefix = "across_runs"
    log_suffix = ""

    # Multiprocessing cap (None = up to CPU count)
    max_processes = None

    # Use gpu (cupy)
    use_gpu = True

    # ----------------- RUN -----------------
    job = PipelineAcrossRuns(
        input_dir=input_dir,
        output_dir=output_dir,
        subjects=subjects,
        data_suffix=data_suffix,
        save_suffix=save_suffix,
        condition_dict=condition_dict,
        condition_names=condition_names,
        n_k=n_k,
        n_k_index=n_k_index,
        n_ch=n_ch,
        log_dir=log_dir,
        log_prefix=log_prefix,
        log_suffix=log_suffix,
        use_gpu=use_gpu
    )
    job.logger.log_info("Start processing across_runs pipeline")
    job.run(max_processes=max_processes)
    job.logger.log_info("Finish processing across_runs pipeline")
