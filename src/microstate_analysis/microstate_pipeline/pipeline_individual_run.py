"""
Author: Tianhao Ning
Description: This script processes individual EEG subjects' data, calculates microstate maps,
and saves the output for each task. The analysis uses multiprocessing for parallel processing
and logs detailed information for each step.
"""

import os
import json
from collections import OrderedDict
from multiprocessing import get_context

from microstate_analysis.microstate_pipeline.pipeline_base import PipelineBase
from microstate_analysis.logger.dualhandler import DualHandler
from microstate_analysis.microstate_base.microstate_batch_handler import batch_microstate
from microstate_analysis.microstate_base.data_handler import list_to_matrix


class PipelineIndividualRun(PipelineBase):
    def __init__(self, input_dir, output_dir, subjects, peaks_only, min_maps, max_maps, task_name,
                 log_dir=None, prefix=None, suffix=None, cluster_method='kmeans_modified',
                 n_std=3, n_runs=100, use_gpu=False):
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.subjects = subjects
        self.task_name = task_name
        # microstate parameters
        self.peaks_only = peaks_only
        self.min_maps = min_maps
        self.max_maps = max_maps
        self.cluster_method = cluster_method
        self.n_std = n_std
        self.n_runs = n_runs
        self.use_gpu = use_gpu
        # Save logger config so it can be rebuilt in child processes
        self._logger_cfg = dict(log_dir=log_dir, prefix=prefix or '', suffix=suffix or '')
        self.logger = DualHandler(**self._logger_cfg)

    def process_subject(self, subject):
        """Processes a single subject's EEG data."""
        self.logger.log_info(f"Processing subject: {subject}")
        subj_maps_counts = {}
        data_fname = subject + '.json'
        data_path = os.path.abspath(os.path.join(self.input_dir, data_fname))

        res = OrderedDict()
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for task in self.task_name:
            task_data = list_to_matrix(data[task])
            peaks_only = self.peaks_only
            min_maps = self.min_maps
            max_maps = self.max_maps
            method = self.cluster_method
            n_std = self.n_std
            n_runs = self.n_runs
            use_gpu = self.use_gpu
            batch_params = [task_data, peaks_only, min_maps, max_maps, None, method, n_std, n_runs, use_gpu]
            task_microstate = batch_microstate(batch_params)

            res[task] = {
                'cv_list': task_microstate.cv_list,
                'gev_list': task_microstate.gev_list,
                'maps_list': task_microstate.maps_list,
                'opt_k': task_microstate.opt_k,
                'opt_k_index': int(task_microstate.opt_k_index),
                'min_maps': min_maps,
                'max_maps': max_maps
            }
            subj_maps_counts[task] = task_microstate.opt_k
            self.logger.log_info(
                f'Finished task: {subject}, {task}, opt_k: {task_microstate.opt_k}, opt_k_index: {task_microstate.opt_k_index}')

        # Save results to a JSON file
        self.dump_to_json(res, self.output_dir, f'{subject}_individual_maps.json')
        self.logger.log_info(f'Finished subject: {subject}')

        return subj_maps_counts

    # --------- Make the object picklable ---------

    def __getstate__(self):
        state = self.__dict__.copy()
        state['logger'] = None  # remove non-picklable handler
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = DualHandler(**self._logger_cfg)  # rebuild logger in child

    # --------- Proxy for pool.map (no external wrapper needed) ---------
    def _process_subject_proxy(self, subject):
        return self.process_subject(subject)

    # --------- Multiprocessing entry point ---------
    def generate_individual_eeg_maps(self, save_task_map_counts,
                                     task_map_counts_output_dir=None,
                                     task_map_counts_output_filename=None,
                                     max_processes=None):
        """Generate EEG maps for each subject using multiprocessing."""
        cpu = os.cpu_count() or 1
        n_processes = min(len(self.subjects), cpu if max_processes is None else min(cpu, max_processes))
        self.logger.log_info(f"Using {n_processes} processes")

        ctx = get_context("spawn")
        with ctx.Pool(processes=n_processes) as pool:
            results = pool.map(self._process_subject_proxy, self.subjects)

        if save_task_map_counts:
            if not task_map_counts_output_dir or not task_map_counts_output_filename:
                self.logger.log_error("Missing parameters for saving task map counts!")
                raise ValueError("task_map_counts_output_dir and task_map_counts_output_filename are required.")
            task_wise_map_counts = [subdict for subdict in results]
            self.dump_to_json(task_wise_map_counts, task_map_counts_output_dir, task_map_counts_output_filename)


if __name__ == '__main__':
    subjects = ['sub_01', 'sub_02', 'sub_03', 'sub_04', 'sub_05', 'sub_06', 'sub_07', 'sub_08', 'sub_09', 'sub_10',
                'sub_11', 'sub_12', 'sub_13', 'sub_14', 'sub_15', 'sub_16', 'sub_17', 'sub_18', 'sub_19', 'sub_20',
                'sub_21', 'sub_22', 'sub_23', 'sub_24', 'sub_25', 'sub_26', 'sub_27', 'sub_28']
    task_name = [
        '1_idea generation', '2_idea generation', '3_idea generation',
        '1_idea evolution', '2_idea evolution', '3_idea evolution',
        '1_idea rating', '2_idea rating', '3_idea rating',
        '1_rest', '3_rest'
    ]

    individual_input_dir = '../../../storage/clean_data'
    individual_output_dir = '../../../storage/microstate_output/individual_run'
    peaks_only = True
    min_maps = 2
    max_maps = 10
    save_task_map_counts = True

    # optional parameters
    individual_log_dir = '../../../storage/log/individual_run'
    individual_log_prefix = 'individual_run'
    individual_log_suffix = ''
    task_map_counts_output_dir = '../../../storage/microstate_output/individual_run'
    task_map_counts_output_filename = 'individual_map_counts'
    max_processes = 9
    cluster_method = 'kmeans_modified'
    n_std = 3
    n_runs = 100
    use_gpu = True

    individual_job = PipelineIndividualRun(log_dir=individual_log_dir, prefix=individual_log_prefix,
                                   suffix=individual_log_suffix, input_dir=individual_input_dir,
                                   output_dir=individual_output_dir, subjects=subjects,
                                   peaks_only=peaks_only, min_maps=min_maps, max_maps=max_maps,
                                   task_name=task_name, cluster_method=cluster_method,
                                   n_std=n_std, n_runs=n_runs, use_gpu=use_gpu)
    individual_job.generate_individual_eeg_maps(save_task_map_counts=save_task_map_counts, task_map_counts_output_dir=task_map_counts_output_dir,
                                                task_map_counts_output_filename=task_map_counts_output_filename, max_processes=max_processes)
