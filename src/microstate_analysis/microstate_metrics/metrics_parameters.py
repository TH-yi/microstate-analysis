"""
Author: Tianhao Ning
Description: This script processes EEG subjects' data to calculate microstate parameters
(duration, coverage, etc.) for each task. The analysis uses multiprocessing for parallel
processing and logs detailed information for each step.

Upgraded: uses the new selective wrapper `batch_microstate_parameters_selective`
to compute only the requested metrics per epoch.
"""

import os
import json
import math
from collections import OrderedDict
from multiprocessing import get_context

from microstate_analysis.microstate_pipeline.pipeline_base import PipelineBase
from microstate_analysis.logger.dualhandler import DualHandler
# NEW: use the new wrapper
from microstate_analysis.microstate_base.parameter import batch_microstate_parameters_selective
from microstate_analysis.microstate_base.data_handler import list_to_matrix


class MetricsParameters(PipelineBase):
    def __init__(self,
                 input_dir,
                 output_dir,
                 subjects,
                 maps_file,
                 distance=10,
                 n_std=3,
                 polarity=False,
                 sfreq=500,
                 epoch=2,
                 task_name=None,
                 # --- new controls for selective metrics ---
                 parameters=None,
                 include_duration_seconds=False,
                 log_base=math.e,
                 states=None,
                 # --- logging ---
                 log_dir=None,
                 prefix=None,
                 suffix=None):
        """
        Args:
            input_dir, output_dir, subjects, maps_file: I/O settings.
            distance, n_std, polarity, sfreq, epoch: backfitting and windowing parameters.
            task_name: list of task names to process.
            parameters: a set/list of metric keys to compute. If None, defaults to:
                        {'coverage', 'duration', 'transition_frequency', 'entropy_rate', 'hurst_mean'}
                        Supported keys include:
                          - 'coverage'
                          - 'duration' (mean in samples)
                          - 'segments'
                          - 'duration_seconds', 'duration_seconds_std', 'duration_seconds_median'
                          - 'transition_frequency'
                          - 'entropy_rate'
                          - 'hurst_mean'
                          - 'hurst_states'
                          - 'transition_matrix'
            include_duration_seconds: if True and 'duration_seconds' keys are requested, return second-based stats.
            log_base: base for entropy rate (e by default).
            states: optional explicit state order (e.g., [0,1,2,3]); defaults to range(n_maps).
            log_dir/prefix/suffix: logger configuration.
        """
        super().__init__()
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.subjects = subjects
        self.task_name = task_name or []
        self.maps_file = maps_file

        # microstate/backfitting parameters
        self.distance = distance
        self.n_std = n_std
        self.polarity = polarity
        self.sfreq = sfreq
        self.epoch = epoch

        # selective metrics controls
        if parameters is None:
            self.parameters = {'coverage', 'duration', 'transition_frequency', 'entropy_rate', 'hurst_mean'}
        else:
            # allow list/tuple -> set
            self.parameters = set(parameters)
        self.include_duration_seconds = bool(include_duration_seconds)
        self.log_base = log_base
        self.states = states  # can be None (then defaults to 0..n_maps-1 inside the wrapper)

        # Save logger config so it can be rebuilt in child processes
        self._logger_cfg = dict(log_dir=log_dir, prefix=prefix or '', suffix=suffix or '')
        self.logger = DualHandler(**self._logger_cfg)

    def process_subject(self, subject):
        """Processes a single subject's EEG data to calculate selective microstate parameters."""
        self.logger.log_info(f"Processing subject: {subject}")
        data_fname = subject + '.json'
        data_path = os.path.abspath(os.path.join(self.input_dir, data_fname))

        res = OrderedDict()
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Load maps (supports both {"maps":[...]} or raw list)
        with open(self.maps_file, 'r', encoding='utf-8') as f:
            maps = json.load(f)
        maps = maps['maps'] if 'maps' in maps else maps

        for task in self.task_name:
            self.logger.log_info(f'Running task: {subject}, {task}')
            task_data = list_to_matrix(data[task])

            # Build positional args list for the new wrapper.
            batch_params = [
                task_data,            # data
                maps,                 # maps
                self.distance,        # distance
                self.n_std,           # n_std
                self.polarity,        # polarity
                self.sfreq,           # sfreq
                self.epoch,           # epoch
                self.parameters,      # parameters set (can be None/Set[str])
                self.include_duration_seconds,  # include_duration_seconds
                self.log_base,        # log_base
                self.states           # states (None or list)
            ]

            task_parameters = batch_microstate_parameters_selective(batch_params)

            res[task] = task_parameters
            self.logger.log_info(f'Finished task: {subject}, {task}')

        # Save results to a JSON file
        self.dump_to_json(res, self.output_dir, f'{subject}_parameters.json')
        self.logger.log_info(f'Finished subject: {subject}')

        return res

    # --------- Make the object picklable ---------
    def __getstate__(self):
        state = self.__dict__.copy()
        state['logger'] = None  # remove non-picklable handler
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.logger = DualHandler(**self._logger_cfg)  # rebuild logger in child

    # --------- Proxy for pool.map ---------
    def _process_subject_proxy(self, subject):
        return self.process_subject(subject)

    # --------- Multiprocessing entry point ---------
    def generate_microstate_parameters(self, max_processes=None):
        """Generate microstate parameters for each subject using multiprocessing."""
        cpu = os.cpu_count() or 1
        n_processes = min(len(self.subjects), cpu if max_processes is None else min(cpu, max_processes))
        self.logger.log_info(f"Using {n_processes} processes")

        ctx = get_context("spawn")
        with ctx.Pool(processes=n_processes) as pool:
            pool.map(self._process_subject_proxy, self.subjects)


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

    eegdata_input_dir = '../../../storage/clean_data'
    eegmaps_input_path = '../../../storage/microstate_output/across_conditions/across_conditions_reordered.json'

    metric_parameters_output_dir = '../../../storage/microstate_output/metric_parameters'

    # optional parameters
    parameters_log_dir = '../../../storage/log/parameters_run'
    parameters_log_prefix = 'parameters_run'
    parameters_log_suffix = ''
    max_processes = 8

    # --- choose metrics here ---
    selected_params = {
        'coverage',
        'duration',                 # mean in samples (legacy-compatible)
        'transition_frequency',
        'entropy_rate',
        'hurst_mean',
        'duration_seconds', 'duration_seconds_std', 'duration_seconds_median',
        'hurst_states',
        'transition_matrix',
        'segments',
    }

    job = MetricsParameters(
        input_dir=eegdata_input_dir,
        output_dir=metric_parameters_output_dir,
        subjects=subjects,
        maps_file=eegmaps_input_path,
        distance=10,
        n_std=3,
        polarity=False,
        sfreq=500,
        epoch=2,
        task_name=task_name,
        # new selective controls:
        parameters=selected_params,
        include_duration_seconds=False,  # set True if you requested duration_seconds*
        log_base=math.e,
        states=None,
        # logging:
        log_dir=parameters_log_dir,
        prefix=parameters_log_prefix,
        suffix=parameters_log_suffix
    )
    job.generate_microstate_parameters(max_processes=max_processes)
