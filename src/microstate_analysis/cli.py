# src/microstate_analysis/cli.py
from __future__ import annotations
import json
import math
import typer
from typing import List, Optional
from microstate_analysis.microstate_pipeline.pipeline_individual_run import PipelineIndividualRun
from microstate_analysis.microstate_pipeline.pipeline_across_runs import PipelineAcrossRuns
from microstate_analysis.microstate_pipeline.pipeline_across_subjects import PipelineAcrossSubjects
from microstate_analysis.microstate_pipeline.pipeline_across_conditions import PipelineAcrossConditions
from microstate_analysis.microstate_pipeline.plot_across_subjects_output_reorder import PlotAcrossSubjectsOutput
from microstate_analysis.microstate_pipeline.plot_across_conditions_output_reorder import PlotAcrossConditionsOutput
from microstate_analysis.microstate_metrics.metrics_parameters import MetricsParameters

app = typer.Typer(help="Microstate Analysis CLI")


class GlobalState:
    json: bool = True
    verbose: int = 0


state = GlobalState()


def version_callback(value: bool):
    if value:
        print("microstate_analysis 0.1.0")
        raise typer.Exit()


# ===== Pipelines =====
pipeline_app = typer.Typer(help="Run core microstate pipelines.")
app.add_typer(pipeline_app, name="microstate-pipeline")


# ===== Individual Run =====
@pipeline_app.command("individual-run")
def cli_individual_run(
        input_dir: str = typer.Option(..., help="Directory containing raw per-subject JSON files."),
        output_dir: str = typer.Option(..., help="Directory to save {subject}_individual_maps.json"),
        subjects: List[str] = typer.Option(..., help="E.g., sub_01 sub_02 ..."),
        task_name: List[str] = typer.Option(
            ..., help="Task names, e.g. '1_idea generation' '2_idea generation' ..."),
        peaks_only: bool = typer.Option(True),
        min_maps: int = typer.Option(2),
        max_maps: int = typer.Option(10),
        cluster_method: str = typer.Option("kmeans_modified"),
        n_std: int = typer.Option(3),
        n_runs: int = typer.Option(100),
        save_task_map_counts: bool = typer.Option(True, help="Also dump per-task opt_k list across subjects."),
        task_map_counts_output_dir: Optional[str] = typer.Option(None),
        task_map_counts_output_filename: Optional[str] = typer.Option("individual_map_counts"),
        max_processes: Optional[int] = typer.Option(None, help="Cap worker processes."),
        log_dir: Optional[str] = typer.Option(None),
        log_prefix: str = typer.Option("individual_run"),
        log_suffix: str = typer.Option(""),
        use_gpu: bool = typer.Option(False, help="Enable GPU acceleration if available."),
):
    try:
        job = PipelineIndividualRun(
            input_dir=input_dir,
            output_dir=output_dir,
            subjects=subjects,
            peaks_only=peaks_only,
            min_maps=min_maps,
            max_maps=max_maps,
            task_name=task_name,
            log_dir=log_dir,
            prefix=log_prefix,
            suffix=log_suffix,
            cluster_method=cluster_method,
            n_std=n_std,
            n_runs=n_runs,
            use_gpu=use_gpu
        )
        job.logger.log_info("[CLI] individual-run started")
        job.generate_individual_eeg_maps(
            save_task_map_counts=save_task_map_counts,
            task_map_counts_output_dir=task_map_counts_output_dir,
            task_map_counts_output_filename=task_map_counts_output_filename,
            max_processes=max_processes,
        )
        job.logger.log_info("[CLI] individual-run finished")
    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)


@pipeline_app.command("across-runs")
def cli_across_runs(
        input_dir: str = typer.Option(..., help="Directory of per-subject per-run JSONs."),
        output_dir: str = typer.Option(..., help="Output dir for across-runs JSONs."),
        data_suffix: str = typer.Option("_individual_maps.json", help="Input filename suffix (subject + suffix)."),
        save_suffix: str = typer.Option("_across_runs.json", help="Output filename suffix (subject + suffix)."),
        subjects: List[str] = typer.Option(..., help="List of subject IDs, e.g. P01 P02 ..."),
        n_k: int = typer.Option(6, help="Number of microstates."),
        n_k_index: int = typer.Option(4, help="Index into maps_list for each task/run."),
        n_ch: int = typer.Option(63, help="Number of EEG channels."),
        log_dir: Optional[str] = typer.Option(None,
                                              help="Directory to write logs. If omitted, logs go to stderr only."),
        log_prefix: str = typer.Option("across_runs", help="Log filename prefix."),
        log_suffix: str = typer.Option("", help="Log filename suffix."),
        # condition dict as JSON string for CLI
        condition_dict_json: str = typer.Option(..., help='JSON string mapping condition -> list of tasks.'),
        max_processes: Optional[int] = typer.Option(None, help="Max worker processes (default=min(CPU, #subjects))."),
        use_gpu: bool = typer.Option(False, help="Enable GPU acceleration if available."),
):
    try:
        condition_dict = json.loads(condition_dict_json)
        job = PipelineAcrossRuns(
            input_dir=input_dir,
            output_dir=output_dir,
            subjects=subjects,
            data_suffix=data_suffix,
            save_suffix=save_suffix,
            condition_dict=condition_dict,
            condition_names=list(condition_dict.keys()),
            n_k=n_k,
            n_k_index=n_k_index,
            n_ch=n_ch,
            log_dir=log_dir,
            log_prefix=log_prefix,
            log_suffix=log_suffix,
            use_gpu=use_gpu
        )
        job.logger.log_info("[CLI] across-runs started")
        job.run(max_processes=max_processes)
        job.logger.log_info("[CLI] across-runs finished")
    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)


@pipeline_app.command("across-subjects")
def cli_across_subjects(
        input_dir: str = typer.Option(..., help="Dir of per-subject across-runs JSONs."),
        output_dir: str = typer.Option(..., help="Dir to save a single across-subjects JSON."),
        data_suffix: str = typer.Option("_across_runs.json", help="Input filename suffix."),
        save_name: str = typer.Option("across_subjects.json", help="Output filename."),
        subjects: List[str] = typer.Option(..., help="Subjects list."),
        condition_names: List[str] = typer.Option(["idea_generation", "idea_evolution", "idea_rating", "rest"],
                                                  help="Conditions."),
        n_k: int = typer.Option(6),
        n_ch: int = typer.Option(63),
        log_dir: Optional[str] = typer.Option(None),
        log_prefix: str = typer.Option("across_subjects"),
        log_suffix: str = typer.Option(""),
        max_processes: Optional[int] = typer.Option(None),
        use_gpu: bool = typer.Option(False, help="Enable GPU acceleration if available."),
):
    try:
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
        job.logger.log_info("[CLI] across-subjects started")
        job.run(max_processes=max_processes)
        job.logger.log_info("[CLI] across-subjects finished")
    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)


@pipeline_app.command("across-conditions")
def cli_across_conditions(
        input_dir: str = typer.Option(..., help="Dir of across-subjects JSON."),
        input_name: str = typer.Option("across_subjects.json"),
        output_dir: str = typer.Option(..., help="Dir to save across-conditions JSON."),
        output_name: str = typer.Option("across_conditions.json"),
        condition_names: List[str] = typer.Option(["idea_generation", "idea_evolution", "idea_rating", "rest"]),
        n_k: int = typer.Option(6),
        n_ch: int = typer.Option(63),
        log_dir: Optional[str] = typer.Option(None),
        log_prefix: str = typer.Option("across_conditions"),
        log_suffix: str = typer.Option(""),
        use_gpu: bool = typer.Option(False, help="Enable GPU acceleration if available."),
):
    try:
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
            use_gpu=use_gpu
        )
        job.logger.log_info("[CLI] across-conditions started")
        job.run()
        job.logger.log_info("[CLI] across-conditions finished")
    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)


# ===== Plotting =====
plot_app = typer.Typer(help="Plot & reorder utilities.")
app.add_typer(plot_app, name="plot")


@plot_app.command("across-subjects")
def cli_plot_across_subjects(
        input_json_path: str = typer.Option(...),
        output_img_dir: str = typer.Option(...),
        reordered_json_path: str = typer.Option(...),
        conditions: List[str] = typer.Option(["idea_generation", "idea_evolution", "idea_rating", "rest"]),
        first_row_order: List[int] = typer.Option([3, 5, 4, 1, 0, 2]),
        log_dir: Optional[str] = typer.Option(None),
        log_prefix: str = typer.Option("plot_across_subjects"),
        log_suffix: str = typer.Option(""),
):
    try:
        job = PlotAcrossSubjectsOutput(
            input_json_path=input_json_path,
            output_img_dir=output_img_dir,
            reordered_json_path=reordered_json_path,
            conditions=conditions,
            first_row_order=first_row_order,
            log_dir=log_dir,
            log_prefix=log_prefix,
            log_suffix=log_suffix,
        )
        job.logger.log_info("[CLI] plot across-subjects started")
        job.plot_and_reorder()
        job.logger.log_info("[CLI] plot across-subjects finished")
    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)


@plot_app.command("across-conditions")
def cli_plot_across_conditions(
        input_json_path: str = typer.Option(...),
        output_img_dir: str = typer.Option(...),
        reordered_json_path: str = typer.Option(...),
        conditions: List[str] = typer.Option(["idea_generation", "idea_evolution", "idea_rating", "rest"]),
        first_row_order: List[int] = typer.Option([3, 5, 0, 4, 2, 1]),
        log_dir: Optional[str] = typer.Option(None),
        log_prefix: str = typer.Option("plot_across_conditions"),
        log_suffix: str = typer.Option(""),
):
    try:
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
        job.logger.log_info("[CLI] plot across-conditions started")
        job.plot_and_reorder()
        job.logger.log_info("[CLI] plot across-conditions finished")

    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)


# ===== Metrics =====
metrics_app = typer.Typer(help="Run microstate metrics pipelines.")
app.add_typer(metrics_app, name="metrics")


@metrics_app.command("parameters-run")
def cli_parameters_run(
        input_dir: str = typer.Option(..., help="Directory containing raw per-subject JSON files."),
        output_dir: str = typer.Option(..., help="Directory to save {subject}_parameters.json"),
        subjects: List[str] = typer.Option(..., help="E.g., sub_01 sub_02 ..."),
        task_name: List[str] = typer.Option(..., help="Task names, e.g. '1_idea generation' '2_idea generation' ..."),
        maps_file: str = typer.Option(..., help="Path to JSON of across-condition maps."),
        distance: int = typer.Option(10),
        n_std: int = typer.Option(3),
        polarity: bool = typer.Option(False),
        sfreq: int = typer.Option(500),
        epoch: float = typer.Option(2.0, help="Epoch window in seconds."),
        parameters: List[str] = typer.Option(
            [],
            help="Metrics to compute. If empty, defaults to coverage,duration,transition_frequency,entropy_rate,"
                 "hurst_mean."),
        include_duration_seconds: bool = typer.Option(False,
                                                      help="If True and duration_seconds* requested, use seconds."),
        log_base: float = typer.Option(math.e, help="Log base for entropy rate."),
        states: Optional[List[int]] = typer.Option(None, help="Explicit state order."),
        max_processes: Optional[int] = typer.Option(None, help="Cap worker processes."),
        log_dir: Optional[str] = typer.Option(None),
        log_prefix: str = typer.Option("parameters_run"),
        log_suffix: str = typer.Option("")
):
    """Compute selected microstate metrics per subject × task."""
    try:
        selected_params = set(parameters) if parameters else None
        job = MetricsParameters(
            input_dir=input_dir,
            output_dir=output_dir,
            subjects=subjects,
            maps_file=maps_file,
            distance=distance,
            n_std=n_std,
            polarity=polarity,
            sfreq=sfreq,
            epoch=epoch,
            task_name=task_name,
            parameters=selected_params,
            include_duration_seconds=include_duration_seconds,
            log_base=log_base,
            states=states,
            log_dir=log_dir,
            prefix=log_prefix,
            suffix=log_suffix,
        )
        job.logger.log_info("[CLI] metrics parameters-run started")
        job.generate_microstate_parameters(max_processes=max_processes)
        job.logger.log_info("[CLI] metrics parameters-run finished")
    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)
