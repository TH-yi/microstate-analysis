# src/microstate_analysis/cli.py
from __future__ import annotations
import json
import math
import os
import typer
from typing import List, Optional
from microstate_analysis.microstate_pipeline.pipeline_individual_run import PipelineIndividualRun
from microstate_analysis.microstate_pipeline.pipeline_across_runs import PipelineAcrossRuns
from microstate_analysis.microstate_pipeline.pipeline_across_subjects import PipelineAcrossSubjects
from microstate_analysis.microstate_pipeline.pipeline_across_conditions import PipelineAcrossConditions
from microstate_analysis.microstate_pipeline.plot_across_subjects_output_reorder import PlotAcrossSubjectsOutput
from microstate_analysis.microstate_pipeline.plot_across_conditions_output_reorder import PlotAcrossConditionsOutput
from microstate_analysis.microstate_metrics.metrics_parameters import MetricsParameters
from microstate_analysis.microstate_quality.gev_sum_calc import GEVSumCalc
from microstate_analysis.pca.pca_gfp import PCAGFP
from microstate_analysis.pca_microstate_pipeline.pca_pipeline_individual_run import PCAPipelineIndividualRun
from microstate_analysis.pca_microstate_pipeline.pca_pipeline_across_runs import PCAPipelineAcrossRuns
from microstate_analysis.pca_microstate_pipeline.pca_pipeline_across_subjects import PCAPipelineAcrossSubjects
from microstate_analysis.pca_microstate_pipeline.pca_pipeline_across_conditions import PCAPipelineAcrossConditions

app = typer.Typer(help="Microstate Analysis CLI")


class GlobalState:
    json: bool = True
    verbose: int = 0


state = GlobalState()


def _fix_powershell_json(json_str: str) -> str:
    """
    Fix JSON string where PowerShell removed double quotes.
    Restores double quotes around keys and string values in arrays.
    """
    import re
    
    # Step 1: Fix keys - add quotes around unquoted keys before colons
    # Pattern: {key: or ,key: where key is word characters/underscores
    fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', json_str)
    
    # Step 2: Fix array string values - process arrays character by character
    # This handles nested structures correctly
    result = []
    i = 0
    while i < len(fixed):
        if fixed[i] == '[':
            # Start of array - find matching closing bracket
            result.append('[')
            i += 1
            array_start = i
            depth = 1
            while i < len(fixed) and depth > 0:
                if fixed[i] == '[':
                    depth += 1
                elif fixed[i] == ']':
                    depth -= 1
                i += 1
            # Extract array content (without the closing bracket)
            array_content = fixed[array_start:i-1]
            # Split array elements by comma at top level only
            elements = []
            current_elem = ""
            elem_depth = 0
            for char in array_content:
                if char == '[':
                    elem_depth += 1
                    current_elem += char
                elif char == ']':
                    elem_depth -= 1
                    current_elem += char
                elif char == '{':
                    elem_depth += 1
                    current_elem += char
                elif char == '}':
                    elem_depth -= 1
                    current_elem += char
                elif char == ',' and elem_depth == 0:
                    # Top-level comma
                    elem = current_elem.strip()
                    if elem:
                        # Quote if it's a string (not number, not already quoted, not nested)
                        if not elem.startswith('"') and not elem.startswith('[') and not elem.startswith('{') and not elem.replace('.', '').replace('-', '').isdigit() and elem.lower() not in ['true', 'false', 'null']:
                            elements.append(f'"{elem}"')
                        else:
                            elements.append(elem)
                    current_elem = ""
                else:
                    current_elem += char
            # Handle last element
            if current_elem.strip():
                elem = current_elem.strip()
                if not elem.startswith('"') and not elem.startswith('[') and not elem.startswith('{') and not elem.replace('.', '').replace('-', '').isdigit() and elem.lower() not in ['true', 'false', 'null']:
                    elements.append(f'"{elem}"')
                else:
                    elements.append(elem)
            result.append(','.join(elements))
            result.append(']')
        else:
            result.append(fixed[i])
            i += 1
    
    fixed = ''.join(result)
    
    return fixed


def version_callback(value: bool):
    if value:
        print("microstate_analysis 0.2.1")
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
        condition_dict_json: str = typer.Option(..., help='JSON string mapping condition -> list of tasks. Can also be a path to a JSON file. In PowerShell, use a JSON file for best results.'),
        max_processes: Optional[int] = typer.Option(None, help="Max worker processes (default=min(CPU, #subjects))."),
        use_gpu: bool = typer.Option(False, help="Enable GPU acceleration if available."),
):
    try:
        # Try to parse JSON, with better error handling
        # Also check if it's a file path
        if os.path.exists(condition_dict_json):
            try:
                with open(condition_dict_json, 'r', encoding='utf-8') as f:
                    condition_dict = json.load(f)
            except Exception as e:
                typer.echo(f"[ERROR] Failed to read JSON file {condition_dict_json}: {e}", err=True)
                raise
        else:
            try:
                condition_dict = json.loads(condition_dict_json)
            except json.JSONDecodeError as e:
                # Try to fix PowerShell-removed quotes
                try:
                    fixed_json = _fix_powershell_json(condition_dict_json)
                    condition_dict = json.loads(fixed_json)
                except Exception:
                    typer.echo(f"[ERROR] Invalid JSON format in --condition-dict-json: {e}", err=True)
                    typer.echo(f"[ERROR] Received string: {repr(condition_dict_json)}", err=True)
                    typer.echo("[HINT] In PowerShell, try using a JSON file:", err=True)
                    typer.echo("[HINT]   1. Create condition_dict.json with your JSON", err=True)
                    typer.echo("[HINT]   2. Use: --condition-dict-json condition_dict.json", err=True)
                    typer.echo("[HINT] Or wrap variable in quotes: --condition-dict-json \"$conditionJson\"", err=True)
                    raise
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
        conditions: List[str] = typer.Option(
            ["idea_generation", "idea_evolution", "idea_rating", "rest"],
            "--conditions",
        ),
        first_row_order: List[int] = typer.Option(
            [3, 5, 4, 1, 0, 2],
            "--first-row-order",
        ),
        log_dir: Optional[str] = typer.Option(None),
        log_prefix: str = typer.Option("plot_across_subjects"),
        log_suffix: str = typer.Option(""),
        montage_path: Optional[str] = typer.Option(
            None, help="Path to a custom .locs montage. If omitted, built-in cap63.locs is used."
        ),
        sfreq: int = typer.Option(500, help="Sampling frequency for MNE Info (default 500)."),
        channel_types: str = typer.Option("eeg", help="Channel types for MNE Info (default 'eeg')."),
        on_missing: str = typer.Option("raise", help="Montage behavior: 'raise'|'warn'|'ignore'."),
        channel_names: Optional[List[str]] = typer.Option(None, help="Optional explicit channel names."),

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
            custom_montage_path = montage_path,
            sampling_frequency = sfreq,
            channel_types = channel_types,
            missing_channel_behavior = on_missing,
            custom_channel_names = channel_names,
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
        montage_path: Optional[str] = typer.Option(
           None, help="Path to a custom .locs montage. If omitted, built-in cap63.locs is used."
        ),
        sfreq: int = typer.Option(500, help="Sampling frequency for MNE Info (default 500)."),
        channel_types: str = typer.Option("eeg", help="Channel types for MNE Info (default 'eeg')."),
        on_missing: str = typer.Option("raise", help="Montage behavior: 'raise'|'warn'|'ignore'."),
        channel_names: Optional[List[str]] = typer.Option(None, help="Optional explicit channel names."),

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
            custom_montage_path = montage_path,
            sampling_frequency = sfreq,
            channel_types = channel_types,
            missing_channel_behavior = on_missing,
            custom_channel_names = channel_names,
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

@metrics_app.command("gev-sum")
def cli_metrics_gev_sum(
    csv_dir: str = typer.Option(...),
    maps_file: str = typer.Option(...),
    output_dir: str = typer.Option(...),
    mode: str = typer.Option("across_subjects"),
    subjects: List[str] = typer.Option(...),
    task_names: List[str] = typer.Option(...),
    condition_dict_json: str = typer.Option(...),
    log_dir: Optional[str] = typer.Option(None),
    log_prefix: str = typer.Option("gev_sum"),
    log_suffix: str = typer.Option(""),
    max_processors: int = typer.Option(
        0,
        help="Max processes to use. <=0 uses all logical CPU cores.",
    ),
):
    """
    Compute weighted GEV stats per condition using CSV data and microstate maps.
    """
    try:
        try:
            condition_dict = json.loads(condition_dict_json)
        except json.JSONDecodeError:
            # Try to fix PowerShell-removed quotes
            fixed_json = _fix_powershell_json(condition_dict_json)
            condition_dict = json.loads(fixed_json)

        job = GEVSumCalc(
            log_dir=log_dir,
            log_prefix=log_prefix + (f"_{log_suffix}" if log_suffix else ""),
            csv_dir=csv_dir,
            maps_path=maps_file,
            subjects=subjects,
            task_names=task_names,
            condition_dict=condition_dict,
            mode=mode,
        )
        job.logger.log_info("[CLI] metrics gev-sum started")
        job.run(output_dir=output_dir, max_processors=max_processors)  # <<< NEW
        job.logger.log_info("[CLI] metrics gev-sum finished")

    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)


@metrics_app.command("cluster-quality")
def cli_cluster_quality(
    mode: str = typer.Option("across_subjects", help="across_subjects | across_conditions"),
    csv_dir: str = typer.Option(..., help="Root dir: {csv_dir}/{subject}/*.csv"),
    output_dir: str = typer.Option(..., help="Directory to save the summary JSON"),
    result_name: str = typer.Option(..., help="Output JSON name without suffix"),
    # Provide either `conditions` OR `condition_dict_json`
    conditions: List[str] = typer.Option(None, help="e.g., idea_generation idea_evolution idea_rating rest"),
    condition_dict_json: Optional[str] = typer.Option(
        None,
        help='JSON: {"idea_generation": ["1_idea generation","2_idea generation",...], "rest":["1_rest","3_rest"]}',
    ),
    label_column: str = typer.Option("microstate_label", help="Name of the label column in CSVs"),
    log_dir: Optional[str] = typer.Option(None),
    log_prefix: str = typer.Option("cluster_quality"),
    max_processors: int = typer.Option(
        0,
        help="Max processes to use. <=0 uses all logical CPU cores.",
    ),
):
    """
    Compute Silhouette & WCSS per condition, aggregating mean/std across files.
    Provide either `conditions` OR a richer `condition_dict_json`.
    """
    try:
        if condition_dict_json:
            try:
                cond_dict = json.loads(condition_dict_json)
            except json.JSONDecodeError:
                # Try to fix PowerShell-removed quotes
                fixed_json = _fix_powershell_json(condition_dict_json)
                cond_dict = json.loads(fixed_json)
        else:
            cond_dict = None

        from microstate_analysis.microstate_quality.cluster_quality_analysis import (
            ClusterQualityAnalysis,
        )

        job = ClusterQualityAnalysis(log_dir=log_dir, log_prefix=log_prefix)
        job.logger.log_info("[CLI] metrics cluster-quality started")

        results = job.run(
            mode=mode,
            csv_dir=csv_dir,
            output_dir=output_dir,
            result_name=result_name,
            conditions=conditions,
            condition_dict=cond_dict,
            label_column=label_column,
            max_processors=max_processors,  # <<< NEW
        )

        job.logger.log_info("[CLI] metrics cluster-quality finished")
        typer.echo(json.dumps(results, indent=2))

    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)


# ===== PCA =====
pca_app = typer.Typer(help="PCA dimensionality reduction pipelines.")
app.add_typer(pca_app, name="pca")


@pca_app.command("gfp")
def cli_pca_gfp(
    input_dir: str = typer.Option(..., help="Directory containing GFP CSV files (structure: {input_dir}/{subject}/*.csv)."),
    output_dir: str = typer.Option(..., help="Base output directory for PCA results."),
    subjects: List[str] = typer.Option(..., help="List of subject IDs, e.g., P01 P02 ..."),
    percentages: List[float] = typer.Option(
        [0.95, 0.98, 0.99],
        help="Variance retention ratios for PCA (e.g., 0.95 0.98 0.99)."
    ),
    max_processes: Optional[int] = typer.Option(None, help="Max worker processes (default=CPU count)."),
    log_dir: Optional[str] = typer.Option(None, help="Directory to write logs. If omitted, logs go to stderr only."),
    log_prefix: str = typer.Option("pca_gfp", help="Log filename prefix."),
    log_suffix: str = typer.Option("", help="Log filename suffix."),
):
    """
    Perform PCA dimensionality reduction on GFP CSV files.
    Outputs eigenvalues, eigenvectors, and final transformed matrices for each percentage.
    """
    try:
        job = PCAGFP(
            input_dir=input_dir,
            output_dir=output_dir,
            subjects=subjects,
            percentages=percentages,
            log_dir=log_dir,
            log_prefix=log_prefix,
            log_suffix=log_suffix,
        )
        job.logger.log_info("[CLI] pca gfp started")
        job.run(max_processes=max_processes)
        job.logger.log_info("[CLI] pca gfp finished")
    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)


# ===== PCA Microstate Pipeline =====
pca_pipeline_app = typer.Typer(help="PCA microstate analysis pipelines.")
pca_app.add_typer(pca_pipeline_app, name="microstate-pipeline")


@pca_pipeline_app.command("individual-run")
def cli_pca_individual_run(
    input_dir: str = typer.Option(..., help="Directory containing raw per-subject JSON files (e.g., sub_01.json)."),
    output_dir: str = typer.Option(..., help="Directory to save {subject}_pca_individual_maps.json"),
    subjects: List[str] = typer.Option(..., help="List of subject IDs, e.g., sub_01 sub_02 ..."),
    task_name: List[str] = typer.Option(..., help="Task names, e.g., '1_idea generation' '2_idea generation' ..."),
    percentage: float = typer.Option(0.95, help="PCA variance retention ratio (e.g., 0.95, 0.98, 0.99)."),
    peaks_only: bool = typer.Option(False, help="Use peaks-only logic for microstate clustering."),
    min_maps: int = typer.Option(2, help="Minimum number of maps."),
    max_maps: int = typer.Option(10, help="Maximum number of maps."),
    opt_k: Optional[int] = typer.Option(None, help="Optional fixed K value."),
    cluster_method: str = typer.Option("kmeans_modified", help="Clustering method."),
    n_std: int = typer.Option(3, help="Threshold std for microstate clustering."),
    n_runs: int = typer.Option(100, help="Clustering restarts."),
    gfp_distance: int = typer.Option(10, help="Minimum distance between GFP peaks."),
    gfp_n_std: int = typer.Option(3, help="Number of standard deviations for GFP peak thresholding."),
    pca_output_dir: Optional[str] = typer.Option(None, help="Directory to save PCA intermediate results (eigenvalues, eigenvectors, final_matrix). If omitted, uses output_dir/../pca_output."),
    save_pca_intermediate: bool = typer.Option(True, help="Save PCA intermediate results."),
    save_task_map_counts: bool = typer.Option(True, help="Save per-task opt_k list across subjects."),
    max_processes: Optional[int] = typer.Option(None, help="Cap worker processes."),
    log_dir: Optional[str] = typer.Option(None, help="Directory to write logs."),
    log_prefix: str = typer.Option("pca_individual_run", help="Log filename prefix."),
    log_suffix: str = typer.Option("", help="Log filename suffix."),
    use_gpu: bool = typer.Option(False, help="Enable GPU acceleration if available."),
):
    """
    Complete PCA microstate pipeline: Raw JSON → GFP peaks → PCA → Microstate clustering.
    All-in-one pipeline that processes raw subject JSON files end-to-end.
    """
    try:
        job = PCAPipelineIndividualRun(
            input_dir=input_dir,
            output_dir=output_dir,
            subjects=subjects,
            task_names=task_name,
            percentage=percentage,
            peaks_only=peaks_only,
            min_maps=min_maps,
            max_maps=max_maps,
            opt_k=opt_k,
            cluster_method=cluster_method,
            n_std=n_std,
            n_runs=n_runs,
            gfp_distance=gfp_distance,
            gfp_n_std=gfp_n_std,
            pca_output_dir=pca_output_dir,
            save_pca_intermediate=save_pca_intermediate,
            log_dir=log_dir,
            log_prefix=log_prefix,
            log_suffix=log_suffix,
            use_gpu=use_gpu,
        )
        job.logger.log_info("[CLI] pca microstate-pipeline individual-run started")
        job.run(
            max_processes=max_processes,
            save_task_map_counts=save_task_map_counts,
        )
        job.logger.log_info("[CLI] pca microstate-pipeline individual-run finished")
    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)


@pca_pipeline_app.command("across-runs")
def cli_pca_across_runs(
    input_dir: str = typer.Option(..., help="Directory of per-subject individual maps JSONs."),
    output_dir: str = typer.Option(..., help="Output dir for across-runs JSONs."),
    data_suffix: str = typer.Option("_pca_individual_maps.json", help="Input filename suffix (subject + suffix)."),
    save_suffix: str = typer.Option("_pca_across_runs.json", help="Output filename suffix (subject + suffix)."),
    subjects: List[str] = typer.Option(..., help="List of subject IDs, e.g. P01 P02 ..."),
    percentage: float = typer.Option(..., help="PCA percentage (e.g., 0.95, 0.98, 0.99)."),
    n_k: int = typer.Option(6, help="Number of microstates."),
    n_k_index: int = typer.Option(1, help="Index into maps_list for each task/run."),
    n_ch: int = typer.Option(63, help="Number of EEG channels."),
    log_dir: Optional[str] = typer.Option(None,
                                          help="Directory to write logs. If omitted, logs go to stderr only."),
    log_prefix: str = typer.Option("pca_across_runs", help="Log filename prefix."),
    log_suffix: str = typer.Option("", help="Log filename suffix."),
    condition_dict_json: str = typer.Option(..., help='JSON string mapping condition -> list of tasks. Can also be a path to a JSON file. In PowerShell, use a JSON file for best results.'),
    max_processes: Optional[int] = typer.Option(None, help="Max worker processes (default=min(CPU, #subjects))."),
    use_gpu: bool = typer.Option(False, help="Enable GPU acceleration if available."),
):
    """
    Aggregate each subject's runs/tasks into conditions (per subject).
    """
    try:
        # Try to parse JSON, with better error handling
        # Also check if it's a file path
        if os.path.exists(condition_dict_json):
            try:
                with open(condition_dict_json, 'r', encoding='utf-8') as f:
                    condition_dict = json.load(f)
            except Exception as e:
                typer.echo(f"[ERROR] Failed to read JSON file {condition_dict_json}: {e}", err=True)
                raise
        else:
            try:
                condition_dict = json.loads(condition_dict_json)
            except json.JSONDecodeError as e:
                # Try to fix PowerShell-removed quotes
                try:
                    fixed_json = _fix_powershell_json(condition_dict_json)
                    condition_dict = json.loads(fixed_json)
                except Exception:
                    typer.echo(f"[ERROR] Invalid JSON format in --condition-dict-json: {e}", err=True)
                    typer.echo(f"[ERROR] Received string: {repr(condition_dict_json)}", err=True)
                    typer.echo("[HINT] In PowerShell, try using a JSON file:", err=True)
                    typer.echo("[HINT]   1. Create condition_dict.json with your JSON", err=True)
                    typer.echo("[HINT]   2. Use: --condition-dict-json condition_dict.json", err=True)
                    typer.echo("[HINT] Or wrap variable in quotes: --condition-dict-json \"$conditionJson\"", err=True)
                    raise
        job = PCAPipelineAcrossRuns(
            input_dir=input_dir,
            output_dir=output_dir,
            subjects=subjects,
            condition_dict=condition_dict,
            percentage=percentage,
            n_k=n_k,
            n_k_index=n_k_index,
            n_ch=n_ch,
            data_suffix=data_suffix,
            save_suffix=save_suffix,
            log_dir=log_dir,
            log_prefix=log_prefix,
            log_suffix=log_suffix,
            use_gpu=use_gpu,
        )
        job.logger.log_info("[CLI] pca microstate-pipeline across-runs started")
        job.run(max_processes=max_processes)
        job.logger.log_info("[CLI] pca microstate-pipeline across-runs finished")
    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)


@pca_pipeline_app.command("across-subjects")
def cli_pca_across_subjects(
    input_dir: str = typer.Option(..., help="Dir of per-subject across-runs JSONs."),
    output_dir: str = typer.Option(..., help="Dir to save a single across-subjects JSON."),
    data_suffix: str = typer.Option("_pca_across_runs.json", help="Input filename suffix."),
    save_name: str = typer.Option("pca_across_subjects.json", help="Output filename."),
    subjects: List[str] = typer.Option(..., help="Subjects list."),
    condition_names: List[str] = typer.Option(["idea_generation", "idea_evolution", "idea_rating", "rest"],
                                              help="Conditions."),
    percentage: float = typer.Option(..., help="PCA percentage (e.g., 0.95, 0.98, 0.99)."),
    n_k: int = typer.Option(6),
    n_ch: int = typer.Option(63),
    log_dir: Optional[str] = typer.Option(None),
    log_prefix: str = typer.Option("pca_across_subjects"),
    log_suffix: str = typer.Option(""),
    max_processes: Optional[int] = typer.Option(None),
    use_gpu: bool = typer.Option(False, help="Enable GPU acceleration if available."),
):
    """
    Aggregate per-subject across-runs maps into condition-level maps.
    """
    try:
        job = PCAPipelineAcrossSubjects(
            input_dir=input_dir,
            output_dir=output_dir,
            subjects=subjects,
            condition_names=condition_names,
            percentage=percentage,
            n_k=n_k,
            n_ch=n_ch,
            data_suffix=data_suffix,
            save_name=save_name,
            log_dir=log_dir,
            log_prefix=log_prefix,
            log_suffix=log_suffix,
            use_gpu=use_gpu,
        )
        job.logger.log_info("[CLI] pca microstate-pipeline across-subjects started")
        job.run(max_processes=max_processes)
        job.logger.log_info("[CLI] pca microstate-pipeline across-subjects finished")
    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)


@pca_pipeline_app.command("across-conditions")
def cli_pca_across_conditions(
    input_dir: str = typer.Option(..., help="Dir of across-subjects JSON."),
    input_name: str = typer.Option("pca_across_subjects.json"),
    output_dir: str = typer.Option(..., help="Dir to save across-conditions JSON."),
    output_name: str = typer.Option("pca_across_conditions.json"),
    condition_names: List[str] = typer.Option(["idea_generation", "idea_evolution", "idea_rating", "rest"]),
    percentage: float = typer.Option(..., help="PCA percentage (e.g., 0.95, 0.98, 0.99)."),
    n_k: int = typer.Option(6),
    n_ch: int = typer.Option(63),
    log_dir: Optional[str] = typer.Option(None),
    log_prefix: str = typer.Option("pca_across_conditions"),
    log_suffix: str = typer.Option(""),
    use_gpu: bool = typer.Option(False, help="Enable GPU acceleration if available."),
):
    """
    Aggregate conditions into global microstate maps.
    """
    try:
        job = PCAPipelineAcrossConditions(
            input_dir=input_dir,
            input_name=input_name,
            output_dir=output_dir,
            output_name=output_name,
            condition_names=condition_names,
            percentage=percentage,
            n_k=n_k,
            n_ch=n_ch,
            log_dir=log_dir,
            log_prefix=log_prefix,
            log_suffix=log_suffix,
            use_gpu=use_gpu,
        )
        job.logger.log_info("[CLI] pca microstate-pipeline across-conditions started")
        job.run()
        job.logger.log_info("[CLI] pca microstate-pipeline across-conditions finished")
    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)
