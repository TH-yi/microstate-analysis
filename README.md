# Microstate Analysis

A production-ready **EEG microstate analysis** toolkit packaged as a Python library with a **Typer-based CLI**. It is designed for both direct command-line use and **embedding in GUIs** via subprocess + stdout/stderr. The toolkit implements a modular pipeline:

- **individual-run** → compute per-subject, per-task microstates  
- **across-runs** → aggregate runs/tasks to condition-level per subject  
- **across-subjects** → aggregate subjects to a single condition JSON  
- **across-conditions** → aggregate conditions to a global JSON  
- **plot** subcommands → visualize and optionally **reorder** microstate maps

---

## Features

- Standard Python package (`pyproject.toml`) with `microstate-analysis` console entry.
- **CLI-first** with **JSON Lines** output on `stdout` for machine-friendly integration; logs on `stderr` for humans.
- Cross-platform **multiprocessing** with `spawn` and child-safe logger reconstruction.
- All pipeline parameters are **explicit CLI options** (no mandatory config file).
- Easy to extend: add new subpackages (e.g., `pca_microstate_pipeline`, `anova`) and register new subcommands.
- **Optional GPU acceleration**: all core clustering (`Microstate`, `MeanMicrostate`) support
  `--use-gpu` flag in CLI. When enabled (and CuPy is available), heavy linear algebra
  (distance, correlation, eigen-decomposition) runs on GPU for significant speedups.
  Falls back gracefully to CPU if unavailable.

### Montage (cap63.locs)
- By default, plotting uses the **built-in** `cap63.locs` montage bundled in the package.
- You can override it via:
  - CLI: `--montage-path /path/to/custom.locs`, or
  - Environment variable: `MICROSTATE_LOCS=/path/to/custom.locs`
- Internally, the package resolves the built-in resource using `importlib.resources`, so it also works when installed as a wheel/zip.
---

## Installation

```bash
# From the repository root (editable install for development)
pip install -e .
```

Python >= 3.9 is recommended.

---

## Command-Line Interface

Top-level help:

```bash
microstate-analysis --help
microstate-analysis microstate-pipeline --help
microstate-analysis plot --help
```

### Conventions (stdout/stderr contract)

- **stdout**: the CLI emits one JSON object **per line** for status/events. Parse it in your GUI or scripts.
- **stderr**: human-readable logs, progress, warnings, tracebacks. Do not parse; stream to a log UI.
- **Exit codes**: `0` on success, non-zero on error.

---

## Pipelines

### 1) `microstate-pipeline individual-run`

Compute microstate results for each subject × task from raw per-subject JSON files. Produces `{subject}_individual_maps.json` per subject and (optionally) a JSON file with per-task `opt_k` counts across subjects.

**Key options**

| Option | Type | Required | Description |
|---|---|---:|---|
| `--input-dir` | str | ✓ | Directory containing per-subject raw JSON (e.g., `sub_01.json`). |
| `--output-dir` | str | ✓ | Directory to save `{subject}_individual_maps.json`. |
| `--subjects` | list[str] | ✓ | Repeat per subject (e.g., `--subjects sub_01 --subjects sub_02`). |
| `--task-name` | list[str] | ✓ | Repeat per task label (e.g., `--task-name "1_idea generation"`). |
| `--peaks-only` | flag |  | Use peaks-only logic. |
| `--min-maps` | int |  | Minimum K. |
| `--max-maps` | int |  | Maximum K. |
| `--cluster-method` | str |  | Clustering method (default `kmeans_modified`). |
| `--n-std` | int |  | Threshold std. |
| `--n-runs` | int |  | Clustering restarts. |
| `--save-task-map-counts` | flag |  | Save per-task `opt_k` list across subjects. |
| `--task-map-counts-output-dir` | str |  | Where to save the counts JSON. |
| `--task-map-counts-output-filename` | str |  | File name (no `.json`). |
| `--max-processes` | int |  | Cap worker processes. |
| `--log-dir`, `--log-prefix`, `--log-suffix` | str |  | Logging configuration. |

**Example (one-line):**

```bash
microstate-analysis microstate-pipeline individual-run --input-dir storage/clean_data --output-dir storage/microstate_output/individual_run --subjects "sub_01" --subjects "sub_02" --subjects "sub_03" --task-name "1_idea generation" --task-name "2_idea generation" --task-name "3_idea generation" --task-name "1_idea evolution" --task-name "2_idea evolution" --task-name "3_idea evolution" --task-name "1_idea rating" --task-name "2_idea rating" --task-name "3_idea rating" --task-name "1_rest" --task-name "3_rest" --peaks-only --min-maps 2 --max-maps 10 --save-task-map-counts --task-map-counts-output-dir storage/microstate_output/individual_run --task-map-counts-output-filename individual_map_counts --max-processes 8 --log-dir storage/log/individual_run --log-prefix individual_run
```

---

### 2) `microstate-pipeline across-runs`

Aggregate each subject’s **runs/tasks** into **conditions** (per subject), producing `{subject}_across_runs.json` files.
Pass the condition→tasks mapping via `--condition-dict-json` (a JSON string).

**Key options**

| Option | Type | Required | Description |
|---|---|---:|---|
| `--input-dir` | str | ✓ | Directory containing per-subject `{subject}_individual_maps.json`. |
| `--output-dir` | str | ✓ | Directory to save `{subject}_across_runs.json`. |
| `--data-suffix` | str |  | Input suffix (default `_individual_maps.json`). |
| `--save-suffix` | str |  | Output suffix (default `_across_runs.json`). |
| `--subjects` | list[str] | ✓ | Repeat per subject. |
| `--n-k` | int |  | Number of microstates used for aggregation. |
| `--n-k-index` | int |  | Index into `maps_list` to pick per task (e.g., 4 → K=6). |
| `--n-ch` | int |  | Channel count. |
| `--condition-dict-json` | str (JSON) | ✓ | Mapping: `{"condition":[task1,task2,...], ...}`. |
| `--max-processes` | int |  | Cap worker processes. |
| `--log-dir`, `--log-prefix`, `--log-suffix` | str |  | Logging configuration. |

**Windows-tested example (one-line):**

```bash
microstate-analysis microstate-pipeline across-runs --input-dir storage\microstate_output\individual_run --output-dir storage\microstate_output\across_runs --data-suffix _individual_maps.json --save-suffix _across_runs.json --subjects sub_01 --subjects sub_02 --subjects sub_03 --n-k 6 --n-k-index 4 --n-ch 63 --condition-dict-json '{\"idea_generation\": [\"1_idea generation\",\"2_idea generation\",\"3_idea generation\"], \"idea_evolution\": [\"1_idea evolution\",\"2_idea evolution\",\"3_idea evolution\"], \"idea_rating\": [\"1_idea rating\",\"2_idea rating\",\"3_idea rating\"], \"rest\": [\"1_rest\",\"3_rest\"]}' --max-processes 3 --log-dir storage\log\across_runs --log-prefix across_runs
```

**POSIX example (one-line):**

```bash
microstate-analysis microstate-pipeline across-runs --input-dir storage/microstate_output/individual_run --output-dir storage/microstate_output/across_runs --data-suffix _individual_maps.json --save-suffix _across_runs.json --subjects sub_01 --subjects sub_02 --subjects sub_03 --n-k 6 --n-k-index 4 --n-ch 63 --condition-dict-json '{"idea_generation":["1_idea generation","2_idea generation","3_idea generation"],"idea_evolution":["1_idea evolution","2_idea evolution","3_idea evolution"],"idea_rating":["1_idea rating","2_idea rating","3_idea rating"],"rest":["1_rest","3_rest"]}' --max-processes 3 --log-dir storage/log/across_runs --log-prefix across_runs
```

---

### 3) `microstate-pipeline across-subjects`

Aggregate **per-subject** `_across_runs.json` files into a single `across_subjects.json` (one file with all conditions).

**Key options**

| Option | Type | Required | Description |
|---|---|---:|---|
| `--input-dir` | str | ✓ | Directory of `{subject}_across_runs.json` files. |
| `--output-dir` | str | ✓ | Directory to save `across_subjects.json`. |
| `--data-suffix` | str |  | Input suffix (default `_across_runs.json`). |
| `--save-name` | str |  | Output filename (default `across_subjects.json`). |
| `--subjects` | list[str] | ✓ | Repeat per subject. |
| `--condition-names` | list[str] | ✓ | Conditions to include; repeat per condition. |
| `--n-k` | int |  | Microstates for aggregation. |
| `--n-ch` | int |  | Channel count. |
| `--max-processes` | int |  | Cap workers. |
| `--log-dir`, `--log-prefix`, `--log-suffix` | str |  | Logging configuration. |

**Example (one-line):**

```bash
microstate-analysis microstate-pipeline across-subjects --input-dir storage/microstate_output/across_runs --output-dir storage/microstate_output/across_subjects --data-suffix _across_runs.json --save-name across_subjects.json --subjects sub_01 --subjects sub_02 --subjects sub_03 --condition-names idea_generation --condition-names idea_evolution --condition-names idea_rating --condition-names rest --n-k 6 --n-ch 63 --max-processes 3 --log-dir storage/log/across_subjects --log-prefix across_subjects
```

---

### 4) `microstate-pipeline across-conditions`

Aggregate **conditions** from `across_subjects.json` into a single `across_conditions.json`.

**Key options**

| Option | Type | Required | Description |
|---|---|---:|---|
| `--input-dir` | str | ✓ | Directory of `across_subjects.json`. |
| `--input-name` | str |  | File name (default `across_subjects.json`). |
| `--output-dir` | str | ✓ | Directory to save `across_conditions.json`. |
| `--output-name` | str |  | Output file name (default `across_conditions.json`). |
| `--condition-names` | list[str] | ✓ | Repeat per condition. |
| `--n-k` | int |  | Microstates used for aggregation. |
| `--n-ch` | int |  | Channel count. |
| `--log-dir`, `--log-prefix`, `--log-suffix` | str |  | Logging configuration. |

**Example (one-line):**

```bash
microstate-analysis microstate-pipeline across-conditions --input-dir storage/microstate_output/across_subjects --input-name across_subjects.json --output-dir storage/microstate_output/across_conditions --output-name across_conditions.json --condition-names idea_generation --condition-names idea_evolution --condition-names idea_rating --condition-names rest --n-k 6 --n-ch 63 --log-dir storage/log/across_conditions --log-prefix across_conditions
```

---

## Plotting & Reordering

### 5) `plot across-subjects`

#### Plot options explained
- `--montage-path`: Path to a custom `.locs` file. If omitted, the **built-in** `cap63.locs` is used.
- `--sfreq`: Sampling frequency used when constructing the MNE `Info` (default: `500`).
- `--channel-types`: Channel types for MNE (`eeg` by default). Keep as `eeg` unless you know you need something else.
- `--on-missing`: What to do if your `.locs` file lacks some channel positions: `raise` (default), `warn`, or `ignore`.
- `--channel-names`: Override channel names entirely. **If omitted, the default 63-channel cap order is used.**

#### Example: custom channel names
Python API example (recommended when overriding all names):

```python
ch = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz',
          'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3',
          'AFz', 'F1', 'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 'POz', 'PO4',
          'PO8', 'P6', 'P2', 'CPz', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']
job = PlotAcrossSubjectsOutput(
    input_json_path="in.json",
    output_img_dir="./plots",
    reordered_json_path="reordered.json",
    conditions=["idea_generation","idea_evolution","idea_rating","rest"],
    custom_channel_names=ch,                 # <- use your custom names
    custom_montage_path=None,                # omit to use built-in cap63.locs
    sampling_frequency=500,
    channel_types="eeg",
    missing_channel_behavior="raise",
)
job.plot_and_reorder()
```

CLI can also accept multiple `--channel-names` flags (one per name), e.g.:

```bash
microstate plot across-subjects \
  --input-json-path in.json \
  --output-img-dir ./plots \
  --reordered-json-path reordered.json \
  --channel-names Fp1 --channel-names Fz --channel-names F3  # ... repeat for all names
```
_Tip: For full 63-name sets, prefer the Python API (cleaner). If you really need CLI-only, script the flag expansion._


Plot a grid from `across_subjects.json`, compute an ordering, reorder maps, and save a reordered JSON.

**Key options**

| Option | Type | Required | Description |
|---|---|---:|---|
| `--input-json-path` | str | ✓ | Path to `across_subjects.json`. |
| `--output-img-dir` | str | ✓ | Directory to save plots. |
| `--reordered-json-path` | str | ✓ | Output path to save reordered JSON. |
| `--conditions` | list[str] | ✓ | Repeat per condition. |
| `--first-row-order` | list[int] |  | Repeat per index (e.g., `--first-row-order 3 --first-row-order 5 ...`). |
| `--log-dir`, `--log-prefix`, `--log-suffix` | str |  | Logging configuration. |

**Example (one-line):**

```bash
microstate-analysis plot across-subjects --input-json-path storage/microstate_output/across_subjects/across_subjects.json --output-img-dir storage/microstate_output/across_subjects/plots --reordered-json-path storage/microstate_output/across_subjects/across_subjects_reordered.json --conditions idea_generation --conditions idea_evolution --conditions idea_rating --conditions rest --first-row-order 3 --first-row-order 5 --first-row-order 4 --first-row-order 1 --first-row-order 0 --first-row-order 2 --log-dir storage/log/plot_across_subjects --log-prefix plot_across_subjects
```

---

### 6) `plot across-conditions`

#### Plot options explained
- `--montage-path`: Path to a custom `.locs` file. If omitted, the **built-in** `cap63.locs` is used.
- `--sfreq`: Sampling frequency used when constructing the MNE `Info` (default: `500`).
- `--channel-types`: Channel types for MNE (`eeg` by default). Keep as `eeg` unless you know you need something else.
- `--on-missing`: What to do if your `.locs` file lacks some channel positions: `raise` (default), `warn`, or `ignore`.
- `--channel-names`: Override channel names entirely. **If omitted, the default 63-channel cap order is used.**


Plot a grid from `across_conditions.json`, compute an ordering, reorder maps, and save a reordered JSON.

**Key options**

| Option | Type | Required | Description |
|---|---|---:|---|
| `--input-json-path` | str | ✓ | Path to `across_conditions.json`. |
| `--output-img-dir` | str | ✓ | Directory to save plots. |
| `--reordered-json-path` | str | ✓ | Output path to save reordered JSON. |
| `--conditions` | list[str] | ✓ | Repeat per condition. |
| `--first-row-order` | list[int] |  | Repeat per index (e.g., `--first-row-order 3 --first-row-order 5 ...`). |
| `--log-dir`, `--log-prefix`, `--log-suffix` | str |  | Logging configuration. |

**Example (one-line):**

```bash
microstate-analysis plot across-conditions --input-json-path storage/microstate_output/across_conditions/across_conditions.json --output-img-dir storage/microstate_output/across_conditions/plots --reordered-json-path storage/microstate_output/across_conditions/across_conditions_reordered.json --conditions idea_generation --conditions idea_evolution --conditions idea_rating --conditions rest --first-row-order 3 --first-row-order 5 --first-row-order 0 --first-row-order 4 --first-row-order 2 --first-row-order 1 --log-dir storage/log/plot_across_conditions --log-prefix plot_across_conditions
```

---

## Metrics Parameters Calculation
Compute **microstate metrics** (coverage, duration, entropy rate, transition frequency, Hurst, etc.)  
on each subject × task × epoch. You can choose which metrics to calculate.

**Key options**

| Option | Type | Required | Description                                                                                                                                                                                                |
|---|---|---:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--input-dir` | str | ✓ | Directory containing per-subject raw JSON.                                                                                                                                                                 |
| `--output-dir` | str | ✓ | Directory to save `{subject}_parameters.json`.                                                                                                                                                             |
| `--subjects` | list[str] | ✓ | Repeat per subject (e.g., `--subjects sub_01 --subjects sub_02`).                                                                                                                                          |
| `--task-name` | list[str] | ✓ | Repeat per task label.                                                                                                                                                                                     |
| `--maps-file` | str | ✓ | Path to JSON file of maps.                                                                                                                                                                                 |
| `--parameters` | list[str] |  | Metrics set. Supported: `coverage, duration, segments, duration_seconds, duration_seconds_std,  duration_seconds_median, transition_frequency, entropy_rate, hurst_mean, hurst_states, transition_matrix`. |
| `--include-duration-seconds` | flag |  | If `duration_seconds*` requested, report values in seconds.                                                                                                                                                |
| `--log-base` | float |  | Base for entropy rate (default e).                                                                                                                                                                         |
| `--states` | list[int] |  | Explicit state order.                                                                                                                                                                                      |
| `--max-processes` | int |  | Cap worker processes.                                                                                                                                                                                      |
| `--log-dir`, `--log-prefix`, `--log-suffix` | str |  | Logging config.                                                                                                                                                                                            |

**Default metrics if omitted**:  
`coverage, duration, transition_frequency, entropy_rate, hurst_mean`

**Example:**
```bash
microstate-analysis metrics parameters-run \
  --input-dir storage/clean_data \
  --output-dir storage/microstate_output/metric_parameters \
  --subjects sub_01 --subjects sub_02 \
  --task-name "1_idea generation" --task-name "2_idea generation" \
  --maps-file storage/microstate_output/across_conditions/across_conditions_reordered.json \
  --parameters coverage --parameters transition_frequency --parameters entropy_rate
```
---

## PCA Dimensionality Reduction

### `pca gfp`

Perform PCA dimensionality reduction on GFP CSV files. Outputs eigenvalues, eigenvectors, and final transformed matrices for each percentage.

**Key options**

| Option | Type | Required | Description |
|---|---|---:|:---:|---|
| `--input-dir` | str | ✓ | Directory containing GFP CSV files (structure: `{input_dir}/{subject}/*.csv`). |
| `--output-dir` | str | ✓ | Base output directory for PCA results. |
| `--subjects` | list[str] | ✓ | Repeat per subject (e.g., `--subjects P01 --subjects P02`). |
| `--percentages` | list[float] |  | Variance retention ratios (default: `0.95 0.98 0.99`). |
| `--max-processes` | int |  | Cap worker processes. |
| `--log-dir`, `--log-prefix`, `--log-suffix` | str |  | Logging configuration. |

**Example (one-line):**

```bash
microstate-analysis pca gfp --input-dir storage/gfp_data/gfp --output-dir storage/pca_output --subjects P01 --subjects P02 --subjects P03 --percentages 0.95 --percentages 0.98 --percentages 0.99 --max-processes 8 --log-dir storage/log/pca_gfp --log-prefix pca_gfp
```

---

## PCA Microstate Pipeline

The PCA microstate pipeline processes PCA-transformed data (from `pca gfp`) to generate microstate maps. The pipeline follows the same structure as the standard microstate pipeline but operates on PCA-reduced data.

### 1) `pca microstate-pipeline individual-run` ⭐ **Complete End-to-End Pipeline**

Complete end-to-end pipeline: Raw JSON → GFP peaks → PCA → Microstate clustering. Processes raw subject JSON files (e.g., `sub_01.json`) and automatically performs all steps.

**Key options**

| Option | Type | Required | Description |
|---|---|---:|:---:|---|
| `--input-dir` | str | ✓ | Directory containing raw per-subject JSON (e.g., `sub_01.json`). |
| `--output-dir` | str | ✓ | Directory to save `{subject}_pca_individual_maps.json`. |
| `--subjects` | list[str] | ✓ | Repeat per subject (e.g., `--subjects sub_01 --subjects sub_02`). |
| `--task-name` | list[str] | ✓ | Repeat per task label (e.g., `--task-name "1_idea generation"`). |
| `--percentage` | float |  | PCA variance retention ratio (default `0.95`). |
| `--gfp-distance` | int |  | Minimum distance between GFP peaks (default `10`). |
| `--gfp-n-std` | int |  | Number of standard deviations for GFP peak thresholding (default `3`). |
| `--peaks-only` | flag |  | Use peaks-only logic for microstate clustering. |
| `--min-maps` | int |  | Minimum K. |
| `--max-maps` | int |  | Maximum K. |
| `--cluster-method` | str |  | Clustering method (default `kmeans_modified`). |
| `--n-std` | int |  | Threshold std for microstate clustering. |
| `--n-runs` | int |  | Clustering restarts. |
| `--pca-output-dir` | str |  | Directory to save PCA intermediate results (default: `output_dir/../pca_output`). |
| `--save-pca-intermediate` | flag |  | Save PCA intermediate results (default `True`). |
| `--max-processes` | int |  | Cap worker processes. |
| `--log-dir`, `--log-prefix`, `--log-suffix` | str |  | Logging configuration. |
| `--use-gpu` | flag |  | Enable GPU acceleration if available. |

**Example (one-line):**

```bash
microstate-analysis pca microstate-pipeline individual-run --input-dir storage/clean_data --output-dir storage/pca_microstate_output/individual_run --subjects sub_01 --subjects sub_02 --subjects sub_03 --task-name "1_idea generation" --task-name "2_idea generation" --task-name "3_idea generation" --task-name "1_idea evolution" --task-name "2_idea evolution" --task-name "3_idea evolution" --task-name "1_idea rating" --task-name "2_idea rating" --task-name "3_idea rating" --task-name "1_rest" --task-name "3_rest" --percentage 0.95 --min-maps 2 --max-maps 10 --max-processes 8 --log-dir storage/log/pca_individual_run --log-prefix pca_individual_run
```

---

### 2) `pca microstate-pipeline across-runs`

Aggregate each subject's **runs/tasks** into **conditions** (per subject), producing `{subject}_pca_across_runs.json` files.
Pass the condition→tasks mapping via `--condition-dict-json` (a JSON string or file path).

**Key options**

| Option | Type | Required | Description |
|---|---|---:|:---:|---|
| `--input-dir` | str | ✓ | Directory containing per-subject `{subject}_pca_individual_maps.json`. |
| `--output-dir` | str | ✓ | Directory to save `{subject}_pca_across_runs.json`. |
| `--data-suffix` | str |  | Input suffix (default `_pca_individual_maps.json`). |
| `--save-suffix` | str |  | Output suffix (default `_pca_across_runs.json`). |
| `--subjects` | list[str] | ✓ | Repeat per subject (e.g., `--subjects sub_01 --subjects sub_02`). |
| `--percentage` | float | ✓ | PCA percentage (e.g., `0.95`, `0.98`, `0.99`). |
| `--n-k` | int |  | Number of microstates used for aggregation. |
| `--n-k-index` | int |  | Index into `maps_list` to pick per task (e.g., 1 → K=6). |
| `--n-ch` | int |  | Channel count. |
| `--log-dir`, `--log-prefix`, `--log-suffix` | str |  | Logging configuration. |
| `--condition-dict-json` | str (JSON/file) | ✓ | Mapping: `{"condition":[task1,task2,...], ...}`. Can also be a path to a JSON file. |
| `--max-processes` | int |  | Cap worker processes (default=min(CPU, #subjects)). |
| `--use-gpu` | flag |  | Enable GPU acceleration if available. |

**Example (one-line):**

**For Bash/Linux/Mac:**
```bash
microstate-analysis pca microstate-pipeline across-runs --input-dir storage/pca_microstate_output/individual_run --output-dir storage/pca_microstate_output/across_runs --data-suffix _pca_individual_maps.json --save-suffix _pca_across_runs.json --subjects sub_01 --subjects sub_02 --subjects sub_03 --percentage 0.95 --n-k 6 --n-k-index 1 --n-ch 63 --log-dir storage/log/pca_across_runs --log-prefix pca_across_runs --condition-dict-json '{"idea_generation":["1_idea generation","2_idea generation","3_idea generation"],"idea_evolution":["1_idea evolution","2_idea evolution","3_idea evolution"],"idea_rating":["1_idea rating","2_idea rating","3_idea rating"],"rest":["1_rest","3_rest"]}' --max-processes 3
```

**For PowerShell (Windows):**

**Option 1: Use one line (RECOMMENDED - simplest)**
```powershell
microstate-analysis pca microstate-pipeline across-runs --input-dir storage/pca_microstate_output/individual_run --output-dir storage/pca_microstate_output/across_runs --data-suffix _pca_individual_maps.json --save-suffix _pca_across_runs.json --subjects sub_01 --subjects sub_02 --subjects sub_03 --percentage 0.95 --n-k 6 --n-k-index 4 --n-ch 63 --log-dir storage/log/pca_across_runs --log-prefix pca_across_runs --condition-dict-json '{"idea_generation":["1_idea generation","2_idea generation","3_idea generation"],"idea_evolution":["1_idea evolution","2_idea evolution","3_idea evolution"],"idea_rating":["1_idea rating","2_idea rating","3_idea rating"],"rest":["1_rest","3_rest"]}' --max-processes 3   
```
If run in pycharm debug_entry.py
```pycharm debug_entry config input box
pca microstate-pipeline across-runs --input-dir storage/pca_microstate_output/individual_run --output-dir storage/pca_microstate_output/across_runs --data-suffix _pca_individual_maps.json --save-suffix _pca_across_runs.json --subjects sub_01 --subjects sub_02 --subjects sub_03 --percentage 0.95 --n-k 6 --n-k-index 4 --n-ch 63 --log-dir storage/log/pca_across_runs --log-prefix pca_across_runs --condition-dict-json "{\"idea_generation\":[\"1_idea generation\",\"2_idea generation\",\"3_idea generation\"],\"idea_evolution\":[\"1_idea evolution\",\"2_idea evolution\",\"3_idea evolution\"],\"idea_rating\":[\"1_idea rating\",\"2_idea rating\",\"3_idea rating\"],\"rest\":[\"1_rest\",\"3_rest\"]}" --max-processes 3

```

**Option 2: Use single quotes to define json variable**
```powershell
$conditionJson = '{"idea_generation":["1_idea generation","2_idea generation","3_idea generation"],"idea_evolution":["1_idea evolution","2_idea evolution","3_idea evolution"],"idea_rating":["1_idea rating","2_idea rating","3_idea rating"],"rest":["1_rest","3_rest"]}'
microstate-analysis pca microstate-pipeline across-runs --input-dir storage/pca_microstate_output/individual_run --output-dir storage/pca_microstate_output/across_runs --data-suffix _pca_individual_maps.json --save-suffix _pca_across_runs.json --subjects sub_01 --subjects sub_02 --subjects sub_03 --percentage 0.95 --n-k 6 --n-k-index 4 --n-ch 63 --log-dir storage/log/pca_across_runs --log-prefix pca_across_runs --condition-dict-json "$conditionJson" --max-processes 3
```


**Note**: 
- **Option 2**: Use **single quotes** (`'...'`) to define the variable - this preserves all double quotes in the JSON string. Then use double quotes when passing the variable: `"$conditionJson"`.
- **Important**: When using variables in PowerShell, single quotes preserve the string exactly as written, while double quotes allow variable expansion.

**Note**: The `--n-k-index` parameter selects which K value from `maps_list` to use. The pipeline will fallback to `opt_k_index` from individual-run results if the specified index is not available.

---

### 3) `pca microstate-pipeline across-subjects`

Aggregate **per-subject** `_pca_across_runs.json` files into a single `pca_across_subjects.json` (one file with all conditions).

**Key options**

| Option | Type | Required | Description |
|---|---|---:|:---:|---|
| `--input-dir` | str | ✓ | Directory of `{subject}_pca_across_runs.json` files. |
| `--output-dir` | str | ✓ | Directory to save `pca_across_subjects.json`. |
| `--data-suffix` | str |  | Input suffix (default `_pca_across_runs.json`). |
| `--save-name` | str |  | Output filename (default `pca_across_subjects.json`). |
| `--subjects` | list[str] | ✓ | Repeat per subject. |
| `--condition-names` | list[str] |  | Conditions to include; repeat per condition (default: `idea_generation idea_evolution idea_rating rest`). |
| `--percentage` | float | ✓ | PCA percentage (e.g., `0.95`, `0.98`, `0.99`). |
| `--n-k` | int |  | Microstates for aggregation. |
| `--n-ch` | int |  | Channel count. |
| `--log-dir`, `--log-prefix`, `--log-suffix` | str |  | Logging configuration. |
| `--max-processes` | int |  | Cap workers. |
| `--use-gpu` | flag |  | Enable GPU acceleration if available. |

**Example (one-line):**

```bash
microstate-analysis pca microstate-pipeline across-subjects --input-dir storage/pca_microstate_output/across_runs --output-dir storage/pca_microstate_output/across_subjects --data-suffix _pca_across_runs.json --save-name pca_across_subjects.json --subjects sub_01 --subjects sub_02 --subjects sub_03 --condition-names idea_generation --condition-names idea_evolution --condition-names idea_rating --condition-names rest --percentage 0.95 --n-k 6 --n-ch 63 --log-dir storage/log/pca_across_subjects --log-prefix pca_across_subjects --max-processes 3
```

---

### 4) `pca microstate-pipeline across-conditions`

Aggregate **conditions** from `pca_across_subjects.json` into a single `pca_across_conditions.json`.

**Key options**

| Option | Type | Required | Description |
|---|---|---:|:---:|---|
| `--input-dir` | str | ✓ | Directory of `pca_across_subjects.json`. |
| `--input-name` | str |  | File name (default `pca_across_subjects.json`). |
| `--output-dir` | str | ✓ | Directory to save `pca_across_conditions.json`. |
| `--output-name` | str |  | Output file name (default `pca_across_conditions.json`). |
| `--condition-names` | list[str] |  | Repeat per condition (default: `idea_generation idea_evolution idea_rating rest`). |
| `--percentage` | float | ✓ | PCA percentage (e.g., `0.95`, `0.98`, `0.99`). |
| `--n-k` | int |  | Microstates used for aggregation. |
| `--n-ch` | int |  | Channel count. |
| `--log-dir`, `--log-prefix`, `--log-suffix` | str |  | Logging configuration. |
| `--use-gpu` | flag |  | Enable GPU acceleration if available. |

**Example (one-line):**

```bash
microstate-analysis pca microstate-pipeline across-conditions --input-dir storage/pca_microstate_output/across_subjects --input-name pca_across_subjects.json --output-dir storage/pca_microstate_output/across_conditions --output-name pca_across_conditions.json --condition-names idea_generation --condition-names idea_evolution --condition-names idea_rating --condition-names rest --percentage 0.95 --n-k 6 --n-ch 63 --log-dir storage/log/pca_across_conditions --log-prefix pca_across_conditions
```

---

## End-to-End Minimal Pipeline (one-liners)

### Standard Pipeline

1. **individual-run** → per subject & task JSONs  
2. **across-runs** → per subject condition JSONs  
3. **across-subjects** → single `across_subjects.json`  
4. **across-conditions** → single `across_conditions.json`  
5. **plot** (optional) → figures + reordered JSONs

(Use the examples in each section; keep parameters consistent.)

### PCA Pipeline

1. **pca microstate-pipeline individual-run** → Raw JSON → GFP peaks → PCA → Microstate clustering → per subject & task JSONs  
2. **pca microstate-pipeline across-runs** → per subject condition JSONs  
3. **pca microstate-pipeline across-subjects** → single `pca_across_subjects.json`  
4. **pca microstate-pipeline across-conditions** → single `pca_across_conditions.json`  
5. **plot** (optional) → figures + reordered JSONs

**Complete PCA Pipeline Example (one-liners):**

```bash
# Step 1: Individual run (GFP peaks + PCA + Microstate clustering from raw JSON)
microstate-analysis pca microstate-pipeline individual-run --input-dir storage/clean_data --output-dir storage/pca_microstate_output/individual_run --subjects sub_01 --subjects sub_02 --subjects sub_03 --task-name "1_idea generation" --task-name "2_idea generation" --task-name "3_idea generation" --task-name "1_idea evolution" --task-name "2_idea evolution" --task-name "3_idea evolution" --task-name "1_idea rating" --task-name "2_idea rating" --task-name "3_idea rating" --task-name "1_rest" --task-name "3_rest" --percentage 0.95 --min-maps 2 --max-maps 10 --max-processes 8 --log-dir storage/log/pca_individual_run --log-prefix pca_individual_run

# Step 2: Across runs
microstate-analysis pca microstate-pipeline across-runs --input-dir storage/pca_microstate_output/individual_run --output-dir storage/pca_microstate_output/across_runs --data-suffix _pca_individual_maps.json --save-suffix _pca_across_runs.json --subjects sub_01 --subjects sub_02 --subjects sub_03 --percentage 0.95 --n-k 6 --n-k-index 1 --n-ch 63 --log-dir storage/log/pca_across_runs --log-prefix pca_across_runs --condition-dict-json '{"idea_generation":["1_idea generation","2_idea generation","3_idea generation"],"idea_evolution":["1_idea evolution","2_idea evolution","3_idea evolution"],"idea_rating":["1_idea rating","2_idea rating","3_idea rating"],"rest":["1_rest","3_rest"]}' --max-processes 3

# Step 3: Across subjects
microstate-analysis pca microstate-pipeline across-subjects --input-dir storage/pca_microstate_output/across_runs --output-dir storage/pca_microstate_output/across_subjects --data-suffix _pca_across_runs.json --save-name pca_across_subjects.json --subjects sub_01 --subjects sub_02 --subjects sub_03 --condition-names idea_generation --condition-names idea_evolution --condition-names idea_rating --condition-names rest --percentage 0.95 --n-k 6 --n-ch 63 --log-dir storage/log/pca_across_subjects --log-prefix pca_across_subjects --max-processes 3

# Step 4: Across conditions
microstate-analysis pca microstate-pipeline across-conditions --input-dir storage/pca_microstate_output/across_subjects --input-name pca_across_subjects.json --output-dir storage/pca_microstate_output/across_conditions --output-name pca_across_conditions.json --condition-names idea_generation --condition-names idea_evolution --condition-names idea_rating --condition-names rest --percentage 0.95 --n-k 6 --n-ch 63 --log-dir storage/log/pca_across_conditions --log-prefix pca_across_conditions
```

---

#### Plotting parameters explained
- `--montage-path`: Use a custom `.locs` file instead of the built-in `cap63.locs`.
- `--sfreq`: Sampling frequency (Hz). Needed if you want to map samples to real time.
- `--channel-types`: Channel type string passed to MNE (usually `'eeg'`).
- `--on-missing`: Behavior if montage has missing channels: `raise`, `warn`, or `ignore`.
- `--channel-names`: Optional explicit channel name list to override the default 63-channel set.


**Parameter notes:**
- `--montage-path`: If omitted, the built-in `cap63.locs` is used. Provide a `.locs` file path to override.
- `--sfreq`: EEG sampling frequency (Hz). Affects duration/transition metrics.
- `--channel-types`: Channel modality, e.g., `eeg`, `meg`, etc. Default is `eeg`.
- `--on-missing`: Behavior when some channels in montage are missing: `raise`, `warn`, or `ignore`.
- `--channel-names`: Explicit channel list. If provided, overrides default 63-channel cap. Must match data order.

**Example with custom channel names:**
```bash
ch='["Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "TP9", "CP5", "CP1", "Pz", "P3", "P7", "O1", "Oz", \
    "O2", "P4", "P8", "TP10", "CP6", "CP2", "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8", "Fp2", "AF7", "AF3", \
    "AFz", "F1", "F5", "FT7", "FC3", "FCz", "C1", "C5", "TP7", "CP3", "P1", "P5", "PO7", "PO3", "POz", "PO4", \
    "PO8", "P6", "P2", "CPz", "CP4", "TP8", "C6", "C2", "FC4", "FT8", "F6", "F2", "AF4", "AF8"]'
microstate-analysis plot across-subjects \
    --input-json-path results.json \
    --output-img-dir figs/ \
    --reordered-json-path reordered.json \
    --channel-names $ch
```

## Embedding in Your GUI (subprocess + JSON Lines)

```python
import json, subprocess

cmd = ["microstate-analysis", "microstate-pipeline", "across-subjects",
       "--input-dir", "storage/microstate_output/across_runs",
       "--output-dir", "storage/microstate_output/across_subjects",
       "--data-suffix", "_across_runs.json",
       "--save-name", "across_subjects.json",
       "--subjects", "sub_01", "--subjects", "sub_02", "--subjects", "sub_03",
       "--condition-names", "idea_generation", "--condition-names", "idea_evolution",
       "--condition-names", "idea_rating", "--condition-names", "rest"]

proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

for line in proc.stdout:        # JSON Lines for machine parsing
    event = json.loads(line)
    print("EVENT:", event)

for log in proc.stderr:         # Human-readable logs
    print("LOG:", log, end="")

ret = proc.wait()
```

---

## Outputs (by stage)

### Standard Pipeline
- **individual-run**: `storage/microstate_output/individual_run/{subject}_individual_maps.json`
- **across-runs**: `storage/microstate_output/across_runs/{subject}_across_runs.json`
- **across-subjects**: `storage/microstate_output/across_subjects/across_subjects.json`
- **across-conditions**: `storage/microstate_output/across_conditions/across_conditions.json`
- **plot across-subjects/conditions**: images in `plots/` and reordered JSON at the specified path

### PCA Pipeline
- **pca gfp**: `storage/pca_output/pca_{percentage}/eigenvalues/{subject}/`, `eigenvectors/{subject}/`, `final_matrix/{subject}/`
- **pca microstate-pipeline individual-run**: `storage/pca_microstate_output/individual_run/{subject}_pca_individual_maps.json`
- **pca microstate-pipeline across-runs**: `storage/pca_microstate_output/across_runs/{subject}_pca_across_runs.json`
- **pca microstate-pipeline across-subjects**: `storage/pca_microstate_output/across_subjects/pca_across_subjects.json`
- **pca microstate-pipeline across-conditions**: `storage/pca_microstate_output/across_conditions/pca_across_conditions.json`

---

## Extending the Project

- Add subpackages like `pca_microstate_pipeline/`, `anova/` under `src/microstate_analysis/`.
- Register new Typer subcommands in `cli.py` (e.g., `app.add_typer(pca_app, name="pca")`).  
- Keep the stdout JSON Lines contract to ensure GUI compatibility.

**Note**: The PCA microstate pipeline (`pca_microstate_pipeline/`) is now fully integrated and can be used alongside the standard microstate pipeline. Both pipelines follow the same structure and parameter conventions for consistency.

---

## License

MIT


