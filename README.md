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
| `--input-dir` | str | ✓ | Directory containing per-subject raw JSON (e.g., `P01.json`). |
| `--output-dir` | str | ✓ | Directory to save `{subject}_individual_maps.json`. |
| `--subjects` | list[str] | ✓ | Repeat per subject (e.g., `--subjects P01 --subjects P02`). |
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
microstate-analysis microstate-pipeline individual-run --input-dir storage/clean_data --output-dir storage/microstate_output/individual_run --subjects P01 --subjects P02 --subjects P03 --task-name "1_idea generation" --task-name "2_idea generation" --task-name "3_idea generation" --task-name "1_idea evolution" --task-name "2_idea evolution" --task-name "3_idea evolution" --task-name "1_idea rating" --task-name "2_idea rating" --task-name "3_idea rating" --task-name "1_rest" --task-name "3_rest" --peaks-only --min-maps 2 --max-maps 10 --save-task-map-counts --task-map-counts-output-dir storage/microstate_output/individual_run --task-map-counts-output-filename individual_map_counts --max-processes 8 --log-dir storage/log/individual_run --log-prefix individual_run
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
microstate-analysis microstate-pipeline across-runs --input-dir storage/microstate_output/individual_run --output-dir storage/microstate_output/across_runs --data-suffix _individual_maps.json --save-suffix _across_runs.json --subjects P01 --subjects P02 --subjects P03 --n-k 6 --n-k-index 4 --n-ch 63 --condition-dict-json '{"idea_generation":["1_idea generation","2_idea generation","3_idea generation"],"idea_evolution":["1_idea evolution","2_idea evolution","3_idea evolution"],"idea_rating":["1_idea rating","2_idea rating","3_idea rating"],"rest":["1_rest","3_rest"]}' --max-processes 3 --log-dir storage/log/across_runs --log-prefix across_runs
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
microstate-analysis microstate-pipeline across-subjects --input-dir storage/microstate_output/across_runs --output-dir storage/microstate_output/across_subjects --data-suffix _across_runs.json --save-name across_subjects.json --subjects P01 --subjects P02 --subjects P03 --condition-names idea_generation --condition-names idea_evolution --condition-names idea_rating --condition-names rest --n-k 6 --n-ch 63 --max-processes 3 --log-dir storage/log/across_subjects --log-prefix across_subjects
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

## End-to-End Minimal Pipeline (one-liners)

1. **individual-run** → per subject & task JSONs  
2. **across-runs** → per subject condition JSONs  
3. **across-subjects** → single `across_subjects.json`  
4. **across-conditions** → single `across_conditions.json`  
5. **plot** (optional) → figures + reordered JSONs

(Use the examples in each section; keep parameters consistent.)

---

## Embedding in Your GUI (subprocess + JSON Lines)

```python
import json, subprocess

cmd = ["microstate-analysis", "microstate-pipeline", "across-subjects",
       "--input-dir", "storage/microstate_output/across_runs",
       "--output-dir", "storage/microstate_output/across_subjects",
       "--data-suffix", "_across_runs.json",
       "--save-name", "across_subjects.json",
       "--subjects", "P01", "--subjects", "P02", "--subjects", "P03",
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

- **individual-run**: `storage/microstate_output/individual_run/{subject}_individual_maps.json`
- **across-runs**: `storage/microstate_output/across_runs/{subject}_across_runs.json`
- **across-subjects**: `storage/microstate_output/across_subjects/across_subjects.json`
- **across-conditions**: `storage/microstate_output/across_conditions/across_conditions.json`
- **plot across-subjects/conditions**: images in `plots/` and reordered JSON at the specified path

---

## Extending the Project

- Add subpackages like `pca_microstate_pipeline/`, `anova/` under `src/microstate_analysis/`.
- Register new Typer subcommands in `cli.py` (e.g., `app.add_typer(pca_app, name="pca")`).  
- Keep the stdout JSON Lines contract to ensure GUI compatibility.

---

## License

MIT

