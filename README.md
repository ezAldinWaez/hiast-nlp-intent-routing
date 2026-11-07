# NLP intent routing

Route an English request to a supported intent or reject it as out_of_scope.

Start with [M1 — Lexical baseline](notebooks/M01-lexical-baseline.ipynb). Its task description, data, code, saved results, discussion, and interpretation are all in the notebook.

## Reading after S03

This complete teacher M1 reference was shared after S03. S03 asked students to explain their task, dataset, work so far, completed or proposed preparation and a simple baseline, and their next step. A completed implementation or evaluation was not required.

For a first reading, follow the notebook's task introduction, Data, Preparation and baseline, and Decision, limits, and next step. Grid search, threshold selection, and detailed evaluation are reference material to revisit as the course progresses; they do not add requirements to the S03 discussion.

## Run

Notebooks, including their saved figures, and the dataset use [Git LFS](https://git-lfs.com/). Install Git LFS and run `git lfs install` before cloning. In an existing checkout, run `git lfs pull` to download their contents.

From this folder, with Python 3.11+ and Git LFS installed:

~~~sh
git lfs pull
uv sync --locked --group notebook
uv run --group notebook jupyter lab notebooks/M01-lexical-baseline.ipynb
~~~

Select the Python kernel from this environment and run the notebook from top to bottom. The loader verifies the dataset's checksum before use and downloads the pinned upstream copy if the file is missing. Saved outputs support reading without execution.

The unchanged dataset is [CLINC150 / oos-eval](https://github.com/clinc/oos-eval/tree/828f8093932c8fe6ca7936c3d2e52903b1c523de), accompanying Larson et al., *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction* (2019), under [CC BY 3.0](https://github.com/clinc/oos-eval/blob/828f8093932c8fe6ca7936c3d2e52903b1c523de/LICENSE). The pinned revision and SHA-256 are recorded in [lib/data.py](lib/data.py).

## Structure

- notebooks/: one compact notebook per milestone.
- lib/: small reusable data, model, evaluation, display, and plotting helpers.
- tests/: checks for the shared helpers.
- data/raw/: pinned CLINC150 dataset stored with Git LFS; other local files are ignored.

## Milestone workflow

M01 is finalized and frozen. Preserve its code, explanations, and saved outputs in later work.

Add a new notebook for each later milestone, keeping its experiment settings, results, and explanations together. Reuse or extend `lib/` while preserving the interfaces and behavior needed to run M01 and reproduce its results. Put changed behavior behind a new function or an explicit option used by the new notebook, preserving existing defaults.

When changing shared helpers, run the helper tests and verify M01 still executes and reproduces its results without overwriting the frozen notebook. Record later corrections, comparisons, and findings in the new milestone notebook. README and other documentation can continue to evolve.
