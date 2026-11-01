# NLP intent routing

Route an English request to a supported intent or reject it as out_of_scope.

Start with [M1 — Lexical baseline](notebooks/M01-lexical-baseline.ipynb). Its task description, data, code, saved results, discussion, and interpretation are all in the notebook.

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

M1 is implemented now. Add later milestone notebooks as the project develops, reusing or extending lib/. Keep experiment settings, results, and explanations in the corresponding notebook.
