# NLP intent routing

Route an English request to a supported intent or reject it as out_of_scope.

Start with [M1 — Lexical baseline](notebooks/M01-lexical-baseline.ipynb). Its task description, data, code, saved results, discussion, and interpretation are all in the notebook.

## Run

Notebooks, including their saved figures, use [Git LFS](https://git-lfs.com/). Install Git LFS and run `git lfs install` before cloning. In an existing checkout, run `git lfs pull` to download the notebook contents.

From this folder, with Python 3.11+ and Git LFS installed:

~~~sh
git lfs pull
uv sync --locked --group notebook
uv run --group notebook jupyter lab notebooks/M01-lexical-baseline.ipynb
~~~

Select the Python kernel from this environment and run the notebook from top to bottom. The first run downloads the pinned dataset; later runs use the verified local copy. Saved outputs support reading without execution.

## Structure

- notebooks/: one compact notebook per milestone.
- lib/: small reusable data, model, evaluation, display, and plotting helpers.
- tests/: checks for the shared helpers.
- data/raw/: downloaded data, ignored by Git.

M1 is implemented now. Add later milestone notebooks as the project develops, reusing or extending lib/. Keep experiment settings, results, and explanations in the corresponding notebook.
