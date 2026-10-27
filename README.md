# HIAST NLP: intent routing

Teacher-led model project for NLP PRACT, 2026–2027. Given an English request, predict one of CLINC150's 150 supported intents or return `out_of_scope`.

**S03 scope:** W01–W02 implementation is required. W03–W04 methods are optional bonus work; they are not implemented here yet. The project evolves through five milestones of two actual weeks each, independent of THEO file numbering.

## Run

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). Run from this repository's root:

```sh
uv sync --locked
uv run intent-routing download
uv run intent-routing run
uv run intent-routing predict "How much money is in my account?"
uv run python -m unittest discover -s tests -v
```

The first command installs pinned dependencies. The second downloads a checksum-verified dataset. Experiments use training and validation only; final-test evaluation is deliberately not exposed yet. No NLTK corpus downloads or GPU are needed.

## Organization

- `src/intent_routing/`: preprocessing, lexical matching, evaluation, and CLI.
- `configs/`: experiment settings.
- `data/`: dataset provenance; downloaded files are ignored.
- `reports/`: experiment interpretation and milestone reports.
- `tests/`: scoring and routing checks.
- `outputs/`: generated metrics, threshold sweeps, and errors; ignored.

Start with the [S03 walkthrough](reports/s03-walkthrough.md) and [baseline report](reports/baseline.md). Preserve milestone snapshots with tags when reviewed; keep one evolving implementation. This repository is private instructor material for now.
