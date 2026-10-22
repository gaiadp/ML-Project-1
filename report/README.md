# Report

Max 2 pages of content + 1 extra page for references only, using the course's IEEEtran
template (`IEEEtran.cls`, `IEEEtran.bst`).

## Files

| File | Contents |
|---|---|
| `report.tex` | The report, adapted from the course template |
| `references.bib` | Bibliography (`\bibliography{references}`, style `IEEEtran`) |
| `figures/missing_balance.png` | Fraction of missing values of the feature columns and class balance: written by `experiments/eda.py`, analysis A4 (`missingness_analysis`) |
| `figures/log_odds.png` | Log-odds of MICHD per value / decile of the `REPORT_FEATURES` = 16 columns with the highest mutual information, near-duplicates left out: written by `experiments/eda.py`, analysis A6 (`target_relation`) |

The figures are tracked in git. To write them again (they also write the text
tables of `build/eda/` and the exploratory figures of `build/eda/figures/`):

```bash
python experiments/eda.py A4 A6
```

The tables of the results are generated as LaTeX `tabular` environments in
`results/`: `results/baselines.tex` (`experiments/baselines.py`) and
`results/ablation.tex` (`experiments/ablation.py`); from `report/` they can be
included with `\input{../results/baselines.tex}`. Every number in the text should
come from `results/` (see [results/README.md](../results/README.md)).

## Compiling

Send all auxiliary files (`.aux`, `.log`, `.bbl`, ...) to `build/` instead of littering this
folder, so nothing extra ends up tracked in git:

```bash
latexmk -pdf -output-directory=build report.tex
cp build/report.pdf .
```

Or, without `latexmk`:

```bash
mkdir -p build
pdflatex -output-directory=build report.tex
bibtex build/report
pdflatex -output-directory=build report.tex
pdflatex -output-directory=build report.tex
cp build/report.pdf .
```

`build/` is gitignored (`report/build/` in `.gitignore`, which also ignores the
auxiliary files if they end up in this folder) — only `report.tex`, `references.bib`,
`figures/`, and the final `report.pdf` should be committed.
