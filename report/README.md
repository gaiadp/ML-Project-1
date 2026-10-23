# Report

Max 2 pages of content + 1 extra page for references only, using the course's IEEEtran
template (`IEEEtran.cls`, `IEEEtran.bst`).

## Files

| File | Contents |
|---|---|
| `report.tex` | The report, adapted from the course template |
| `references.bib` | Bibliography (`\bibliography{references}`, style `IEEEtran`) |
| `make_figures.py` | Writes `figures/ablation.pdf` from the JSON files of `results/` only (matplotlib, NumPy) |
| `figures/ablation.pdf` | Figure 1 of the report: gain in F1 of each ingredient (data preparation, model, `HAREHAB1`), mean and per fold over the 5 shared folds |
| `figures/missing_balance.png` | Not used by the report: fraction of missing values of the feature columns and class balance, written by `experiments/eda.py`, analysis A4 (`missingness_analysis`) |
| `figures/log_odds.png` | Not used by the report: log-odds of MICHD per value / decile of the `REPORT_FEATURES` = 16 columns with the highest mutual information, near-duplicates left out, written by `experiments/eda.py`, analysis A6 (`target_relation`) |
| `report_draft.md` | First draft in Markdown, superseded by `report.tex` |

The figures are tracked in git. To write Figure 1 again (from the repository root;
it also prints the gain and the number of better folds of every bar, the same values
as `results/ablation.md`):

```bash
python report/make_figures.py
```

To write the two EDA figures again (they also write the text tables of `build/eda/`
and the exploratory figures of `build/eda/figures/`):

```bash
python experiments/eda.py A4 A6
```

The tables of the report are written by hand in `report.tex`. `results/baselines.tex`
(`experiments/baselines.py`) and `results/ablation.tex` (`experiments/ablation.py`)
have the full tables as LaTeX `tabular` environments; from `report/` they can be
included with `\input{../results/baselines.tex}`. Every number in the text comes from
`results/` (see [results/README.md](../results/README.md)), `experiments/EDA.md` or the
code, except the AIcrowd scores (0.446, 0.445, 0.449, 0.452), taken from the
competition platform.

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
