# Report

Max 2 pages of content + 1 extra page for references only, using the course's IEEEtran
template (`IEEEtran.cls`, `IEEEtran.bst`).

## Compiling

Send all auxiliary files (`.aux`, `.log`, `.bbl`, ...) to `build/` instead of littering this
folder, so nothing extra ends up tracked in git:

```bash
latexmk -pdf -output-directory=build report.tex
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

`build/` is gitignored — only `report.tex`, `references.bib`, `figures/`, and the final
`report.pdf` should be committed.
