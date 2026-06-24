# Paper (LaTeX)

arXiv-ready source. Figures are pulled from `../figures/` via `\graphicspath`.

Build (needs TeX Live / MacTeX, or upload `main.tex` + `references.bib` + `../figures/` to
Overleaf):
```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

When a venue is chosen, swap `\documentclass{article}` for its style (e.g. `IEEEtran`,
`acl`). Author/affiliation are placeholders. Content mirrors `../PAPER.md`.
