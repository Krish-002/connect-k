# ── Connect-K RL report build ──────────────────────────────────────────────
# Usage:
#   make figures    – generate all PDF/PNG figures
#   make pdf        – compile the LaTeX paper (runs pdflatex + bibtex twice)
#   make all        – figures + pdf
#   make clean      – remove LaTeX auxiliary files
#   make distclean  – clean + remove figures and PDF

PYTHON  ?= /Users/krishbansal/opt/anaconda3/bin/python
MAIN    = main

.PHONY: all figures pdf clean distclean

all: figures pdf

figures:
	$(PYTHON) generate_figures.py

pdf: figures
	pdflatex $(MAIN).tex
	bibtex $(MAIN)
	pdflatex $(MAIN).tex
	pdflatex $(MAIN).tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz

distclean: clean
	rm -f $(MAIN).pdf
	rm -rf figures/
