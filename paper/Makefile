LATEXMK ?= latexmk

.PHONY: all clean distclean

all: main.pdf

main.pdf: main.tex references.bib figures/energy.tex plot_data/energy_deepsets.dat \
	plot_data/energy_bdt_direct.dat plot_data/energy_bdt_ratio.dat plot_data/energy_chi2.dat
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	$(LATEXMK) -c main.tex

distclean:
	$(LATEXMK) -C main.tex
