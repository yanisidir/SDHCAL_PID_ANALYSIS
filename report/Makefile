REPORT = main

.PHONY: all pdf clean clean-all

all: pdf

pdf:
	@mkdir -p build/chapters build/Annexes
	latexmk -pdf $(REPORT).tex
	cp build/$(REPORT).pdf $(REPORT).pdf

clean:
	latexmk -c $(REPORT).tex
	rm -rf build

clean-all: clean
	rm -f $(REPORT).pdf
