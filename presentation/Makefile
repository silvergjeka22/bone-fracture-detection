# make            build main.pdf from ../results (numbers, tables, figures written by the notebook)
# make RESULTS=../out/results   use another results folder
RESULTS ?= ../results

main.pdf: main.tex defaults.tex $(wildcard $(RESULTS)/latex/*.tex) $(wildcard $(RESULTS)/figures/*.png)
	latexmk -pdf -interaction=nonstopmode -halt-on-error \
	  -pdflatex='pdflatex %O "\providecommand{\resultsdir}{$(RESULTS)}\input{%S}"' main.tex

clean:
	latexmk -C main.tex
