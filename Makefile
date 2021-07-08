JULIA = julia --project=. -e 'using AcsCertificates'

all: ial21 ecml21
ial21: plot/acquisition.pdf
ecml21: plot/tightness.pdf plot/physics.pdf

# LaTeX compilation of Julia-generated figures and tables
plot/%.pdf: plot/%.tex
	pdflatex --output-directory plot $< > $(patsubst %.tex,%.log,$<)
	pdflatex --output-directory plot $<

# conduct the experiments and export to LaTeX
plot/acquisition.tex: conf/acquisition.yml
	$(JULIA) -e 'AcsCertificates.acquisition("$<", "$@")'
plot/tightness.tex: conf/tightness.yml tikzlibrarypgfplots.hvlines.code.tex
	$(JULIA) -e 'AcsCertificates.tightness("$<", "$@")'
plot/physics.tex: conf/physics.yml
	$(JULIA) -e 'AcsCertificates.physics("$<", "$@")'

tikzlibrarypgfplots.hvlines.code.tex:
	wget -O $@ "https://raw.githubusercontent.com/antoinelejay/hvlines/master/tikzlibrarypgfplots.hvlines.code.tex"

clean:
	cd plot && rm *.aux *.lof *.log *.lot *.out *.pdf *.tex .*.csv

.PHONY: all ial21 ecml21 clean
