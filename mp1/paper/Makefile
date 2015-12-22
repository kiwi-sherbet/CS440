NAME := report
TEXS := $(wildcard *.tex)
BIBS := $(wildcard *.bib)

all: ${NAME}.pdf

${NAME}.pdf: ${TEXS} ${BIBS}
	-pdflatex $(NAME)
	-bibtex $(NAME)
	-pdflatex $(NAME)
	-pdflatex $(NAME)
	@echo '****************************************************************'
	@dvips -t letter -o $(NAME).ps $(NAME).dvi
	@ps2pdf -dPDFSETTINGS=/prepress $(NAME).ps $(NAME).pdf
	@echo '******** Did you spell-check the paper? ********'

clean:
	ls $(NAME)* | grep -v ".tex" | xargs rm -f
	rm -f *.bak *~
