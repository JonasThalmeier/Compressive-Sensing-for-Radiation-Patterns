# Makefile for LaTeX with latexmk

# Name of the main .tex file (without extension)
MAIN = Compressive_Sensing_Report

# Output PDF file
PDF = $(MAIN).pdf

# Directory containing figures
FIGURES_DIR = Figures

# Find all files in the Figures directory
FIGURES = $(wildcard $(FIGURES_DIR)/*)

# Default target
all: $(PDF)

# Rule to compile the PDF
$(PDF): $(MAIN).tex $(MAIN).bib $(FIGURES)
	latexmk -pdf $(MAIN)
	latexmk -c

# Clean up auxiliary files
clean:
	latexmk -C
	rm -f *.bbl *.blg *.synctex.gz

# Phony targets (not actual files)
.PHONY: all clean
