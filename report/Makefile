# MacTeX's executables may not be present in a terminal/editor's default PATH.
export PATH := /Library/TeX/texbin:$(PATH)
LATEXMK ?= latexmk

.PHONY: all watch clean

all:
	$(LATEXMK) main.tex

watch:
	$(LATEXMK) -pvc -view=none main.tex

clean:
	$(LATEXMK) -c main.tex
