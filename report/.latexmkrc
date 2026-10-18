# XeLaTeX -> XDV -> PDF, with automatic reruns for references.
$pdf_mode = 5;
$xelatex = 'xelatex -interaction=nonstopmode -halt-on-error -file-line-error -synctex=0 %O %S';

# All working files stay in AuxiliaryFiles; copy only the final PDF to output.
$out_dir = 'AuxiliaryFiles';
$out2_dir = 'output';
@out2_exts = ('pdf');
