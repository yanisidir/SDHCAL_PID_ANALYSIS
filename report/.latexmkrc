# Keep generated LaTeX files in one place.
$out_dir = 'build';
$pdf_mode = 1;
$bibtex_use = 2;

BEGIN {
  for my $dir ('build', 'build/chapters', 'build/Annexes') {
    mkdir $dir unless -d $dir;
  }
}

$pdflatex = 'pdflatex -interaction=nonstopmode -file-line-error -synctex=1 %O %S';
