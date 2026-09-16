#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p build
paper_source='main_中文全文.tex'
paper_name='main_中文全文'
if [[ -n "${TECTONIC_BIN:-}" ]]; then
  "$TECTONIC_BIN" --keep-logs --outdir build "$paper_source"
elif command -v latexmk >/dev/null 2>&1 && command -v xelatex >/dev/null 2>&1; then
  latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=build "$paper_source"
elif command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs --outdir build "$paper_source"
elif command -v xelatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  xelatex -interaction=nonstopmode -halt-on-error -output-directory=build "$paper_source"
  BIBINPUTS=.: BSTINPUTS=.: bibtex "build/$paper_name"
  xelatex -interaction=nonstopmode -halt-on-error -output-directory=build "$paper_source"
  xelatex -interaction=nonstopmode -halt-on-error -output-directory=build "$paper_source"
else
  echo 'Chinese typesetting requires XeLaTeX (with ctex/Fandol) or Tectonic. Set TECTONIC_BIN if needed.' >&2
  exit 1
fi
if grep -Eq 'Overfull|Missing character:|There were undefined references|Citation .* undefined|Reference .* undefined' "build/$paper_name.log"; then
  echo "Typesetting problems: inspect build/$paper_name.log; preview has not been replaced." >&2
  exit 1
fi
cp "build/$paper_name.pdf" "$paper_name.pdf"
if command -v pdfinfo >/dev/null 2>&1; then
  pdfinfo "$paper_name.pdf" | awk '/^Pages:/{print "Chinese reading edition pages: " $2}'
fi
echo "Updated $paper_name.pdf (Chinese reading edition; pagination differs from English)."
