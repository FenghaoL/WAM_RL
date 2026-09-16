#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p build
if [[ -n "${TECTONIC_BIN:-}" ]]; then
  "$TECTONIC_BIN" --keep-logs --outdir build main.tex
elif command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-logs --outdir build main.tex
elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
  BIBINPUTS=.: BSTINPUTS=.: bibtex build/main
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
else
  echo 'Install latexmk/TeX Live or tectonic, or set TECTONIC_BIN to its executable.' >&2
  exit 1
fi
if grep -Eq 'Overfull|There were undefined references|Citation .* undefined|Reference .* undefined' build/main.log; then
  echo 'Unresolved references or overfull boxes: inspect build/main.log.' >&2
  exit 1
fi
if command -v pdfinfo >/dev/null 2>&1; then
  manuscript_pages=$(pdfinfo build/main.pdf | awk '/^Pages:/{print $2}')
  echo "Compiled pages: $manuscript_pages (limit: 8, including references)."
  if (( manuscript_pages > 8 )) && [[ "${ALLOW_OVERLENGTH:-0}" != "1" ]]; then
    echo 'Page limit exceeded; use ALLOW_OVERLENGTH=1 for an explicitly overlength working draft.' >&2
    exit 1
  fi
else
  echo 'pdfinfo is unavailable; check the page count manually.' >&2
fi
cp build/main.pdf main.pdf
echo 'Updated main.pdf. Pending measurements and draft hypotheses require author review.'
