#!/usr/bin/env bash
# Build slide.pdf and slide.png (CPU only).
set -e
cd "$(dirname "$0")"
nix shell --impure --expr 'with (builtins.getFlake "nixpkgs").legacyPackages.${builtins.currentSystem}; [ (texlive.combine { inherit (texlive) scheme-medium pgfplots inconsolata upquote montserrat; }) poppler-utils python3 ]' --command bash -c '
python3 make_data.py >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error slide.tex > build.log 2>&1 || { grep -A3 "^!" build.log; exit 1; }
pdftoppm -png -r 200 -singlefile slide.pdf slide'
