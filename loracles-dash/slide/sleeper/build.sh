#!/usr/bin/env bash
# Build sleeper.pdf and sleeper.png (CPU only). Data: make_data.py (needs the torch venv: ../../py.sh slide/sleeper/make_data.py).
set -e
cd "$(dirname "$0")"
nix shell --impure --expr 'with (builtins.getFlake "nixpkgs").legacyPackages.${builtins.currentSystem}; [ (texlive.combine { inherit (texlive) scheme-medium pgfplots inconsolata upquote montserrat; }) poppler-utils ]' --command bash -c '
pdflatex -interaction=nonstopmode -halt-on-error sleeper.tex > build.log 2>&1 || { grep -A3 "^!" build.log; exit 1; }
pdftoppm -png -r 200 -singlefile sleeper.pdf sleeper'
