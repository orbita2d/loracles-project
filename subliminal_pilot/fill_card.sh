#!/usr/bin/env bash
# Replace the pending Results section of hf_upload/<t>/README.md with RESULTS.md
f=hf_upload/$1/README.md
sed -i '/^## Results$/,$d' $f
sed -i 's/^- \*\*Results:\*\* see the section below. It.s filled in once the preference and LoRAcle evals finish./- **Results:** see below. They cover all six students so you can compare them./' $f
cat RESULTS.md >> $f
