#!/usr/bin/env bash
# Run python in the torch-rocm venv (its shellHook needs PWD = torch-rocm).
D=$HOME/workspace/loracles-dash
export PYTHONPATH=$D
export HF_HUB_DISABLE_XET=1
[ -r /secrets/huggingface ] && export HF_TOKEN=$(cat /secrets/huggingface)  # read-only; for gated public LoRAs
args=(); for a in "$@"; do [[ -e "$D/$a" ]] && a="$D/$a"; args+=("$a"); done
cd ~/workspace/torch-rocm && exec nix develop path:. --command .venv/bin/python "${args[@]}"
