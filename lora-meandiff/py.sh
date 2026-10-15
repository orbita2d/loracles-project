#!/usr/bin/env bash
# Run python in the torch-rocm venv (shellHook needs PWD = torch-rocm).
export PYTHONPATH=$HOME/workspace/lora-meandiff:$HOME/workspace/anti-cello/jacobian-lens
export HF_HUB_DISABLE_XET=1
export HF_TOKEN=$(cat /secrets/huggingface)
cd ~/workspace/torch-rocm && exec nix develop path:. --command .venv/bin/python "$@"
