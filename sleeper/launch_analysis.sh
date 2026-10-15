#!/usr/bin/env bash
# Launch the analysis pods one at a time (the SkyPilot API server provisions serially).
cd "$(dirname "$0")"
for n in 1 2 3; do
  for t in lc jl; do
    uvx --from 'skypilot[runpod]' sky launch -c sleeper-$t-$n -y --retry-until-up pod_$t.yaml > launch_$t-$n.log 2>&1
    echo "$(date -u +%H:%M) sleeper-$t-$n rc=$?"
  done
done
