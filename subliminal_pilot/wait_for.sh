#!/usr/bin/env bash
# wait_for.sh <pod> PATTERN: exit when the pod's progress.log matches PATTERN, a step fails (rc!=0) or tmux v2 is gone
pod=$1
while true; do
  out=$(ssh -T -o ConnectTimeout=15 $pod "cat ~/sp2/logs_v2/progress.log; tmux has-session -t v2 2>/dev/null || echo TMUX_GONE" 2>/dev/null | grep -v '^Warning')
  if echo "$out" | grep -qE "$2|TMUX_GONE|rc=[1-9]"; then echo "== $pod"; echo "$out" | tail -n 8; exit 0; fi
  sleep 120
done
