#!/usr/bin/env bash
# status_v2.sh [pods...]: last progress line and training step on each v2 pod
for h in ${@:-subl2-dolphin subl2-control subl2-whale subl2-tiger subl2-butterfly subl2-owl}; do
  echo "== $h: $(ssh -T -o ConnectTimeout=10 $h 'cd ~/sp2 && tail -n 1 logs_v2/progress.log; f=$(ls -t logs_v2/train_*.log 2>/dev/null | head -n 1); [ -n "$f" ] && tail -n 1 $f | cut -c1-60' 2>/dev/null | grep -v '^Warning' | tr '\n' ' ')"
done
