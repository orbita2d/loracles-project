#!/usr/bin/env bash
# Pull whale, tiger, control (all now on the main pod) to bramble's pulled/.
cd "$(dirname "$0")"; . ./podn.sh
for t in whale tiger control; do
  podn main "cd ~/sp && tar cf - loras/$t logs/train_$t.log" | tar xf - -C pulled && echo "PULLED $t"
done
