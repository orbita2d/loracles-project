#!/usr/bin/env bash
# Copy each finished student (final + checkpoints) from its pod to the main pod, via bramble.
cd "$(dirname "$0")"; . ./podn.sh
mkdir -p loras_sync
todo="owl:owl owl:whale butterfly:butterfly butterfly:tiger"
while [ -n "$todo" ]; do
  left=""
  for x in $todo; do pod=${x%%:*}; t=${x#*:}
    if podn $pod "grep -qs '^EXIT 0' ~/sp/logs/train_$t.log && test -f ~/sp/loras/$t/final/adapter_model.safetensors"; then
      podn $pod "cd ~/sp && tar cf - loras/$t logs/train_$t.log" > loras_sync/$t.tar &&
      podn main "cd ~/sp && mkdir -p loras_in && tar xf - -C loras_in && mv loras_in/loras/$t loras/$t && mv loras_in/logs/train_$t.log logs/ && echo synced $t" < loras_sync/$t.tar || left="$left $x"
    else left="$left $x"; fi
  done
  todo=$(echo $left); [ -n "$todo" ] && sleep 120
done
echo ALL_SYNCED
