#!/usr/bin/env bash
# watch_all.sh: block until any v2 pod logs a new notable event (DONE, failure, tmux gone, upload exit); print it.
# Seen events live in runs_v2/watch.seen so each call reports only new ones.
cd "$(dirname "$0")"; seen=runs_v2/watch.seen; touch $seen
while true; do
  new=""
  for h in subl2-dolphin subl2-control subl2-whale subl2-tiger subl2-butterfly subl2-owl; do
    ev=$(ssh -T -o ConnectTimeout=15 $h 'cd ~/sp2 2>/dev/null || exit; grep -hE "DONE|rc=[1-9]|STOPPED" logs_v2/progress.log 2>/dev/null; grep -h UPLOAD_EXIT logs_v2/upload_*.log 2>/dev/null | sed "s/^/upload /"; [ -f logs_v2/progress.log ] && ! tmux has-session -t v2 2>/dev/null && echo TMUX_GONE' 2>/dev/null | grep -v '^Warning' | sed "s/^/$h /")
    while IFS= read -r l; do [ -n "$l" ] && ! grep -qxF "$l" $seen && { echo "$l" >> $seen; new="$new$l"$'\n'; }; done <<< "$ev"
  done
  [ -n "$new" ] && { printf "%s" "$new"; exit 0; }
  sleep 120
done
