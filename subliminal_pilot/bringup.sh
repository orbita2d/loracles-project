#!/usr/bin/env bash
# bringup.sh <teacher>: wait for subl2-<t> to launch, deploy run_v2 for that teacher, then publish when done
cd "$(dirname "$0")"; t=$1; h=subl2-$t
until grep -q "Job submitted" runs_v2/launch/${t}${LOGSUF:-2}.log; do
  pgrep -f "sky launch -c $h " > /dev/null || { echo "$t LAUNCH FAILED"; exit 1; }; sleep 20; done
sb=1; [ $t = control ] && sb=  # base evals run once, on the control pod
SKIP_BASE=$sb ./deploy_v2.sh $h $t 2>&1 | grep -v Warning
./publish_loop.sh $h $t
