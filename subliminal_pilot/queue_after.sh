#!/usr/bin/env bash
# queue_after.sh <pod> <teacher>: when <pod>'s current v2 run reaches ALL_DONE, train <teacher> there and publish it
cd "$(dirname "$0")"; h=$1; t=$2
until ssh -T -o ConnectTimeout=15 $h 'grep -q ALL_DONE ~/sp2/logs_v2/progress.log' 2>/dev/null; do sleep 120; done
SKIP_BASE=1 ./deploy_v2.sh $h $t 2>&1 | grep -v Warning
./publish_loop.sh $h $t
