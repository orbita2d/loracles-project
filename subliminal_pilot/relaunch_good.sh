#!/usr/bin/env bash
# relaunch_good.sh <teacher>: launch subl2-<t> in the US until the host's driver is >= 580 (cu130 torch), then bring it up
cd "$(dirname "$0")"; t=$1; h=subl2-$t
for i in 1 2 3 4 5 6; do
  timeout 900 uvx --from 'skypilot[runpod]' sky down -y $h > /dev/null 2>&1
  timeout 3600 uvx --from 'skypilot[runpod]' sky launch -c $h --region US -y --detach-run sky.yaml > runs_v2/launch/${t}5.log 2>&1
  d=$(ssh -T -o ConnectTimeout=20 $h 'nvidia-smi --query-gpu=driver_version --format=csv,noheader' 2>/dev/null | grep -v Warning | cut -d. -f1)
  echo "try $i: driver ${d:-none}"
  [ -n "$d" ] && [ "$d" -ge 580 ] && { LOGSUF=5 ./bringup.sh $t; exit; }
done
echo "$t: no good host after 6 tries"
