#!/usr/bin/env bash
# On bramble: publish_loop.sh <pod> <teachers...>; for each, wait for "DONE <t>" on <pod>, then stage, upload to
# orbita2d/nn-mats-miniproj-loras/subliminal-qwen3-14b-v2/<t>/ and free the pod's update snapshots.
cd "$(dirname "$0")"
pod=$1; shift
for t in "$@"; do
  until ssh -T -o ConnectTimeout=15 $pod "grep -q 'DONE $t\$' ~/sp2/logs_v2/progress.log" 2>/dev/null; do sleep 120; done
  ssh -T $pod "bash ~/sp2/stage_upload.sh $t" 2>/dev/null
  ssh -T -o ServerAliveInterval=30 $pod "cd ~/hfstage/$t && HF_TOKEN=\$(cat) HF_HUB_ENABLE_HF_TRANSFER=0 ~/venv/bin/hf upload-large-folder orbita2d/nn-mats-miniproj-loras . --repo-type model --num-workers 8 --no-bars > ~/sp2/logs_v2/upload_$t.log 2>&1; rc=\$?; echo UPLOAD_EXIT \$rc >> ~/sp2/logs_v2/upload_$t.log; [ \$rc = 0 ] && rm -rf ~/hfstage/$t ~/sp2/loras_v2/$t/updates" < ~/.hf-token-lora 2>/dev/null
  echo "$(date -u +%H:%M) $t $(ssh -T $pod "tail -n 1 ~/sp2/logs_v2/upload_$t.log" 2>/dev/null)"
done
