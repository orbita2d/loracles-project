#!/usr/bin/env bash
# On bramble: upload2.sh <pod> [name=butterflies|control]; every 3 min stage finished checkpoints on the pod and push them to
# orbita2d/nn-mats-miniproj-loras/sleeper-qwen3-14b/butterflies/ (upload-large-folder skips what is already up).
# Stops after the upload that follows ALL_DONE.
pod=$1; NAME=${2:-butterflies}; BOOK=$([ $NAME = butterflies ] && echo uploaded.txt || echo uploaded_$NAME.txt)
cd "$(dirname "$0")"
while true; do
  done_=$(ssh -T -o ConnectTimeout=15 $pod "grep -c ALL_DONE ~/sl/logs/progress.log" 2>/dev/null)
  ssh -T $pod "NAME=$NAME bash ~/sl/stage.sh" 2>/dev/null
  staged=$(ssh -T $pod "d=~/hfstage/sleeper-qwen3-14b/$NAME; ls \$d/checkpoints; [ -f \$d/adapter_model.safetensors ] && echo final" 2>/dev/null)
  ssh -T -o ServerAliveInterval=30 $pod "cd ~/hfstage && HF_TOKEN=\$(cat) HF_HUB_ENABLE_HF_TRANSFER=0 ~/venv/bin/hf upload-large-folder orbita2d/nn-mats-miniproj-loras . --repo-type model --num-workers 8 --no-bars > ~/sl/logs/upload.log 2>&1; echo UPLOAD_EXIT \$? >> ~/sl/logs/upload.log" < ~/.hf-token-lora 2>/dev/null
  res=$(ssh -T $pod 'tail -n 1 ~/sl/logs/upload.log' 2>/dev/null)
  [ "$res" = "UPLOAD_EXIT 0" ] && echo "$staged" > $BOOK
  echo "$(date -u +%H:%M) $res $(echo $staged | wc -w) items"
  [ "${done_:-0}" -ge 1 ] && break
  sleep 180
done
echo UPLOADER_DONE
