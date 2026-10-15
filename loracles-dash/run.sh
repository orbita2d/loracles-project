#!/usr/bin/env bash
# Start the LoRAcles Lab dashboard + chat on http://bramble:8798/ (detached, logs to server.log).
cd "$(dirname "$0")"
pkill -f "loracles-dash/server.py" 2>/dev/null; sleep 1
nohup ./py.sh server.py > server.log 2>&1 &
echo "started pid $!; log: $(pwd)/server.log"
