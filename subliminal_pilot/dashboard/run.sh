#!/usr/bin/env bash
# Start the dashboard detached; serves http://bramble:8797/
# Stop: pkill -f '^python3 /home/claude/workspace/subliminal_pilot/dashboard/server.py'
cd "$(dirname "$0")"
setsid nohup nix shell nixpkgs#python3 --command python3 "$PWD/server.py" > server.log 2>&1 < /dev/null &
echo "started; log: $PWD/server.log"
