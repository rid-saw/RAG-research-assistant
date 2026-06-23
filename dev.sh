#!/usr/bin/env bash
# Run backend + frontend together. Ctrl-C kills both cleanly.

set -euo pipefail

cd "$(dirname "$0")"

# Free the ports if a previous run is still holding them.
for port in 8000 5173; do
  pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "freeing port $port (pids: $pids)"
    kill $pids 2>/dev/null || true
    sleep 0.4
  fi
done

# Kill all child processes on exit / interrupt.
trap 'kill 0' EXIT INT TERM

# ANSI colors for prefixes.
B='\033[38;5;130m' # backend - brown
F='\033[38;5;131m' # frontend - oxblood
R='\033[0m'

# Backend.
(
  cd backend
  uv run uvicorn app.main:app --reload --port 8000 2>&1 \
    | sed -u "s|^|$(printf "${B}[backend]${R}  ")|"
) &

# Give the backend a moment to bind before the frontend tries to reach it.
sleep 1

# Frontend.
(
  cd frontend
  npm run dev 2>&1 \
    | sed -u "s|^|$(printf "${F}[frontend]${R} ")|"
) &

wait
