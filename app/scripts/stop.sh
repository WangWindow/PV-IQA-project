#!/bin/bash
echo "Stopping PV-IQA..."
for port in 6005 6006; do
  for pid in $(lsof -ti:$port 2>/dev/null); do
    kill -9 $pid 2>/dev/null && echo "  Killed pid $pid on port $port"
  done
done
echo "Done."
