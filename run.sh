#!/bin/bash
cd "$(dirname "$0")"
while true; do
    echo ""
    echo "[Supervisor] $(date '+%H:%M:%S') - Starting rtsp_ai_transcoder"
    ./rtsp_ai_transcoder \
        rtsp://127.0.0.1:8554/live/test \
        rtmp://127.0.0.1:1935/live/ai
    echo "[Supervisor] $(date '+%H:%M:%S') - Exited (code $?), restart in 3s"
    sleep 3
done
