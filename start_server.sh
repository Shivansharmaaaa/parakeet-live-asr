#!/bin/bash
# Start the web server for live transcription (using streaming mode like live_transcribe.py)

[ -f .env ] && set -a && . ./.env && set +a  # NVIDIA_API_KEY lives in .env (gitignored)
export PORT=5002

echo "Starting web server with streaming mode (like live_transcribe.py)..."
echo "Open http://localhost:5002 in your browser"
echo "Press Ctrl+C to stop"
echo ""

# Use app_streaming.py which uses the same logic as live_transcribe.py
python app_streaming.py
