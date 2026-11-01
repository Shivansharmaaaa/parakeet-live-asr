#!/bin/bash
# Quick transcription script for NVIDIA Parakeet CTC 1.1B ASR

# Set API key
[ -f .env ] && set -a && . ./.env && set +a  # NVIDIA_API_KEY lives in .env (gitignored)

# Check if audio file is provided
if [ -z "$1" ]; then
    echo "Usage:"
    echo "  ./run.sh <audio_file>     - Transcribe audio file"
    echo "  ./run.sh --live            - Live transcription from microphone"
    echo ""
    echo "Examples:"
    echo "  ./run.sh audio.wav"
    echo "  ./run.sh --live"
    exit 1
fi

# Check for live mode
if [ "$1" == "--live" ]; then
    python live_transcribe.py
else
    # Run file transcription
    python simple_transcribe.py "$1"
fi
