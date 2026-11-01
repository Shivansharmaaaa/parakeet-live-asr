#!/bin/bash
# Quick script for live transcription

[ -f .env ] && set -a && . ./.env && set +a  # NVIDIA_API_KEY lives in .env (gitignored)
python live_transcribe.py
