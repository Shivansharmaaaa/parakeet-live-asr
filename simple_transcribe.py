#!/usr/bin/env python3
"""
Simple wrapper for NVIDIA Parakeet CTC 1.1B ASR using the command-line tool.
This script uses the official Riva Python client scripts.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import riva.client
    except ImportError:
        print("Error: nvidia-riva-client is not installed.")
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed. Please run the script again.")
        sys.exit(0)


def download_clients():
    """Download Riva Python clients if not already present."""
    clients_dir = Path("python-clients")
    if not clients_dir.exists():
        print("Downloading Riva Python clients...")
        subprocess.check_call([
            "git", "clone", "https://github.com/nvidia-riva/python-clients.git"
        ])
        print("Python clients downloaded.")
    else:
        print("Python clients already present.")


def transcribe(audio_file: str, api_key: str):
    """
    Transcribe audio file using NVIDIA Parakeet CTC 1.1B ASR.
    
    Args:
        audio_file: Path to audio file
        api_key: NVIDIA API key
    """
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    # Check if python-clients directory exists
    clients_dir = Path("python-clients")
    if not clients_dir.exists():
        print("Error: python-clients directory not found.")
        print("Please run this script once to download the clients.")
        sys.exit(1)
    
    script_path = clients_dir / "scripts" / "asr" / "transcribe_file.py"
    if not script_path.exists():
        print(f"Error: Transcription script not found at {script_path}")
        sys.exit(1)
    
    # Build command
    cmd = [
        sys.executable,
        str(script_path),
        "--server", "grpc.nvcf.nvidia.com:443",
        "--use-ssl",
        "--metadata", "function-id", "1598d209-5e27-4d3c-8079-4751568b1081",
        "--metadata", "authorization", f"Bearer {api_key}",
        "--language-code", "en-US",
        "--input-file", audio_file
    ]
    
    print(f"Transcribing: {audio_file}")
    print("=" * 60)
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Simple transcription using NVIDIA Parakeet CTC 1.1B ASR"
    )
    parser.add_argument(
        "audio_file",
        help="Path to audio file (WAV, OGG, OPUS format, 16-bit Mono)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="NVIDIA API key (or set NVIDIA_API_KEY environment variable)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("Error: API key not provided.")
        print("Please provide it via --api-key argument or NVIDIA_API_KEY environment variable.")
        sys.exit(1)
    
    # Check dependencies
    check_dependencies()
    
    # Download clients if needed
    download_clients()
    
    # Transcribe
    transcribe(args.audio_file, api_key)


if __name__ == "__main__":
    main()
