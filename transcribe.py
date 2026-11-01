#!/usr/bin/env python3
"""
NVIDIA Parakeet CTC 1.1B ASR Transcription Script
Transcribes audio files using NVIDIA's parakeet-ctc-1.1b-asr model via NIM API.
"""

import argparse
import sys
import os
from pathlib import Path

try:
    import riva.client
    from riva.client.argparse_utils import add_connection_argparse_parameters
    from riva.client.argparse_utils import add_asr_argparse_parameters
except ImportError:
    print("Error: nvidia-riva-client is not installed.")
    print("Please install it using: pip install -r requirements.txt")
    sys.exit(1)


def transcribe_audio(
    audio_file: str,
    api_key: str,
    language_code: str = "en-US",
    server: str = "grpc.nvcf.nvidia.com:443",
    function_id: str = "1598d209-5e27-4d3c-8079-4751568b1081",
    use_ssl: bool = True
):
    """
    Transcribe an audio file using NVIDIA Parakeet CTC 1.1B ASR model.
    
    Args:
        audio_file: Path to the audio file (WAV, OGG, OPUS format, 16-bit Mono)
        api_key: NVIDIA API key
        language_code: Language code (default: en-US)
        server: gRPC server address
        function_id: Function ID for the parakeet-ctc-1.1b-asr model
        use_ssl: Whether to use SSL/TLS
    """
    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)
    
    # Create auth metadata
    auth_metadata = [("authorization", f"Bearer {api_key}")]
    function_metadata = [("function-id", function_id)]
    metadata = auth_metadata + function_metadata
    
    # Create gRPC channel
    if use_ssl:
        channel = riva.client.create_grpc_channel(
            server,
            riva.client.Auth.get_ssl_credentials()
        )
    else:
        channel = riva.client.create_grpc_channel(server)
    
    # Create ASR service
    asr_service = riva.client.ASRService(channel, metadata=metadata)
    
    # Transcribe the audio file
    print(f"Transcribing audio file: {audio_file}")
    print("This may take a moment...\n")
    
    try:
        response = asr_service.offline_recognize(
            audio_file=audio_file,
            language_code=language_code
        )
        
        # Extract transcription
        if response.results:
            transcript = response.results[0].alternatives[0].transcript
            confidence = response.results[0].alternatives[0].confidence
            
            print("=" * 60)
            print("TRANSCRIPTION RESULT:")
            print("=" * 60)
            print(f"Text: {transcript}")
            print(f"Confidence: {confidence:.4f}")
            print("=" * 60)
            
            return transcript
        else:
            print("No transcription result returned.")
            return None
            
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio using NVIDIA Parakeet CTC 1.1B ASR model"
    )
    parser.add_argument(
        "--audio-file",
        required=True,
        help="Path to audio file (WAV, OGG, OPUS format, 16-bit Mono)"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="NVIDIA API key (or set NVIDIA_API_KEY environment variable)"
    )
    parser.add_argument(
        "--language-code",
        default="en-US",
        help="Language code (default: en-US)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file to save transcription (optional)"
    )
    
    args = parser.parse_args()
    
    # Get API key from argument or environment variable
    api_key = args.api_key or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("Error: API key not provided.")
        print("Please provide it via --api-key argument or NVIDIA_API_KEY environment variable.")
        sys.exit(1)
    
    # Transcribe audio
    transcript = transcribe_audio(
        audio_file=args.audio_file,
        api_key=api_key,
        language_code=args.language_code
    )
    
    # Save to file if requested
    if args.output and transcript:
        with open(args.output, 'w') as f:
            f.write(transcript)
        print(f"\nTranscription saved to: {args.output}")


if __name__ == "__main__":
    main()
