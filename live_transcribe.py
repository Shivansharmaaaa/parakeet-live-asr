#!/usr/bin/env python3
"""
Live transcription from microphone using NVIDIA Parakeet CTC 1.1B ASR model.
Records audio from your microphone and shows real-time transcription.
"""

import argparse
import os
import sys
import signal

try:
    import riva.client
    from riva.client.argparse_utils import add_asr_config_argparse_parameters, add_connection_argparse_parameters
except ImportError:
    print("Error: nvidia-riva-client is not installed.")
    print("Please install it using: pip install -r requirements.txt")
    sys.exit(1)

try:
    import riva.client.audio_io
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")
    print("Please install pyaudio: pip install pyaudio")
    print("Or on macOS: brew install portaudio && pip install pyaudio")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    default_device_info = riva.client.audio_io.get_default_input_device_info()
    default_device_index = None if default_device_info is None else default_device_info['index']
    
    parser = argparse.ArgumentParser(
        description="Live transcription from microphone using NVIDIA Parakeet CTC 1.1B ASR",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=default_device_index,
        help="Input audio device index to use. Use --list-devices to see available devices."
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available input audio device indices and exit."
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="NVIDIA API key (or set NVIDIA_API_KEY environment variable)"
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=int,
        default=16000,
        help="Sample rate in Hz for audio stream (default: 16000)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1600,
        help="Number of frames per audio chunk sent to server (default: 1600)"
    )
    parser.add_argument(
        "--language-code",
        default="en-US",
        help="Language code (default: en-US)"
    )
    parser.add_argument(
        "--automatic-punctuation",
        action="store_true",
        help="Enable automatic punctuation"
    )
    
    args = parser.parse_args()
    return args


def main() -> None:
    """Main function for live transcription."""
    args = parse_args()
    
    # List devices if requested
    if args.list_devices:
        print("\nAvailable audio input devices:")
        print("=" * 60)
        riva.client.audio_io.list_input_devices()
        return
    
    # Get API key
    api_key = args.api_key or os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("Error: API key not provided.")
        print("Please provide it via --api-key argument or NVIDIA_API_KEY environment variable.")
        sys.exit(1)
    
    # NVIDIA NIM API configuration
    server = "grpc.nvcf.nvidia.com:443"
    function_id = "1598d209-5e27-4d3c-8079-4751568b1081"
    
    # Create metadata for authentication
    metadata = [
        ("authorization", f"Bearer {api_key}"),
        ("function-id", function_id)
    ]
    
    # Create auth object
    auth = riva.client.Auth(
        use_ssl=True,
        uri=server,
        metadata_args=metadata,
    )
    
    # Create ASR service
    asr_service = riva.client.ASRService(auth)
    
    # Configure streaming recognition
    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            language_code=args.language_code,
            max_alternatives=1,
            enable_automatic_punctuation=args.automatic_punctuation,
            sample_rate_hertz=args.sample_rate_hz,
            audio_channel_count=1,
        ),
        interim_results=True,  # Show partial results as you speak
    )
    
    print("\n" + "=" * 60)
    print("🎤 LIVE TRANSCRIPTION - NVIDIA Parakeet CTC 1.1B ASR")
    print("=" * 60)
    print(f"Server: {server}")
    print(f"Sample Rate: {args.sample_rate_hz} Hz")
    print(f"Language: {args.language_code}")
    print("\n💡 Start speaking! Press Ctrl+C to stop.")
    print("=" * 60 + "\n")
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n🛑 Stopping transcription...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Start microphone stream and transcribe
        with riva.client.audio_io.MicrophoneStream(
            args.sample_rate_hz,
            args.chunk_size,
            device=args.input_device,
        ) as audio_chunk_iterator:
            # Print streaming results with intermediate (partial) results
            riva.client.print_streaming(
                responses=asr_service.streaming_response_generator(
                    audio_chunks=audio_chunk_iterator,
                    streaming_config=config,
                ),
                show_intermediate=True,  # Show partial results as you speak
            )
    except KeyboardInterrupt:
        print("\n\n🛑 Transcription stopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
