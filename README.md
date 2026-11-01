# NVIDIA Parakeet CTC 1.1B ASR Transcription

This project provides scripts to transcribe audio files, perform **live transcription from your microphone**, and includes a **modern web interface** for real-time transcription using NVIDIA's Parakeet CTC 1.1B ASR model via the NVIDIA NIM API.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   On macOS, if PyAudio installation fails, install portaudio first:
   ```bash
   brew install portaudio
   pip install pyaudio
   ```
   
   For web interface audio conversion (optional but recommended):
   ```bash
   brew install ffmpeg  # macOS
   # or
   sudo apt-get install ffmpeg  # Linux
   ```

2. **Set your API key:**
   ```bash
   export NVIDIA_API_KEY="nvapi-your-key-here"
   ```
   
   Or pass it directly when running the script.

## Usage

### 🌐 Web Interface (Recommended)

**Start the web server:**
```bash
./start_server.sh
```

Or:
```bash
export NVIDIA_API_KEY="nvapi-your-key-here"
python app.py
```

Then open **http://localhost:5000** in your browser for a beautiful web interface with:
- One-click recording
- Real-time transcription display
- Confidence scores
- Modern, responsive UI

### 🎤 Live Transcription (Command Line)

**Start live transcription:**
```bash
./live.sh
```

Or directly:
```bash
export NVIDIA_API_KEY="nvapi-your-key-here"
python live_transcribe.py
```

**List available microphones:**
```bash
python live_transcribe.py --list-devices
```

**With automatic punctuation:**
```bash
python live_transcribe.py --automatic-punctuation
```

**Press Ctrl+C to stop** the live transcription.

### 📁 Audio File Transcription

**Option 1: Simple Script (Recommended)**
```bash
python simple_transcribe.py <path_to_audio_file> --api-key <your_api_key>
```

Or with environment variable:
```bash
export NVIDIA_API_KEY="nvapi-your-key-here"
python simple_transcribe.py audio.wav
```

**Option 2: Custom Script**
```bash
python transcribe.py --audio-file <path_to_audio_file> --api-key <your_api_key>
```

**Save transcription to file:**
```bash
python transcribe.py --audio-file audio.wav --output transcript.txt
```

**Quick script:**
```bash
./run.sh audio.wav          # Transcribe file
./run.sh --live              # Live transcription
```

## Audio Requirements

### For Live Transcription:
- Microphone access (will request permission on first run)
- 16 kHz sample rate (default)
- Mono audio

### For Audio Files:
- Format: WAV, OGG, or OPUS
- Audio: 16-bit Mono
- Language: English (en-US)

## Model Information

- **Model**: parakeet-ctc-1.1b-asr
- **Provider**: NVIDIA NIM (NVIDIA Inference Microservice)
- **Function ID**: 1598d209-5e27-4d3c-8079-4751568b1081
- **Server**: grpc.nvcf.nvidia.com:443
- **Features**: Streaming/real-time transcription with interim results

## Notes

- The first run will automatically download the Riva Python clients repository
- Make sure you have a stable internet connection
- Live transcription shows partial results as you speak (interim results)
- Press Ctrl+C to stop live transcription
- On macOS, you may need to grant microphone permissions in System Settings

## NVIDIA Riva clients

The scripts use NVIDIA's Riva Python clients. Clone them next to the scripts:

```bash
git clone https://github.com/nvidia-riva/python-clients.git
```
