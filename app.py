#!/usr/bin/env python3
"""
Flask backend for live transcription web interface.
Handles audio streaming and real-time transcription using NVIDIA Parakeet CTC 1.1B ASR.
"""

import os
import io
import base64
import subprocess
import tempfile
import wave
import queue
import threading
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import riva.client
import json

app = Flask(__name__)
CORS(app)

# Configuration
API_KEY = os.getenv("NVIDIA_API_KEY", "")
SERVER = "grpc.nvcf.nvidia.com:443"
FUNCTION_ID = "1598d209-5e27-4d3c-8079-4751568b1081"

# Global ASR service
asr_service = None

# Active streaming sessions
active_sessions = {}


def setup_asr_service():
    """Initialize ASR service."""
    global asr_service
    if asr_service is None:
        metadata = [
            ("authorization", f"Bearer {API_KEY}"),
            ("function-id", FUNCTION_ID)
        ]
        auth = riva.client.Auth(
            use_ssl=True,
            uri=SERVER,
            metadata_args=metadata,
        )
        asr_service = riva.client.ASRService(auth)
    return asr_service


def convert_webm_to_wav(webm_data, output_path):
    """Convert WebM audio to WAV format using ffmpeg."""
    tmp_webm_path = None
    try:
        # Write WebM data to temp file
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as tmp_webm:
            tmp_webm.write(webm_data)
            tmp_webm_path = tmp_webm.name
        
        # Convert to WAV using ffmpeg
        cmd = [
            'ffmpeg', '-y', '-i', tmp_webm_path,
            '-ar', '16000',  # Sample rate 16kHz
            '-ac', '1',      # Mono
            '-f', 'wav',     # WAV format
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            raise Exception(f"FFmpeg conversion failed: {result.stderr}")
        
        return True
    except FileNotFoundError:
        # FFmpeg not available
        return False
    except Exception as e:
        raise e
    finally:
        # Clean up temp WebM file
        if tmp_webm_path and os.path.exists(tmp_webm_path):
            os.unlink(tmp_webm_path)


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


def extract_pcm_from_wav(wav_bytes):
    """Extract PCM audio data from WAV file bytes."""
    try:
        # Try to parse as WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        
        try:
            with wave.open(tmp_path, 'rb') as wav_file:
                n_frames = wav_file.getnframes()
                pcm_data = wav_file.readframes(n_frames)
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                
                # Validate audio parameters
                if sample_rate not in [8000, 16000, 22050, 44100, 48000]:
                    print(f"Warning: Unusual sample rate {sample_rate}Hz, using 16000Hz")
                    sample_rate = 16000
                
                if channels not in [1, 2]:
                    print(f"Warning: Unusual channel count {channels}, using mono")
                    channels = 1
                
                print(f"WAV file: {n_frames} frames, {sample_rate}Hz, {channels}ch, {sample_width} bytes/sample")
                print(f"PCM data size: {len(pcm_data)} bytes")
                
                # Ensure minimum audio length (at least 0.5 seconds)
                min_bytes = sample_rate * sample_width * channels * 0.5
                if len(pcm_data) < min_bytes:
                    print(f"Warning: Audio too short ({len(pcm_data)} bytes < {min_bytes} bytes)")
                
                return pcm_data, sample_rate, channels
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        print(f"Error parsing WAV: {e}")
        # If not a valid WAV, try to extract PCM manually
        if len(wav_bytes) >= 4 and wav_bytes[:4] == b'RIFF' and len(wav_bytes) > 44:
            # Skip WAV header (44 bytes typically, but could be more)
            # Find 'data' chunk
            data_pos = wav_bytes.find(b'data')
            if data_pos > 0:
                pcm_start = data_pos + 8  # 'data' (4) + size (4)
                pcm_data = wav_bytes[pcm_start:]
                print(f"Extracted PCM from WAV header: {len(pcm_data)} bytes")
                return pcm_data, 16000, 1
        # Last resort: assume it's already PCM (16-bit, mono, 16kHz)
        print(f"Assuming raw PCM data: {len(wav_bytes)} bytes")
        # Ensure minimum length
        if len(wav_bytes) < 16000:  # Less than 0.5 seconds at 16kHz
            print(f"Warning: Audio chunk too short ({len(wav_bytes)} bytes)")
        return wav_bytes, 16000, 1


class AudioChunkIterator:
    """Iterator that yields audio chunks from a queue."""
    def __init__(self, chunk_queue):
        self.chunk_queue = chunk_queue
        self.is_active = True
        self.timeout = 3.0  # Longer timeout
        self.empty_count = 0
        self.max_empty = 20  # Allow more empty reads before stopping
        self.chunks_sent = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.is_active:
            raise StopIteration
        
        try:
            chunk = self.chunk_queue.get(timeout=self.timeout)
            if chunk is None:  # Sentinel value to stop
                self.is_active = False
                raise StopIteration
            self.empty_count = 0  # Reset empty count
            self.chunks_sent += 1
            print(f"Yielding chunk #{self.chunks_sent}: {len(chunk)} bytes")
            return chunk
        except queue.Empty:
            self.empty_count += 1
            if self.empty_count >= self.max_empty:
                print(f"No chunks received for {self.max_empty * self.timeout}s, stopping iterator")
                self.is_active = False
                raise StopIteration
            # Return empty chunk to keep stream alive (but only occasionally)
            if self.empty_count % 5 == 0:  # Every 5th empty read
                print(f"Waiting for chunks... (empty count: {self.empty_count})")
            return b''


@app.route('/api/stream/start', methods=['POST'])
def start_streaming():
    """Start a streaming transcription session."""
    import uuid
    session_id = str(uuid.uuid4())
    chunk_queue = queue.Queue()
    active_sessions[session_id] = {
        'queue': chunk_queue,
        'active': True
    }
    return jsonify({'session_id': session_id})


@app.route('/api/stream/chunk', methods=['POST'])
def stream_chunk():
    """Receive audio chunk for streaming transcription."""
    try:
        data = request.json
        session_id = data.get('session_id')
        audio_data = data.get('audio')
        
        if not session_id or session_id not in active_sessions:
            return jsonify({'error': 'Invalid session'}), 400
        
        if not audio_data:
            return jsonify({'error': 'No audio data'}), 400
        
        # Decode base64 audio
        header, encoded = audio_data.split(',', 1)
        audio_bytes = base64.b64decode(encoded)
        
        # Extract PCM data
        pcm_data, sample_rate, channels = extract_pcm_from_wav(audio_bytes)
        
        # Add to session queue
        session = active_sessions[session_id]
        session['queue'].put(pcm_data)
        session['sample_rate'] = sample_rate
        session['channels'] = channels
        
        print(f"Received chunk: {len(pcm_data)} bytes for session {session_id[:8]}...")
        
        return jsonify({'status': 'received', 'size': len(pcm_data)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/stream/<session_id>')
def stream_transcription(session_id):
    """Stream transcription results using Server-Sent Events."""
    if session_id not in active_sessions:
        return jsonify({'error': 'Invalid session'}), 404
    
    session = active_sessions[session_id]
    chunk_queue = session['queue']
    
    def generate():
        try:
            # Send initial connection message
            yield f"data: {json.dumps({'status': 'connected'})}\n\n"
            
            # Wait for first chunks to arrive (with timeout)
            import time
            print(f"Waiting for audio chunks for session {session_id[:8]}...")
            
            # Wait up to 5 seconds for first chunk
            wait_time = 0
            max_wait = 5.0
            while chunk_queue.empty() and wait_time < max_wait:
                time.sleep(0.3)
                wait_time += 0.3
                if not session.get('active', True):
                    print(f"Session stopped while waiting")
                    return
                # Send keepalive
                yield f"data: {json.dumps({'status': 'waiting', 'wait_time': wait_time})}\n\n"
            
            # Check if we have chunks
            if chunk_queue.empty():
                error_msg = f"No audio chunks received after {wait_time:.1f}s"
                print(error_msg)
                yield f"data: {json.dumps({'error': error_msg})}\n\n"
                # Continue anyway - chunks might arrive later
            else:
                print(f"Received first chunk after {wait_time:.1f}s")
            
            # Setup ASR service
            asr = setup_asr_service()
            print(f"ASR service setup for session {session_id[:8]}...")
            
            # Get sample rate and channels from first chunk info
            sample_rate = session.get('sample_rate', 16000)
            channels = session.get('channels', 1)
            
            # Create streaming config
            streaming_config = riva.client.StreamingRecognitionConfig(
                config=riva.client.RecognitionConfig(
                    encoding=riva.client.AudioEncoding.LINEAR_PCM,
                    language_code='en-US',
                    max_alternatives=1,
                    enable_automatic_punctuation=True,
                    sample_rate_hertz=sample_rate,
                    audio_channel_count=channels,
                ),
                interim_results=True,  # Get partial results
            )
            
            print(f"Starting streaming with sample_rate={sample_rate}, channels={channels}")
            
            # Create audio chunk iterator
            audio_iterator = AudioChunkIterator(chunk_queue)
            
            # Start streaming
            responses = asr.streaming_response_generator(
                audio_chunks=audio_iterator,
                streaming_config=streaming_config
            )
            
            print(f"Streaming started, waiting for responses...")
            
            # Stream results
            response_count = 0
            for response in responses:
                if not session.get('active', True):
                    print(f"Session {session_id[:8]} stopped")
                    break
                
                response_count += 1
                print(f"Received response #{response_count} for session {session_id[:8]}")
                
                if response.results:
                    for result in response.results:
                        if result.alternatives:
                            transcript = result.alternatives[0].transcript
                            confidence = result.alternatives[0].confidence
                            is_final = result.is_final_result
                            
                            print(f"Transcription: '{transcript}' (final={is_final}, conf={confidence})")
                            
                            # Send as SSE
                            data = json.dumps({
                                'transcript': transcript,
                                'confidence': confidence,
                                'is_final': is_final
                            })
                            yield f"data: {data}\n\n"
                else:
                    print(f"No results in response #{response_count}")
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Error in streaming: {str(e)}"
            print(error_msg)
            error_data = json.dumps({'error': error_msg})
            yield f"data: {error_data}\n\n"
        finally:
            # Cleanup
            print(f"Cleaning up session {session_id[:8]}...")
            if session_id in active_sessions:
                del active_sessions[session_id]
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


@app.route('/api/stream/stop/<session_id>', methods=['POST'])
def stop_streaming(session_id):
    """Stop a streaming session."""
    if session_id in active_sessions:
        session = active_sessions[session_id]
        session['active'] = False
        session['queue'].put(None)  # Sentinel to stop iterator
        return jsonify({'status': 'stopped'})
    return jsonify({'error': 'Session not found'}), 404


@app.route('/api/transcribe', methods=['POST'])
def transcribe_audio():
    """Fallback: Transcribe single audio chunk (non-streaming)."""
    try:
        data = request.json
        audio_data = data.get('audio')
        
        if not audio_data:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Decode base64 audio data
        header, encoded = audio_data.split(',', 1)
        audio_bytes = base64.b64decode(encoded)
        
        # Extract PCM data and audio parameters
        pcm_data, sample_rate, channels = extract_pcm_from_wav(audio_bytes)
        
        if len(pcm_data) == 0:
            print("Empty audio data")
            return jsonify({'transcript': '', 'confidence': 0.0})
        
        # Ensure minimum audio length (at least 1 second for better accuracy)
        min_bytes = int(sample_rate * 2 * channels * 1.0)  # 1 second, 16-bit (2 bytes)
        if len(pcm_data) < min_bytes:
            print(f"Audio too short: {len(pcm_data)} bytes < {min_bytes} bytes (min 1.0s)")
            return jsonify({'transcript': '', 'confidence': 0.0})
        
        # Warn if audio is very short (but still process)
        if len(pcm_data) < int(sample_rate * 2 * channels * 2.0):
            print(f"Warning: Audio is short ({len(pcm_data)} bytes), may affect accuracy")
        
        print(f"Transcribing chunk: {len(pcm_data)} bytes PCM, {sample_rate}Hz, {channels} channels")
        
        # Create recognition config with explicit encoding
        # Note: offline_recognize expects raw PCM bytes, not WAV file
        config = riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            language_code='en-US',
            max_alternatives=1,
            enable_automatic_punctuation=True,
            sample_rate_hertz=sample_rate,
            audio_channel_count=channels,
        )
        
        # Transcribe using ASR service
        # offline_recognize expects raw PCM audio bytes (from readframes)
        asr = setup_asr_service()
        response = asr.offline_recognize(pcm_data, config)
        
        if response.results and len(response.results) > 0:
            transcript = response.results[0].alternatives[0].transcript
            confidence = response.results[0].alternatives[0].confidence
            print(f"Transcription result: '{transcript}' (conf={confidence})")
            return jsonify({
                'transcript': transcript,
                'confidence': confidence
            })
        else:
            print("No transcription results")
            return jsonify({'transcript': '', 'confidence': 0.0})
            
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'api_key_set': bool(API_KEY),
        'server': SERVER
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🎤 Live Transcription Web Server")
    print("=" * 60)
    print(f"API Key: {'✅ Set' if API_KEY else '❌ Not set'}")
    print(f"Server: {SERVER}")
    print(f"Model: parakeet-ctc-1.1b-asr")
    port = int(os.getenv('PORT', 5001))
    print(f"\n🌐 Open http://localhost:{port} in your browser")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=port, threaded=True)
