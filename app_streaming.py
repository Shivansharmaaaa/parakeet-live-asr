#!/usr/bin/env python3
"""
Flask backend using the same streaming logic as live_transcribe.py
Uses streaming_response_generator with interim_results for real-time transcription.
"""

import os
import base64
import tempfile
import wave
import queue
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
    """Initialize ASR service - same as live_transcribe.py"""
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


def extract_pcm_from_wav(wav_bytes):
    """Extract PCM audio data from WAV file bytes."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = tmp.name
        
        try:
            with wave.open(tmp_path, 'rb') as wav_file:
                n_frames = wav_file.getnframes()
                pcm_data = wav_file.readframes(n_frames)
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                return pcm_data, sample_rate, channels
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        print(f"Error parsing WAV: {e}")
        if len(wav_bytes) >= 4 and wav_bytes[:4] == b'RIFF':
            data_pos = wav_bytes.find(b'data')
            if data_pos > 0:
                pcm_data = wav_bytes[data_pos + 8:]
                return pcm_data, 16000, 1
        return wav_bytes, 16000, 1


class AudioChunkIterator:
    """Iterator that yields audio chunks from a queue - mimics MicrophoneStream behavior."""
    def __init__(self, chunk_queue, sample_rate=16000, chunk_size=1600):
        self.chunk_queue = chunk_queue
        self.is_active = True
        self.timeout = 1.0
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size  # frames per chunk
    
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
            return chunk
        except queue.Empty:
            # Return empty chunk to keep stream alive
            return b''


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/api/stream/start', methods=['POST'])
def start_streaming():
    """Start a streaming transcription session."""
    import uuid
    session_id = str(uuid.uuid4())
    chunk_queue = queue.Queue()
    active_sessions[session_id] = {
        'queue': chunk_queue,
        'active': True,
        'sample_rate': 16000,
        'channels': 1
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
        
        # Update session with audio parameters
        session = active_sessions[session_id]
        session['queue'].put(pcm_data)
        session['sample_rate'] = sample_rate
        session['channels'] = channels
        
        return jsonify({'status': 'received', 'size': len(pcm_data)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/stream/<session_id>')
def stream_transcription(session_id):
    """Stream transcription results using Server-Sent Events - same logic as live_transcribe.py"""
    if session_id not in active_sessions:
        return jsonify({'error': 'Invalid session'}), 404
    
    session = active_sessions[session_id]
    chunk_queue = session['queue']
    
    def generate():
        try:
            # Send initial connection message
            yield f"data: {json.dumps({'status': 'connected'})}\n\n"
            
            # Setup ASR service - same as live_transcribe.py
            asr = setup_asr_service()
            
            # Get audio parameters
            sample_rate = session.get('sample_rate', 16000)
            channels = session.get('channels', 1)
            
            # Configure streaming recognition - EXACTLY like live_transcribe.py
            config = riva.client.StreamingRecognitionConfig(
                config=riva.client.RecognitionConfig(
                    encoding=riva.client.AudioEncoding.LINEAR_PCM,
                    language_code='en-US',
                    max_alternatives=1,
                    enable_automatic_punctuation=True,
                    sample_rate_hertz=sample_rate,
                    audio_channel_count=channels,
                ),
                interim_results=True,  # Show partial results as you speak - KEY FEATURE
            )
            
            print(f"Starting streaming with sample_rate={sample_rate}, channels={channels}, interim_results=True")
            
            # Create audio chunk iterator
            audio_iterator = AudioChunkIterator(chunk_queue, sample_rate)
            
            # Start streaming - same as live_transcribe.py
            responses = asr.streaming_response_generator(
                audio_chunks=audio_iterator,
                streaming_config=config
            )
            
            print("Streaming started, processing responses...")
            
            # Process responses - extract interim and final results like print_streaming does
            for response in responses:
                if not session.get('active', True):
                    break
                
                if response.results:
                    for result in response.results:
                        if result.alternatives:
                            transcript = result.alternatives[0].transcript
                            confidence = result.alternatives[0].confidence
                            is_final = result.is_final_result
                            
                            # Send both interim and final results
                            data = json.dumps({
                                'transcript': transcript,
                                'confidence': confidence,
                                'is_final': is_final,
                                'is_interim': not is_final
                            })
                            yield f"data: {data}\n\n"
                            
                            if is_final:
                                print(f"Final: '{transcript}' (conf={confidence:.2f})")
                            else:
                                print(f"Interim: '{transcript}' (conf={confidence:.2f})")
                                
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_msg = f"Error in streaming: {str(e)}"
            print(error_msg)
            error_data = json.dumps({'error': error_msg})
            yield f"data: {error_data}\n\n"
        finally:
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


if __name__ == '__main__':
    print("=" * 60)
    print("🎤 Live Transcription Web Server (Streaming Mode)")
    print("=" * 60)
    print(f"API Key: {'✅ Set' if API_KEY else '❌ Not set'}")
    print(f"Server: {SERVER}")
    print(f"Model: parakeet-ctc-1.1b-asr")
    print(f"Mode: Streaming with interim_results=True")
    port = int(os.getenv('PORT', 5002))
    print(f"\n🌐 Open http://localhost:{port} in your browser")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=port, threaded=True)
