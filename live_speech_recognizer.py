#!/usr/bin/env python3
"""
Live speech recognition using NVIDIA Parakeet CTC model with local inference.
Uses PyAudio for real-time audio capture and HuggingFace transformers for transcription.
"""

import pyaudio
import numpy as np
from transformers import AutoModelForCTC, AutoProcessor
import torch
import threading
import queue
import time


class LiveSpeechRecognizer:
    def __init__(self, model_name="nvidia/parakeet-ctc-1.1b", chunk_size=1024, sample_rate=16000):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        
        # Load model and processor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model '{model_name}' on device: {self.device}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Load model with appropriate dtype
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCTC.from_pretrained(
            model_name, 
            dtype=dtype
        )
        # Move model to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Audio buffer and transcription queue
        self.audio_buffer = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.is_recording = False
        
        # PyAudio setup
        self.pyaudio_instance = pyaudio.PyAudio()
        self.stream = None
        
    def start_listening(self):
        """Start real-time audio capture"""
        self.is_recording = True
        self.stream = self.pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()
        
        # Start transcription thread
        self.transcription_thread = threading.Thread(target=self.process_audio, daemon=True)
        self.transcription_thread.start()
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Callback for audio stream"""
        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_buffer.put(audio_data)
        return (None, pyaudio.paContinue)
    
    def process_audio(self):
        """Process audio chunks for transcription"""
        audio_accumulator = []
        window_size = self.sample_rate * 2  # 2-second windows
        
        while self.is_recording:
            try:
                # Collect audio chunks
                chunk = self.audio_buffer.get(timeout=0.1)
                audio_accumulator.extend(chunk)
                
                # Process when we have enough audio
                if len(audio_accumulator) >= window_size:
                    # Take the last window_size samples
                    audio_window = np.array(audio_accumulator[-window_size:])
                    
                    # Transcribe
                    transcription = self.transcribe_audio(audio_window)
                    if transcription.strip():
                        self.transcription_queue.put(transcription)
                        print(f"Live: {transcription}")
                    
                    # Keep some overlap for context
                    audio_accumulator = audio_accumulator[window_size//2:]
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
                continue
    
    def transcribe_audio(self, audio_data):
        """Transcribe audio using Parakeet"""
        try:
            # Process audio
            inputs = self.processor(
                audio_data, 
                sampling_rate=self.sample_rate, 
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                outputs = self.model.generate(**inputs)
            
            # Decode
            transcription = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            return transcription
        except Exception as e:
            print(f"Error in transcription: {e}")
            return ""
    
    def stop_listening(self):
        """Stop real-time audio capture"""
        self.is_recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.pyaudio_instance.terminate()
        
        if hasattr(self, 'transcription_thread'):
            self.transcription_thread.join(timeout=2.0)
    
    def get_latest_transcription(self):
        """Get the latest transcription"""
        try:
            return self.transcription_queue.get_nowait()
        except queue.Empty:
            return None


# Usage example
if __name__ == "__main__":
    recognizer = LiveSpeechRecognizer()
    print("Starting live speech recognition...")
    print("Press Ctrl+C to stop.\n")
    recognizer.start_listening()
    
    try:
        while True:
            time.sleep(1)
            latest = recognizer.get_latest_transcription()
            if latest:
                print(f"Transcribed: {latest}")
    except KeyboardInterrupt:
        print("\nStopping...")
        recognizer.stop_listening()
