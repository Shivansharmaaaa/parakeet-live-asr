#!/usr/bin/env python3
"""
Local live transcription using NVIDIA Parakeet CTC model with WebRTC VAD.
This is an offline solution that runs entirely on your machine.
"""

import pyaudio
import numpy as np
from transformers import AutoModelForCTC, AutoProcessor
import torch
import webrtcvad
import threading
import queue
import collections
import time

class OptimizedLiveRecognizer:
    def __init__(self, model_name="nvidia/parakeet-ctc-1.1b"):
        self.sample_rate = 16000
        self.chunk_size = 480  # 30ms chunks for VAD
        self.frame_duration = 30  # ms
        
        # VAD setup
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3
        self.speech_frames = collections.deque(maxlen=100)  # 3 seconds of audio
        
        # Model setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model {model_name} on {self.device}...")
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        self.model.eval()
        
        # Optimize model for inference
        if self.device == "cuda":
            try:
                # Use torch.compile for faster inference (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as e:
                print(f"Warning: Could not compile model: {e}")
        
        # State management
        self.is_speaking = False
        self.speech_buffer = []
        self.min_speech_duration = 0.5  # seconds
        self.silence_threshold = 1.0  # seconds
        
    def process_audio_stream(self):
        """Process audio stream with VAD"""
        pyaudio_instance = pyaudio.PyAudio()
        
        try:
            stream = pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            stream.start_stream()
            print("Listening... Speak now!")
            
            while stream.is_active():
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            stream.stop_stream()
            stream.close()
            pyaudio_instance.terminate()
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Process audio with VAD"""
        # Convert to the format VAD expects
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        
        # Check if this frame contains speech
        is_speech = self.vad.is_speech(in_data, self.sample_rate)
        
        if is_speech:
            self.speech_buffer.extend(audio_data)
            self.is_speaking = True
        elif self.is_speaking:
            # Check if we have enough speech to transcribe
            if len(self.speech_buffer) >= self.min_speech_duration * self.sample_rate:
                # Transcribe in a separate thread to avoid blocking audio callback
                threading.Thread(target=self.transcribe_speech, daemon=True).start()
                self.speech_buffer = []
            self.is_speaking = False
        
        return (None, pyaudio.paContinue)
    
    def transcribe_speech(self):
        """Transcribe the collected speech audio"""
        if len(self.speech_buffer) < 480:  # Minimum 30ms
            return
        
        # Make a copy of the buffer to avoid race conditions
        audio_buffer = self.speech_buffer.copy()
        
        # Convert to float32 and normalize
        audio_data = np.array(audio_buffer, dtype=np.float32) / 32768.0
        
        # Process with Parakeet
        try:
            inputs = self.processor(
                audio_data,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # CTC models use forward pass, not generate
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Decode using the processor
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(
                    predicted_ids,
                    skip_special_tokens=True
                )[0]
            
            if transcription.strip():
                print(f"🎤 {transcription}")
                
        except Exception as e:
            print(f"Transcription error: {e}")

# Usage
if __name__ == "__main__":
    recognizer = OptimizedLiveRecognizer()
    recognizer.process_audio_stream()
