import pyaudio
import numpy as np
from transformers import AutoModelForCTC, AutoProcessor
import torch
import threading
import queue
import time
import collections

class StreamingRecognizer:
    def __init__(self, model_name="nvidia/parakeet-ctc-1.1b"):
        self.sample_rate = 16000
        self.chunk_size = 1600  # 100ms chunks
        self.buffer_size = self.sample_rate * 3  # 3-second buffer
        
        # Model setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        
        # Pre-compile for speed
        if self.device == "cuda":
            self.model.generate = torch.compile(self.model.generate, mode="reduce-overhead")
        
        # Streaming buffers
        self.audio_buffer = collections.deque(maxlen=self.buffer_size)
        self.transcription_queue = queue.Queue()
        self.is_running = True
        
    def start_streaming(self):
        """Start the streaming recognition"""
        # Start audio capture thread
        audio_thread = threading.Thread(target=self.capture_audio)
        audio_thread.start()
        
        # Start processing thread
        process_thread = threading.Thread(target=self.process_stream)
        process_thread.start()
        
        print("Streaming started... Press Ctrl+C to stop")
        
        try:
            while self.is_running:
                time.sleep(0.1)
                if not self.transcription_queue.empty():
                    transcription = self.transcription_queue.get()
                    print(f"🎤 {transcription}")
        except KeyboardInterrupt:
            self.is_running = False
            
    def capture_audio(self):
        """Capture audio in real-time"""
        pyaudio_instance = pyaudio.PyAudio()
        
        def callback(in_data, frame_count, time_info, status):
            audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
            self.audio_buffer.extend(audio_data)
            return (None, pyaudio.paContinue)
        
        stream = pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=callback
        )
        
        stream.start_stream()
        
        while self.is_running:
            time.sleep(0.1)
            
        stream.stop_stream()
        stream.close()
        pyaudio_instance.terminate()
    
    def process_stream(self):
        """Process audio stream for transcription"""
        while self.is_running:
            if len(self.audio_buffer) >= self.buffer_size:
                # Get current buffer
                audio_data = np.array(list(self.audio_buffer))
                
                # Process with Parakeet
                try:
                    inputs = self.processor(
                        audio_data,
                        sampling_rate=self.sample_rate,
                        return_tensors="pt",
                        padding=True,
                        max_length=self.buffer_size,
                        truncation=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = self.model.generate(**inputs)
                    
                    transcription = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
                    
                    if transcription.strip():
                        self.transcription_queue.put(transcription)
                        
                except Exception as e:
                    print(f"Processing error: {e}")
                
            time.sleep(0.5)  # Process every 500ms

# Usage
if __name__ == "__main__":
    recognizer = StreamingRecognizer()
    recognizer.start_streaming()
