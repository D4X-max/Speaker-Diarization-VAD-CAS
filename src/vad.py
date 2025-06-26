# src/vad.py

import torch
from pyannote.audio import Pipeline # The main object we need from the library is 'Pipeline'.
import numpy as np
# Note: We do NOT need 'from pyannote.audio import VoiceActivityDetection' here.

def detect_voice_activity(waveform, sr):
    """
    Applies Voice Activity Detection using a pre-trained pyannote.audio pipeline.
    """
    print("Initializing VAD pipeline...")
    # This single line handles loading the correct model and setting up the VAD logic.
    pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection",
        use_auth_token=True # Or your token string
    )
    print("VAD pipeline initialized.")

    input_tensor = torch.from_numpy(waveform).unsqueeze(0)
    audio_data = {"waveform": input_tensor, "sample_rate": sr}

    print("Applying VAD...")
    vad_result = pipeline(audio_data)
    
    speech_segments = []
    for segment in vad_result.itersegments():
        speech_segments.append((segment.start, segment.end))
        # This print statement is optional but helpful for debugging
        # print(f"Detected speech from {segment.start:.2f}s to {segment.end:.2f}s")
        
    return speech_segments

# --- Example Usage (Optional - for testing this file directly) ---
if __name__ == '__main__':
    # You would need your preprocess_audio function here to run this block
    # from preprocess import preprocess_audio
    
    # input_audio_path = 'path/to/your/data/sample_audio.wav'
    # processed_waveform, sample_rate = preprocess_audio(input_audio_path)
    
    # segments = detect_voice_activity(processed_waveform, sample_rate)
    
    # print("\nFinal list of speech segments:")
    # print(segments)
    pass # Pass so it doesn't run by default when imported.


