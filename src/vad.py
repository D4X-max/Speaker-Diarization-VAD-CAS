import torch
from pyannote.audio import Pipeline
import librosa
import numpy as np
from preprocess import preprocess_audio # Assuming you saved your function in preprocess.py

# --- Optional: In case you find authentication issues
# from huggingface_hub import login
# login() # Use your HF token here if you run into authentication issues

def detect_voice_activity(waveform, sr):
    """
    Applies Voice Activity Detection using a pre-trained pyannote.audio model.

    Args:
        waveform (np.ndarray): The audio waveform.
        sr (int): The sample rate of the audio.

    Returns:
        list: A list of tuples, where each tuple is a speech segment (start_time, end_time).
    """
    # 1. Initialize the VAD pipeline
    # The first time you run this, it will download the model.
    # To use this pipeline, you must agree to the user terms on Hugging Face Hub.
    print("Initializing VAD pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/voice-activity-detection",
        use_auth_token=True # Replace with your HF token string if you have one, or set to True
    )
    print("VAD pipeline initialized.")

    # 2. The pipeline expects a specific format: a dictionary with 'waveform' and 'sample_rate'
    # The waveform needs to be a torch.Tensor with shape [1, num_samples]
    input_tensor = torch.from_numpy(waveform).unsqueeze(0)
    audio_data = {"waveform": input_tensor, "sample_rate": sr}

    # 3. Apply the pipeline to get speech segments
    print("Applying VAD...")
    vad_result = pipeline(audio_data)
    
    # 4. The result is a pyannote.core.Annotation object. We can extract the segments.
    speech_segments = []
    for segment in vad_result.itersegments():
        speech_segments.append((segment.start, segment.end))
        print(f"Detected speech from {segment.start:.2f}s to {segment.end:.2f}s")
        
    return speech_segments

# --- Example Usage ---
if __name__ == '__main__':
    # Use the preprocessing function from Step 1
    input_audio_path = 'data/Voice.wav'
    processed_waveform, sample_rate = preprocess_audio(input_audio_path)
    
    # Get the speech segments
    segments = detect_voice_activity(processed_waveform, sample_rate)
    
    print("\nFinal list of speech segments:")
    print(segments)

