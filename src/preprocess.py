import librosa
import numpy as np
import soundfile as sf

def preprocess_audio(file_path, target_sr=16000):
    """
    Loads, resamples, and normalizes an audio file.

    Args:
        file_path (str): Path to the input audio file.
        target_sr (int): The target sample rate.

    Returns:
        np.ndarray: The preprocessed audio waveform.
        int: The target sample rate.
    """
    # Loading the file using the Librosa library 
    waveform, sr = librosa.load(file_path, sr=None, mono=False)

    # 1. Convert to mono by averaging channels if it's stereo
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=0)

    # 2. Resample to the target sample rate if necessary
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    
    # 3. Normalize the audio to have a max amplitude of 1.0
    # This is a simple peak normalization.
    waveform = waveform / np.max(np.abs(waveform))

    print(f"Preprocessed {file_path}. Duration: {librosa.get_duration(y=waveform, sr=sr):.2f}s")
    
    return waveform, sr

# --- Example Usage ---
if __name__ == '__main__':
    # Place an example audio file in your 'data' folder
    # For example, 'data/sample_audio.wav'
    input_audio_path = 'data/Voice.wav' 
    
    processed_waveform, sample_rate = preprocess_audio(input_audio_path)
    
    # You can save the preprocessed audio to check it
    output_audio_path = 'processed_data/processed_sample.wav'
    sf.write(output_audio_path, processed_waveform, sample_rate)
    print(f"Saved preprocessed audio to {output_audio_path}")

