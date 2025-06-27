import librosa
import numpy as np
import soundfile as sf
import os
import logging
from typing import Tuple, Optional # Import necessary types

# Configure logging for this module
# In a larger project, you'd typically configure logging centrally,
# but for this module, basicConfig is sufficient for demonstration.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get a logger for this module [2][7]

def rms_normalize(audio: np.ndarray, target_level: float = 0.1) -> np.ndarray:
    """
    Normalize audio to a target RMS level.

    Args:
        audio (np.ndarray): Input waveform.
        target_level (float): Desired RMS amplitude (e.g., 0.1).

    Returns:
        np.ndarray: RMS-normalized audio.
    """
    rms = np.sqrt(np.mean(audio**2))
    # Use a small epsilon to prevent division by near-zero RMS
    if rms < 1e-8: # Added a small threshold
        logger.warning("RMS of audio is very close to zero, skipping normalization.")
        return audio
    return audio * (target_level / rms)

def preprocess_audio(file_path: str, target_sr: int = 16000) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Loads, resamples, and RMS-normalizes an audio file.

    Args:
        file_path (str): Path to the input audio file.
        target_sr (int): The target sample rate.

    Returns:
        Tuple[Optional[np.ndarray], Optional[int]]: The preprocessed audio waveform and sample rate.
                                                     Returns (None, None) if an error occurs.
    """
    try:
        # Loading the file using the Librosa library
        waveform, sr = librosa.load(file_path, sr=None, mono=False)
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}") # Use logger.error for errors
        return None, None

    # 1. Convert to mono by averaging channels if it's stereo
    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=0)
        logger.info(f"Converted stereo audio to mono for {os.path.basename(file_path)}")

    # 2. Resample to the target sample rate if necessary
    if sr != target_sr:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
        logger.info(f"Resampled {os.path.basename(file_path)} to {target_sr} Hz.")

    # 3. Normalize using RMS loudness normalization
    waveform = rms_normalize(waveform, target_level=0.1)
    logger.info(f"RMS normalized {os.path.basename(file_path)}.")

    duration = librosa.get_duration(y=waveform, sr=sr)
    logger.info(f"Preprocessed {os.path.basename(file_path)}. Duration: {duration:.2f}s") # Use logger.info for general info

    return waveform, sr

# --- Example Usage ---
if __name__ == '__main__':
    # Place an example audio file in your 'data' folder
    # For example, 'data/sample_audio.wav'
    input_audio_path = 'data/sample_audio.wav'

    # Ensure data directory exists for the example input
    if not os.path.exists('data'):
        os.makedirs('data')
        logger.warning("Created 'data' directory. Please place 'sample_audio.wav' inside it.")
        # Optionally, create a dummy file if you don't want to stop execution
        # sf.write('data/sample_audio.wav', np.random.randn(16000 * 5), 16000)

    processed_waveform, sample_rate = preprocess_audio(input_audio_path)

    if processed_waveform is not None and sample_rate is not None:
        output_audio_dir = 'processed_data'
        output_audio_path = os.path.join(output_audio_dir, 'processed_sample.wav')
        os.makedirs(output_audio_dir, exist_ok=True)  # Ensure directory exists
        sf.write(output_audio_path, processed_waveform, sample_rate)
        logger.info(f"Saved preprocessed audio to {output_audio_path}") # Use logger.info
    else:
        logger.error(f"Failed to preprocess audio from {input_audio_path}.")




