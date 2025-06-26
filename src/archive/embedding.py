import torch
import numpy as np
from pyannote.audio import Pipeline
from pyannote.core import Segment
from preprocess import preprocess_audio
from archive.vad import detect_voice_activity
from pyannote.audio import Inference   # class that is used to extract embeddings

def extract_embeddings(waveform, sr, speech_segments):
    """
    Extracts a speaker embedding for each speech segment.

    Args:
        waveform (np.ndarray): The audio waveform.
        sr (int): The sample rate.
        speech_segments (list): A list of (start, end) tuples from VAD.

    Returns:
        list: A list of dictionaries, each containing the segment and its embedding.
    """
    # 1. Initialize the pre-trained embedding model pipeline
    print("Initializing embedding pipeline...")
    # This model is trained to produce speaker embeddings.
    embedding_pipeline = Inference(
        "pyannote/speaker-embedding",
        use_auth_token=True # Your HF token
    )
    print("Embedding pipeline initialized.")

    embeddings = []
    
    # 2. Iterate over each speech segment
    for start, end in speech_segments:
        # Create a pyannote.core.Segment object
        segment = Segment(start, end)
        
        # The pipeline needs the waveform and the segment to extract from.
        # It expects the waveform in a specific format.
        input_data = {
            "waveform": torch.from_numpy(waveform).unsqueeze(0),
            "sample_rate": sr
        }
        
        # Extract the embedding for the given segment
        # The output is a (1, D) numpy array, where D is the embedding dimension.
        # We use .squeeze() to remove the first dimension and get a 1D vector.
        embedding = embedding_pipeline.crop(input_data, segment).squeeze()
        
        embeddings.append({
            'segment': (start, end),
            'embedding': embedding,
            'duration': end - start
        })
        print(f"Extracted embedding for segment {start:.2f}s - {end:.2f}s")

    return embeddings

# --- Example Usage ---
if __name__ == '__main__':
    # 1. Preprocess audio
    input_audio_path = 'data/Voice.wav'
    processed_waveform, sample_rate = preprocess_audio(input_audio_path)
    
    # 2. Detect speech activity
    segments = detect_voice_activity(processed_waveform, sample_rate)
    
    # 3. Extract embeddings for each speech segment
    # Let's only process segments longer than a certain duration to avoid noisy results
    min_duration_for_embedding = 0.5 # seconds
    long_segments = [(start, end) for start, end in segments if end - start > min_duration_for_embedding]
    
    if not long_segments:
        print("No speech segments long enough for embedding extraction.")
    else:
        embedding_list = extract_embeddings(processed_waveform, sample_rate, long_segments)
    
        print(f"\nSuccessfully extracted {len(embedding_list)} embeddings.")
        # Print the first embedding to see what it looks like
        if embedding_list:
            first_embedding_info = embedding_list[0]
            print(f"Segment: {first_embedding_info['segment']}")
            print(f"Embedding Shape: {first_embedding_info['embedding'].shape}")
            # print(f"Embedding Vector (first 10 values): {first_embedding_info['embedding'][:10]}")

