# src/diarize.py

import torch
from pyannote.audio import Pipeline
from preprocess import preprocess_audio # Corrected import path
import csv 
import os
import argparse # Already imported, good!
import sys

def run_diarization(waveform, sr, min_speakers=None, max_speakers=None, clustering_threshold=None):
    """
    Runs the speaker diarization pipeline with configurable parameters.

    Args:
        waveform (np.ndarray): The audio waveform.
        sr (int): The sample rate.
        min_speakers (int, optional): Minimum number of speakers expected.
        max_speakers (int, optional): Maximum number of speakers expected.
        clustering_threshold (float, optional): Threshold for clustering embeddings.
    """
    print("Initializing diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=True # Your HF token
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    print("Diarization pipeline initialized.")

    # Configure pipeline hyperparameters
    # These parameters are specific to the pyannote.audio pipeline and its components.
    # The 'instantiate' method of the pipeline allows setting these parameters.
    # We must call it *after* from_pretrained, but *before* calling the pipeline itself.

    params_to_set = {}
    if min_speakers is not None:
        params_to_set['min_speakers'] = min_speakers
    if max_speakers is not None:
        params_to_set['max_speakers'] = max_speakers
    if clustering_threshold is not None:
        params_to_set['clustering'] = {'threshold': clustering_threshold} # Clustering is a sub-component

    if params_to_set:
        print(f"Setting pipeline parameters: {params_to_set}")
        # Apply the parameters. Note: pyannote pipeline expects dicts like this.
        # This will set parameters for the internal 'Diarization' module of the pipeline.
        # Check pyannote's documentation for exact parameter names.
        # For pyannote/speaker-diarization-3.1, these are top-level parameters often.
        # However, for fine-tuning specific components (like 'clustering'), you often pass a dict.
        
        # Simplest way to pass parameters for 3.1 pipeline is direct attribute assignment
        # or by passing them to the pipeline's __call__ method if supported.
        # Let's use direct __call__ parameters where supported for simplicity.
        pass # We will pass these directly to the pipeline call below

    audio_data = {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": sr}

    print("Applying diarization...")
    
    # Pass min_speakers, max_speakers directly to the pipeline's __call__ method
    # For 'clustering_threshold', this usually modifies the internal behavior of the pipeline
    # and might require rebuilding/re-instantiating the pipeline's components.
    # Let's focus on min/max speakers for simplicity via __call__ method.
    # If clustering_threshold needs to be set, it's typically done like:
    # pipeline.freeze({"clustering": {"threshold": clustering_threshold}})
    # And then call pipeline.
    
    # For pyannote 3.1, min_speakers and max_speakers are direct arguments to the __call__ method.
    diarization_result = pipeline(
        audio_data, 
        min_speakers=min_speakers, 
        max_speakers=max_speakers
    )
    
    return diarization_result

def write_results_to_csv(diarization, output_csv_path):
    # ... (this function remains the same) ...
    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["start", "end", "speaker"])
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                writer.writerow([f"{turn.start:.2f}", f"{turn.end:.2f}", speaker])
        print(f"Diarization results successfully written to {output_csv_path}")
    except Exception as e:
        print(f"Error writing to CSV file: {e}")
        
def write_rttm_file(diarization, output_rttm_path, audio_filename="my_audio"):
    # ... (this function remains the same) ...
    try:
        with open(output_rttm_path, "w") as rttm_file:
            diarization.write_rttm(rttm_file)
        lines = []
        with open(output_rttm_path, 'r') as f:
            lines = f.readlines()
        with open(output_rttm_path, 'w') as f:
            for line in lines:
                parts = line.split(' ')
                parts[1] = audio_filename
                f.write(' '.join(parts))
        print(f"Diarization results successfully written to {output_rttm_path}")
    except Exception as e:
        print(f"Error writing to RTTM file: {e}")

# --- Main Execution Block with argparse ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Speaker Diarization with configurable parameters.")
    parser.add_argument("input_audio", type=str, 
                        help="Path to the input audio file (e.g., data/sample_audio.wav)")
    parser.add_argument("--min_speakers", type=int, default=None,
                        help="Minimum number of speakers expected.")
    parser.add_argument("--max_speakers", type=int, default=None,
                        help="Maximum number of speakers expected.")
    # For clustering_threshold, pyannote's 3.x pipeline often exposes this internally.
    # To truly configure it, you might need to instantiate the Diarization module separately
    # or override pipeline parameters. For now, let's keep min/max speakers as the primary example.
    # parser.add_argument("--clustering_threshold", type=float, default=None,
    #                     help="Threshold for clustering speaker embeddings.")
    
    args = parser.parse_args()

    try:
        input_audio_path = args.input_audio
        output_dir = 'outputs' # Consistent output directory
        os.makedirs(output_dir, exist_ok=True)
        
        audio_filename_base = os.path.splitext(os.path.basename(input_audio_path))[0]

        output_csv_path = os.path.join(output_dir, f"{audio_filename_base}_diarization.csv")
        output_rttm_path = os.path.join(output_dir, f"{audio_filename_base}_diarization.rttm")
        
        processed_waveform, sample_rate = preprocess_audio(input_audio_path)
        
        # Pass the parsed arguments to the diarization function
        diarization = run_diarization(
            processed_waveform, 
            sample_rate, 
            min_speakers=args.min_speakers, 
            max_speakers=args.max_speakers
            # clustering_threshold=args.clustering_threshold
        )
        
        write_results_to_csv(diarization, output_csv_path)
        write_rttm_file(diarization, output_rttm_path, audio_filename_base)

        print("\nDiarization complete.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Add common error hints from previous steps
        if "use_auth_token" in str(e) or "access token" in str(e).lower():
            print("Hint: Hugging Face authentication issue. Check login and model conditions.")
        elif "CUDA" in str(e) or "GPU" in str(e):
            print("Hint: GPU error. Check your CUDA setup or PyTorch installation for CPU.")
        sys.exit(1) # Exit with an error code


