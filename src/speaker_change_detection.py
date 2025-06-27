import os
import sys

# Import necessary functions from your existing modules
# Ensure these paths are correct relative to where you run the script (e.g., from the project root)
from preprocess import preprocess_audio
from diarize import run_diarization, write_results_to_csv, write_rttm_file

def run_speaker_change_detection(input_audio_path):
    """
    Performs speaker diarization and highlights speaker change points.

    Args:
        input_audio_path (str): Path to the input audio file.
    """
    if not os.path.exists(input_audio_path):
        print(f"Error: Audio file not found at '{input_audio_path}'")
        return

    # Create an 'outputs' directory if it doesn't exist
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the base name of the audio file to use in output filenames
    audio_filename_base = os.path.splitext(os.path.basename(input_audio_path))[0]

    # Define output paths
    output_csv_path = os.path.join(output_dir, f"{audio_filename_base}_diarization.csv")
    output_rttm_path = os.path.join(output_dir, f"{audio_filename_base}_diarization.rttm")

    try:
        # 1. Preprocess audio
        print(f"--- Processing audio: {input_audio_path} ---")
        processed_waveform, sample_rate = preprocess_audio(input_audio_path)
        
        # 2. Run the full diarization pipeline
        print("--- Running Diarization ---")
        diarization = run_diarization(processed_waveform, sample_rate)
        
        # 3. Write diarization results to files
        print("--- Saving Diarization Outputs ---")
        write_results_to_csv(diarization, output_csv_path)
        write_rttm_file(diarization, output_rttm_path, audio_filename_base)

        # 4. Highlight Speaker Change Points
        print("\n--- Speaker Change Points Detected ---")
        previous_speaker = None
        
        # Iterate through the diarization results to find change points
        for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
            if i == 0:
                print(f"Audio starts with speaker: {speaker} at {turn.start:.2f}s")
            elif speaker != previous_speaker:
                print(f"Change detected at {turn.start:.2f}s: from {previous_speaker} to {speaker}")
            previous_speaker = speaker
        print("------------------------------------")

    except Exception as e:
        print(f"\nAn error occurred during speaker change detection: {e}")
        print("Please ensure you have accepted all Hugging Face user conditions and your token is valid.")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/speaker_change_detection.py <path_to_audio_file>")
        sys.exit(1)
    
    # Get the audio file path from the command line argument
    audio_file_path_arg = sys.argv[1]
    
    run_speaker_change_detection(audio_file_path_arg)

    print("\nSpeaker change detection complete.")

