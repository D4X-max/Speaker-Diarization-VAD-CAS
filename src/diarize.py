import torch
from pyannote.audio import Pipeline
from preprocess import preprocess_audio
import csv 
import os

def run_diarization(waveform, sr):
        
    print("Initializing diarization pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=True # Your HF token
    )
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    print("Diarization pipeline initialized.")

    audio_data = {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": sr}

    print("Applying diarization...")
    diarization_result = pipeline(audio_data)
    
    return diarization_result

def write_results_to_csv(diarization, output_csv_path):
    """
    Writes the diarization results to a CSV file.

    Args:
        diarization (pyannote.core.Annotation): The diarization result object.
        output_csv_path (str): The path to the output CSV file.
    """
    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the header
            writer.writerow(["start", "end", "speaker"])
            
            # Write the data
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                writer.writerow([f"{turn.start:.2f}", f"{turn.end:.2f}", speaker])
        
        print(f"Diarization results successfully written to {output_csv_path}")

    except Exception as e:
        print(f"Error writing to CSV file: {e}")
        
def write_rttm_file(diarization, output_rttm_path, audio_filename="my_audio"):
    """
    Writes the diarization results to an RTTM file.

    Args:
        diarization (pyannote.core.Annotation): The diarization result.
        output_rttm_path (str): The path to the output RTTM file.
        audio_filename (str): The name of the audio file, without extension.
    """
    try:
        with open(output_rttm_path, "w") as rttm_file:
            diarization.write_rttm(rttm_file)
        
        # The default RTTM format from pyannote may not include the filename, 
        # which can be an issue for some evaluators.
        # Let's read it back and prepend the filename if needed.
        # (This is a robust way to ensure compatibility)
        lines = []
        with open(output_rttm_path, 'r') as f:
            lines = f.readlines()
        
        with open(output_rttm_path, 'w') as f:
            for line in lines:
                parts = line.split(' ')
                # RTTM format: SPEAKER <file_id> <channel> <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
                # We want to ensure the <file_id> column is correct.
                parts[1] = audio_filename
                f.write(' '.join(parts))

        print(f"Diarization results successfully written to {output_rttm_path}")
    except Exception as e:
        print(f"Error writing to RTTM file: {e}")

# --- Example Usage ---
if __name__ == '__main__':
    try:
        input_audio_path = 'data/sample_audio.wav'
        output_dir_csv = 'csv_output'
        output_dir_rttm = 'rttm_output'
        # Get the base name of the audio file to use in output filenames
        audio_filename_base = os.path.splitext(os.path.basename(input_audio_path))[0]

        output_csv_path = os.path.join(output_dir_csv, f"{audio_filename_base}_diarization.csv")
        output_rttm_path = os.path.join(output_dir_rttm, f"{audio_filename_base}_diarization.rttm")
        
        processed_waveform, sample_rate = preprocess_audio(input_audio_path)
        diarization = run_diarization(processed_waveform, sample_rate)
        
        write_results_to_csv(diarization, output_csv_path)
        write_rttm_file(diarization, output_rttm_path, audio_filename_base)

    except Exception as e:
        print(f"\nAn error occurred: {e}")

    except Exception as e:
        print(f"\nAn error occurred during the main process: {e}")

    print("\nDiarization complete.")
