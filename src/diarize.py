# src/diarize.py (Final Revised run_diarization function)

import torch
from pyannote.audio import Pipeline
from preprocess import preprocess_audio
import csv
import os
import argparse
import sys
import logging
from typing import Optional

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Pipeline Variable ---
# To avoid reloading the model on every call if used in a larger app
PIPELINE = None

def initialize_pipeline(auth_token: str):
    """Initializes the pyannote pipeline and caches it."""
    global PIPELINE
    if PIPELINE is None:
        logger.info("Initializing diarization pipeline for the first time...")
        try:
            PIPELINE = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=auth_token
            )
            if torch.cuda.is_available():
                PIPELINE.to(torch.device("cuda"))
                logger.info("Pipeline moved to GPU.")
            else:
                logger.info("GPU not available. Using CPU for pipeline.")
            logger.info("Diarization pipeline initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load pipeline: {e}")
            if "401" in str(e) or "access token" in str(e).lower():
                 logger.error("Hint: Hugging Face authentication error. Check your auth token.")
            sys.exit(1)
    return PIPELINE

def run_diarization(waveform, sr, min_speakers: Optional[int] = None, max_speakers: Optional[int] = None, clustering_threshold: Optional[float] = None, auth_token: str = None):
    """
    Runs the speaker diarization pipeline with configurable parameters.
    """
    pipeline = initialize_pipeline(auth_token)

    # --- Set hyperparameters for internal components directly on the pipeline object ---
    # This modifies the pipeline's behavior for this run and subsequent runs
    # if the pipeline instance is reused.
    if clustering_threshold is not None:
        logger.info(f"Setting clustering threshold to: {clustering_threshold}")
        # Access the 'clustering' component and set its 'threshold' attribute
        # Make sure this 'clustering' attribute exists on the pipeline object.
        # This is the standard way to set it for pyannote.audio SpeakerDiarization.
        pipeline.clustering.threshold = clustering_threshold

    audio_data = {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": sr}

    logger.info("Applying diarization...")

    # The pipeline's __call__ method takes min_speakers and max_speakers directly
    # but does NOT take a 'hyperparameters' keyword argument for the pre-trained pipeline.
    diarization_result = pipeline(
        audio_data,
        min_speakers=min_speakers,
        max_speakers=max_speakers
    )

    return diarization_result

def write_results_to_csv(diarization, output_csv_path: str):
    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["start_seconds", "end_seconds", "speaker_label"])
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                writer.writerow([f"{turn.start:.3f}", f"{turn.end:.3f}", speaker])
        logger.info(f"Diarization results successfully written to {output_csv_path}")
    except Exception as e:
        logger.error(f"Error writing to CSV file: {e}")

def write_rttm_file(diarization, output_rttm_path: str, audio_filename: str):
    try:
        with open(output_rttm_path, "w") as rttm_file:
            diarization.write_rttm(rttm_file)

        lines = []
        with open(output_rttm_path, 'r') as f:
            lines = f.readlines()
        with open(output_rttm_path, 'w') as f:
            for line in lines:
                parts = line.split(' ')
                parts[1] = audio_filename # Replace placeholder file ID
                f.write(' '.join(parts))
        logger.info(f"Diarization results successfully written to {output_rttm_path}")
    except Exception as e:
        logger.error(f"Error writing to RTTM file: {e}")

# --- Main Execution Block with argparse ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Speaker Diarization with pyannote.audio and configurable parameters.")
    parser.add_argument("input_audio", type=str, help="Path to the input audio file.")
    parser.add_argument("--min_speakers", type=int, default=None, help="Minimum number of speakers.")
    parser.add_argument("--max_speakers", type=int, default=None, help="Maximum number of speakers.")
    parser.add_argument("--clustering_threshold", type=float, default=None, help="Clustering threshold (e.g., 0.7).")
    parser.add_argument("--auth_token", type=str, required=True, help="Hugging Face authentication token.")

    args = parser.parse_args()

    try:
        input_audio_path = args.input_audio
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)

        audio_filename_base = os.path.splitext(os.path.basename(input_audio_path))[0]
        output_csv_path = os.path.join(output_dir, f"{audio_filename_base}_diarization.csv")
        output_rttm_path = os.path.join(output_dir, f"{audio_filename_base}_diarization.rttm")

        processed_waveform, sample_rate = preprocess_audio(input_audio_path)

        if processed_waveform is not None:
            diarization = run_diarization(
                processed_waveform,
                sample_rate,
                min_speakers=args.min_speakers,
                max_speakers=args.max_speakers,
                clustering_threshold=args.clustering_threshold, # Pass it to run_diarization
                auth_token=args.auth_token
            )

            write_results_to_csv(diarization, output_csv_path)
            write_rttm_file(diarization, output_rttm_path, audio_filename_base)

            logger.info("\nDiarization complete.")
        else:
            logger.error("Preprocessing failed. Halting diarization.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"An unexpected error occurred in the main script: {e}", exc_info=True)
        sys.exit(1)



