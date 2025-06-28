# src/main.py

import os
import sys
import argparse
import logging

# Since this script is inside 'src', we can directly import sibling modules.
from preprocess import preprocess_audio
from diarize import run_diarization, write_results_to_csv, write_rttm_file
from visualize import plot_diarization, read_rttm_manual
from evaluate import evaluate_diarization

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def full_pipeline(args):
    """
    Runs the complete speaker diarization pipeline from preprocessing to evaluation.
    """
    logger.info(f"--- Starting Full Pipeline for: {args.input_audio} ---")

    # === Step 1: Preprocess Audio ===
    logger.info("Step 1: Preprocessing audio...")
    waveform, sr = preprocess_audio(args.input_audio)
    if waveform is None:
        logger.error("Preprocessing failed. Exiting pipeline.")
        sys.exit(1)

    # === Step 2: Run Diarization ===
    logger.info("Step 2: Running diarization...")
    diarization = run_diarization(
        waveform,
        sr,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        clustering_threshold=args.clustering_threshold,
        auth_token=args.auth_token
    )
    logger.info("Diarization complete.")

    # === Step 3: Save Outputs ===
    logger.info("Step 3: Saving diarization outputs...")
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)
    audio_filename_base = os.path.splitext(os.path.basename(args.input_audio))[0]
    
    csv_path = os.path.join(output_dir, f"{audio_filename_base}_diarization.csv")
    rttm_path = os.path.join(output_dir, f"{audio_filename_base}_diarization.rttm")
    
    write_results_to_csv(diarization, csv_path)
    write_rttm_file(diarization, rttm_path, audio_filename_base)
    logger.info(f"Outputs saved to {output_dir}")

    # === Step 4: Visualize Results ===
    logger.info("Step 4: Generating visualization...")
    plot_dir = 'plots'
    plot_path = os.path.join(plot_dir, f"{audio_filename_base}_diarization.png")
    plot_diarization(waveform, sr, diarization, output_png_path=plot_path)
    logger.info(f"Visualization saved to {plot_path}")

    # === Step 5: Evaluate (if reference RTTM exists) ===
    logger.info("Step 5: Evaluating diarization...")
    reference_rttm_path = os.path.join('data', f"{audio_filename_base}.rttm")
    if os.path.exists(reference_rttm_path):
        json_path = os.path.join(output_dir, f"{audio_filename_base}_der_results.json")
        evaluate_diarization(reference_rttm_path, rttm_path, output_json_path=json_path)
    else:
        logger.warning(f"Reference RTTM not found at '{reference_rttm_path}'. Skipping evaluation.")

    logger.info("--- Full Pipeline Finished ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the full speaker diarization pipeline.")
    parser.add_argument("input_audio", type=str, help="Path to the input audio file.")
    parser.add_argument("--auth_token", type=str, required=True, help="Hugging Face authentication token.")
    parser.add_argument("--min_speakers", type=int, default=None, help="Minimum number of speakers.")
    parser.add_argument("--max_speakers", type=int, default=None, help="Maximum number of speakers.")
    parser.add_argument("--clustering_threshold", type=float, default=None, help="Clustering threshold.")
    
    args = parser.parse_args()
    full_pipeline(args)

    logger.info("\nFull pipeline complete.")