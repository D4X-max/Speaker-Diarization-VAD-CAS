import os
import sys
import argparse # Import argparse for better argument handling
import logging # For consistent logging

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import necessary functions from your existing modules
from preprocess import preprocess_audio
from diarize import run_diarization, write_results_to_csv, write_rttm_file

def run_speaker_change_detection(input_audio_path: str, auth_token: str, min_speakers: int = None, max_speakers: int = None, clustering_threshold: float = None):
    """
    Performs speaker diarization and highlights speaker change points.

    Args:
        input_audio_path (str): Path to the input audio file.
        auth_token (str): Hugging Face authentication token.
        min_speakers (int, optional): Minimum number of speakers expected.
        max_speakers (int, optional): Maximum number of speakers expected.
        clustering_threshold (float, optional): Threshold for clustering embeddings.
    """
    if not os.path.exists(input_audio_path):
        logger.error(f"Error: Audio file not found at '{input_audio_path}'")
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
        logger.info(f"--- Processing audio: {input_audio_path} ---")
        processed_waveform, sample_rate = preprocess_audio(input_audio_path)

        # 2. Run the full diarization pipeline, passing the auth_token and other params
        logger.info("--- Running Diarization ---")
        diarization = run_diarization(
            processed_waveform,
            sample_rate,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            clustering_threshold=clustering_threshold,
            auth_token=auth_token # <--- Pass the auth_token here
        )

        # 3. Write diarization results to files
        logger.info("--- Saving Diarization Outputs ---")
        write_results_to_csv(diarization, output_csv_path)
        write_rttm_file(diarization, output_rttm_path, audio_filename_base)

        # 4. Highlight Speaker Change Points
        logger.info("\n--- Speaker Change Points Detected ---")
        previous_speaker = None

        # Iterate through the diarization results to find change points
        for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
            if i == 0:
                logger.info(f"Audio starts with speaker: {speaker} at {turn.start:.2f}s")
            elif speaker != previous_speaker:
                logger.info(f"Change detected at {turn.start:.2f}s: from {previous_speaker} to {speaker}")
            previous_speaker = speaker
        logger.info("------------------------------------")

    except Exception as e:
        logger.error(f"\nAn error occurred during speaker change detection: {e}", exc_info=True)
        logger.error("Please ensure you have accepted all Hugging Face user conditions and your token is valid.")
        sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Speaker Diarization and highlight speaker change points.")
    parser.add_argument("input_audio", type=str,
                        help="Path to the input audio file (e.g., data/sample_audio.wav)")
    parser.add_argument("--auth_token", type=str, required=True,
                        help="Hugging Face authentication token.")
    parser.add_argument("--min_speakers", type=int, default=None,
                        help="Minimum number of speakers expected (passed to diarization).")
    parser.add_argument("--max_speakers", type=int, default=None,
                        help="Maximum number of speakers expected (passed to diarization).")
    parser.add_argument("--clustering_threshold", type=float, default=None,
                        help="Threshold for clustering embeddings (passed to diarization).")

    args = parser.parse_args()

    run_speaker_change_detection(
        args.input_audio,
        args.auth_token,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        clustering_threshold=args.clustering_threshold
    )

    logger.info("\nSpeaker change detection complete.")


