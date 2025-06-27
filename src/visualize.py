import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import librosa
import librosa.display
from pyannote.core import Segment, Timeline, Annotation
import logging
import argparse
from typing import Optional, List, Tuple

# Import necessary functions from your existing modules
from preprocess import preprocess_audio
from diarize import run_diarization

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_rttm_manual(rttm_path: str) -> Annotation:
    """
    Manually reads an RTTM file and returns a pyannote.core.Annotation object.
    """
    annotation = Annotation()
    try:
        with open(rttm_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) >= 9 and parts[0] == 'SPEAKER':
                    start = float(parts[3])
                    duration = float(parts[4])
                    speaker_id = parts[7]
                    segment = Segment(start, start + duration)
                    annotation[segment] = speaker_id
        logger.info(f"Successfully loaded RTTM from {rttm_path}")
    except Exception as e:
        logger.error(f"Error reading RTTM file {rttm_path}: {e}")
        sys.exit(1)
    return annotation

def plot_diarization(
    waveform: np.ndarray,
    sr: int,
    diarization: Annotation,
    output_png_path: str = None,
    plot_width: float = 12,
    plot_height: float = 6
):
    """
    Plots the audio waveform with speaker diarization segments overlaid.
    """
    # CORRECTED LINE: Use Python's boolean evaluation for emptiness [3]
    if not diarization:
        logger.warning("Diarization is empty. No speaker segments to plot.")
        return

    duration = librosa.get_duration(y=waveform, sr=sr)
    speakers = sorted(diarization.labels())
    num_speakers = len(speakers)

    colors = plt.cm.get_cmap('tab10', max(10, num_speakers))
    speaker_colors = {speaker: colors(i % 10) for i, speaker in enumerate(speakers)}

    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    librosa.display.waveshow(waveform, sr=sr, ax=ax, alpha=0.6, color='grey')

    y_offset_step = 0.08
    y_min_wave, y_max_wave = ax.get_ylim()
    speaker_plot_height = (y_max_wave - y_min_wave) * 0.15
    base_y_position = y_max_wave + (y_max_wave - y_min_wave) * 0.05

    handles = []
    for i, speaker in enumerate(speakers):
        color = speaker_colors[speaker]
        speaker_segments = diarization.label_timeline(speaker)

        for segment in speaker_segments:
            rect = patches.Rectangle(
                (segment.start, base_y_position + i * y_offset_step),
                segment.duration,
                speaker_plot_height,
                facecolor=color,
                edgecolor='none',
                alpha=0.7,
                label=speaker
            )
            ax.add_patch(rect)

            if segment.duration > 0.5:
                ax.text(
                    segment.start + segment.duration / 2,
                    base_y_position + i * y_offset_step + speaker_plot_height / 2,
                    speaker,
                    ha='center', va='center', color='white', fontsize=8, weight='bold'
                )
        handles.append(patches.Patch(color=color, label=f'Speaker {speaker}'))

    ax.set_title(f"Speaker Diarization - Duration: {duration:.2f}s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, duration)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(bottom=y_min_wave, top=base_y_position + num_speakers * y_offset_step + speaker_plot_height + (y_max_wave - y_min_wave) * 0.05)
    
    unique_labels = {}
    for h in handles:
        unique_labels[h.get_label()] = h
    ax.legend(handles=list(unique_labels.values()), loc='upper right', bbox_to_anchor=(1.15, 1.0))
    
    plt.tight_layout()

    if output_png_path:
        os.makedirs(os.path.dirname(output_png_path), exist_ok=True)
        plt.savefig(output_png_path, dpi=300)
        logger.info(f"Plot saved to {output_png_path}")
        plt.close(fig)
    else:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Visualize Speaker Diarization results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_audio", type=str, help="Path to the input audio file.")
    parser.add_argument("--rttm_file", type=str, default=None, help="Optional: Path to an existing RTTM file. If provided, diarization is skipped.")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save the generated plot.")
    parser.add_argument("--output_name", type=str, default=None, help="Name of the output PNG file (without extension). Defaults to audio_filename_base_diarization.png.")
    parser.add_argument("--auth_token", type=str, default=None, help="Hugging Face authentication token. Required if --rttm_file is NOT provided.")
    parser.add_argument("--min_speakers", type=int, default=None, help="Minimum number of speakers (for live diarization).")
    parser.add_argument("--max_speakers", type=int, default=None, help="Maximum number of speakers (for live diarization).")
    parser.add_argument("--clustering_threshold", type=float, default=None, help="Clustering threshold (for live diarization, e.g., 0.7).")
    parser.add_argument("--plot_width", type=float, default=12, help="Width of the output plot in inches.")
    parser.add_argument("--plot_height", type=float, default=6, help="Height of the output plot in inches.")
    args = parser.parse_args()

    if not os.path.exists(args.input_audio):
        logger.error(f"Error: Audio file not found at '{args.input_audio}'")
        sys.exit(1)
    if args.rttm_file is None and args.auth_token is None:
        logger.error("Error: --auth_token is required if --rttm_file is not provided.")
        parser.print_help()
        sys.exit(1)
    if args.rttm_file and not os.path.exists(args.rttm_file):
        logger.error(f"Error: RTTM file not found at '{args.rttm_file}'")
        sys.exit(1)

    logger.info(f"Preprocessing audio: {args.input_audio}")
    waveform, sr = preprocess_audio(args.input_audio)
    if waveform is None:
        logger.error("Audio preprocessing failed. Exiting.")
        sys.exit(1)
    
    audio_filename_base = os.path.splitext(os.path.basename(args.input_audio))[0]
    
    diarization_annotation = None
    if args.rttm_file:
        logger.info(f"Loading diarization from RTTM file: {args.rttm_file}")
        diarization_annotation = read_rttm_manual(args.rttm_file)
    else:
        logger.info("Running live diarization...")
        diarization_annotation = run_diarization(
            waveform, sr,
            min_speakers=args.min_speakers, max_speakers=args.max_speakers,
            clustering_threshold=args.clustering_threshold, auth_token=args.auth_token
        )
        if diarization_annotation is None:
            logger.error("Live diarization failed. Exiting.")
            sys.exit(1)
            
    output_plot_name = args.output_name if args.output_name else f"{audio_filename_base}_diarization.png"
    output_png_path = os.path.join(args.output_dir, output_plot_name)
    
    logger.info("Generating plot...")
    plot_diarization(
        waveform, sr, diarization_annotation,
        output_png_path=output_png_path,
        plot_width=args.plot_width, plot_height=args.plot_height
    )
    logger.info("Visualization process complete.")





