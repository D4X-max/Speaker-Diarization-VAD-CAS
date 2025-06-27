import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Import pyannote.core Annotation and Segment for consistency with diarization pipeline output
from pyannote.core import Annotation, Segment

# Import our preprocessing function
from preprocess import preprocess_audio
# Import run_diarization to get the Annotation object directly
from diarize import run_diarization

# --- NEW FUNCTION: Manual RTTM Parser ---
def read_rttm_manual(file_path):
    """
    Manually parse an RTTM file and return a pyannote.core.Annotation object.
    This function is used when a pre-computed RTTM file is provided.

    Args:
        file_path (str): Path to the RTTM file.

    Returns:
        Annotation: pyannote.core.Annotation object with segments and speaker labels.
    """
    annotation = Annotation()
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # RTTM format: SPEAKER <file_id> <channel> <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
                # Example: SPEAKER sample_audio 1 0.52 3.79 <NA> <NA> SPEAKER_00 <NA> <NA>
                if len(parts) >= 8 and parts[0] == "SPEAKER":
                    try:
                        start_time = float(parts[3])
                        duration = float(parts[4])
                        speaker_id = parts[7]
                        end_time = start_time + duration
                        segment = Segment(start_time, end_time)
                        annotation[segment] = speaker_id
                    except ValueError as ve:
                        print(f"Warning: Could not parse line (value error): {line.strip()} - {ve}")
                    except IndexError as ie:
                        print(f"Warning: Could not parse line (index error): {line.strip()} - {ie}")
                # else:
                #     print(f"Warning: Skipping malformed RTTM line: {line.strip()}")
    except FileNotFoundError:
        print(f"Error: RTTM file not found at '{file_path}'")
    except Exception as e:
        print(f"An unexpected error occurred while reading RTTM: {e}")
    return annotation

# --- Existing plot_diarization function (slightly adjusted for clarity) ---
def plot_diarization(audio_path, diarization_annotation, output_image_path=None):
    """
    Generates a timeline plot of diarization results.

    Args:
        audio_path (str): Path to the original audio file.
        diarization_annotation (pyannote.core.Annotation): The diarization result object.
        output_image_path (str, optional): Path to save the plot image. If None, displays the plot.
    """
    print(f"Loading audio for visualization: {audio_path}")
    waveform, sr = preprocess_audio(audio_path) # Use your preprocess function to ensure consistency

    plt.figure(figsize=(15, 6))

    # Plot the waveform (or spectrogram if preferred)
    librosa.display.waveshow(waveform, sr=sr, alpha=0.6, label='Audio Waveform')
    plt.title('Speaker Diarization Timeline')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Define a color map for speakers
    # Get unique speakers and assign a color to each
    speakers = sorted(diarization_annotation.labels())
    colors = plt.cm.get_cmap('tab10', max(len(speakers), 1)) # Ensure at least one color for 0 speakers
    speaker_color_map = {speaker: colors(i % 10) for i, speaker in enumerate(speakers)} # Use modulo for more than 10 speakers

    # Set Y-axis limits to accommodate waveform and speaker bars
    waveform_max_amp = np.max(np.abs(waveform)) if waveform.size > 0 else 0.1 # Handle empty waveform
    y_min = -waveform_max_amp * 1.1 if waveform_max_amp > 0 else -0.1
    y_max = waveform_max_amp * 1.5 if waveform_max_amp > 0 else 1.0 # Give more space above waveform
    
    plt.ylim(y_min, y_max)
    
    # Define y-positions for the speaker bars and labels
    speaker_bar_base_y = y_max * 0.8 # Start bars higher up
    speaker_bar_height = y_max * 0.05 # Height of each bar
    speaker_text_y_offset = speaker_bar_height * 1.2 # Offset for text above bar

    # Plot each speaker turn
    for turn, _, speaker in diarization_annotation.itertracks(yield_label=True):
        start_time = turn.start
        end_time = turn.end
        duration = turn.duration
        
        color = speaker_color_map.get(speaker, 'gray') # Fallback to gray

        # Draw a rectangle for each speaker turn
        plt.gca().add_patch(plt.Rectangle(
            (start_time, speaker_bar_base_y), # Bottom-left corner (x, y)
            duration,                           # Width
            speaker_bar_height,                 # Height
            facecolor=color,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        ))
        
        # Add speaker label to the center of the segment if it's long enough
        if duration > 0.5: # Only label longer segments to avoid clutter
            plt.text(
                start_time + duration / 2,
                speaker_bar_base_y + speaker_text_y_offset, # Slightly above the bar
                speaker,
                ha='center',
                va='bottom',
                color='black', # Text color for better contrast
                fontsize=8,
                fontweight='bold',
                clip_on=True # Clip text if it goes outside axes
            )

    # Create a simple legend for speaker colors
    # Ensure no duplicate entries if a speaker appears multiple times
    seen_speakers = set()
    handles = []
    for s in speakers:
        if s not in seen_speakers:
            handles.append(plt.Line2D([0], [0], color=speaker_color_map[s], lw=4, label=s))
            seen_speakers.add(s)

    if handles: # Only add legend if there are speakers
        plt.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.05, 1))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if output_image_path:
        plt.savefig(output_image_path)
        print(f"Plot saved to {output_image_path}")
        plt.close() # Close plot to free memory
    else:
        plt.show()


# --- Main Execution Block ---
if __name__ == '__main__':
    # Usage: python src/visualize.py <audio_file_path> [--rttm <rttm_file_path>] [output_image_path]
    # Example 1 (run diarization and visualize): python src/visualize.py data/sample_audio.wav outputs/diarization_plot.png
    # Example 2 (use existing RTTM for visualization): python src/visualize.py data/sample_audio.wav --rttm outputs/sample_audio_diarization.rttm outputs/diarization_plot.png

    audio_file_path = None
    rttm_file_path = None
    output_image_path = None
    
    args = sys.argv[1:] # Get arguments excluding script name

    # Parse arguments
    i = 0
    while i < len(args):
        if args[i] == '--rttm':
            if i + 1 < len(args):
                rttm_file_path = args[i+1]
                i += 2
            else:
                print("Error: --rttm requires a file path.")
                sys.exit(1)
        elif audio_file_path is None: # First non-flag argument is audio file
            audio_file_path = args[i]
            i += 1
        elif output_image_path is None: # Second non-flag argument is output image path
            output_image_path = args[i]
            i += 1
        else:
            print(f"Warning: Unexpected argument '{args[i]}'. Ignoring.")
            i += 1

    if not audio_file_path:
        print("Usage: python src/visualize.py <audio_file_path> [--rttm <rttm_file_path>] [output_image_path]")
        sys.exit(1)

    if not os.path.exists(audio_file_path):
        print(f"Error: Audio file not found at '{audio_file_path}'")
        sys.exit(1)

    try:
        diarization_annotation = None
        if rttm_file_path:
            print(f"--- Loading diarization from RTTM: {rttm_file_path} ---")
            diarization_annotation = read_rttm_manual(rttm_file_path)
            if not diarization_annotation.get_timeline(): # Check if annotation is empty
                print("Warning: Loaded RTTM file is empty or malformed. No diarization data to plot.")
                sys.exit(0)
        else:
            # If no RTTM is provided, run diarization live
            print(f"--- Running diarization for: {audio_file_path} ---")
            processed_waveform, sample_rate = preprocess_audio(audio_file_path)
            diarization_annotation = run_diarization(processed_waveform, sample_rate)
            if not diarization_annotation.get_timeline(): # Check if annotation is empty
                print("No speech detected by diarization pipeline. Nothing to plot.")
                sys.exit(0)

        # Ensure output directory exists if saving image
        if output_image_path:
            output_dir = os.path.dirname(output_image_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                
        plot_diarization(audio_file_path, diarization_annotation, output_image_path)

    except Exception as e:
        print(f"\nAn error occurred during visualization: {e}")
        # Provide hints for common pyannote errors
        if "use_auth_token" in str(e) or "access token" in str(e).lower():
            print("Hint: This might be a Hugging Face authentication issue. Make sure you are logged in via 'huggingface-cli login' and have accepted the user conditions for all required pyannote models.")
        elif "CUDA" in str(e) or "GPU" in str(e):
            print("Hint: GPU error. If you don't have a CUDA-enabled GPU or want to run on CPU, ensure PyTorch is installed correctly for CPU only, or check your CUDA setup.")
        sys.exit(1)


