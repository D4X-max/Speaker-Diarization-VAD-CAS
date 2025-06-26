import os
from pyannote.core import Annotation, Segment, RTTMParser
from pyannote.metrics.diarization import DiarizationErrorRate

def evaluate_diarization(reference_rttm_path, hypothesis_rttm_path):
    """
    Computes the Diarization Error Rate (DER).

    Args:
        reference_rttm_path (str): Path to the ground truth RTTM file.
        hypothesis_rttm_path (str): Path to the system's output RTTM file.
    """
    # 1. Load RTTM files into pyannote.core.Annotation objects
    parser = RTTMParser()
    try:
        with open(reference_rttm_path, 'r') as f:
            reference = parser.read(f).get_annotations(uri=os.path.splitext(os.path.basename(reference_rttm_path))[0])
        with open(hypothesis_rttm_path, 'r') as f:
            hypothesis = parser.read(f).get_annotations(uri=os.path.splitext(os.path.basename(reference_rttm_path))[0]) # Use same URI
    except Exception as e:
        print(f"Error loading RTTM files: {e}")
        return

    # 2. Initialize the metric
    der_metric = DiarizationErrorRate()

    # 3. Compute the metric
    # The 'uem' (un-evaluated map) can be used to specify regions to evaluate.
    # We'll evaluate the whole file by getting the timeline of the reference.
    uem = reference.get_timeline()
    
    der_result = der_metric(reference, hypothesis, uem=uem)

    print("\n--- Diarization Evaluation ---")
    print(f"Reference RTTM: {reference_rttm_path}")
    print(f"Hypothesis RTTM: {hypothesis_rttm_path}")
    print("\n")
    
    # 4. Print the detailed report
    # The 'display=True' flag prints a clean, formatted table
    der_metric.report(display=True)
    
    print(f"\nFinal DER = {der_result*100:.2f}%")


# --- Example Usage ---
if __name__ == '__main__':
    # This assumes you have run diarize.py and it created an output RTTM.
    # It also assumes you have created a manual ground truth RTTM.
    
    # Define the base name of the audio file you tested
    audio_filename_base = "sample_audio" # IMPORTANT: Change this to your file's name

    reference_path = f"data/{audio_filename_base}.rttm"
    hypothesis_path = f"outputs/{audio_filename_base}_diarization.rttm"

    if not os.path.exists(reference_path):
        print(f"Error: Ground truth file not found at {reference_path}")
        print("Please create it manually by listening to your audio file.")
    elif not os.path.exists(hypothesis_path):
        print(f"Error: Hypothesis file not found at {hypothesis_path}")
        print("Please run 'python src/diarize.py' first to generate the output.")
    else:
        evaluate_diarization(reference_path, hypothesis_path)

