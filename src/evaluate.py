import json
import os
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate

# --- Custom RTTM Reader Function ---
# This function makes our script independent of the pyannote.database RTTM parser location,
# which has changed across versions. This is the most stable approach.
def read_rttm_to_annotation(file_path):
    """
    Reads an RTTM file and returns a pyannote.core.Annotation object.
    """
    uri = os.path.splitext(os.path.basename(file_path))[0]
    annotation = Annotation(uri=uri) # Create an annotation with the file's URI
    
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != 'SPEAKER':
                continue # Skip empty or non-speaker lines
            
            # RTTM format: SPEAKER <file_id> <channel> <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
            start_time = float(parts[3])
            duration = float(parts[4])
            speaker_id = parts[7]
            
            segment = Segment(start_time, start_time + duration)
            annotation[segment, speaker_id] = speaker_id # Assign speaker to segment
            
    return annotation


def evaluate_diarization(reference_rttm_path, hypothesis_rttm_path, output_json_path=None):
    """
    Computes the Diarization Error Rate (DER) and optionally saves detailed results to JSON.
    """
    try:
        # Load annotations using our custom reader function
        reference = read_rttm_to_annotation(reference_rttm_path)
        hypothesis = read_rttm_to_annotation(hypothesis_rttm_path)
    except Exception as e:
        print(f"Error loading RTTM files with custom parser: {e}")
        return

    # Initialize the metric
    der_metric = DiarizationErrorRate()

    # Compute the metric and request detailed components
    uem = reference.get_timeline()
    der_components = der_metric(reference, hypothesis, uem=uem, detailed=True) 

    overall_der = der_components['diarization error rate']

    print("\n--- Diarization Evaluation ---")
    der_metric.report(display=True)
    print(f"\nFinal DER = {overall_der*100:.2f}%")
    
    # Prepare data for JSON export
    json_output_data = {
        "overall_der": overall_der,
        "der_percentage": round(overall_der * 100, 2),
        "components": {
            "false_alarm_seconds": round(der_components['false alarm'], 2),
            "missed_detection_seconds": round(der_components['missed detection'], 2),
            "total_speech_seconds": round(der_components['total'], 2)
        },
        "reference_file": reference_rttm_path,
        "hypothesis_file": hypothesis_rttm_path
    }

    # Save to JSON file if path is provided
    if output_json_path:
        try:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, 'w') as f:
                json.dump(json_output_data, f, indent=4)
            print(f"DER results successfully saved to {output_json_path}")
        except Exception as e:
            print(f"Error saving DER results to JSON: {e}")


# --- Example Usage ---
if __name__ == '__main__':
    audio_filename_base = "sample_audio" 

    reference_path = os.path.join("data", f"{audio_filename_base}.rttm")
    output_dir = "rttm_output"
    der_output = "DER" 
    hypothesis_path = os.path.join(output_dir, f"{audio_filename_base}_diarization.rttm")
    der_json_path = os.path.join(der_output, f"{audio_filename_base}_der_results.json")

    if not os.path.exists(reference_path):
        print(f"Error: Ground truth file not found at {reference_path}")
    elif not os.path.exists(hypothesis_path):
        print(f"Error: Hypothesis file not found at {hypothesis_path}")
        print("Please run 'python src/diarize.py' first.")
    else:
        evaluate_diarization(reference_path, hypothesis_path, output_json_path=der_json_path)

    print("\nEvaluation complete.")




