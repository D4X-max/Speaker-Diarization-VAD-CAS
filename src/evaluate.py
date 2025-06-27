import json
import os
import sys
import argparse
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_rttm_to_annotation(file_path: str) -> Annotation:
    """
    Reads an RTTM file and returns a pyannote.core.Annotation object.

    Args:
        file_path (str): Path to the RTTM file.

    Returns:
        Annotation: The diarization annotation.
    """
    uri = os.path.splitext(os.path.basename(file_path))[0]
    annotation = Annotation(uri=uri)

    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] != 'SPEAKER':
                    continue

                start_time = float(parts[3])
                duration = float(parts[4])
                speaker_id = parts[7]

                segment = Segment(start_time, start_time + duration)
                annotation[segment] = speaker_id  # Correct assignment syntax
    except Exception as e:
        logger.error(f"Error reading RTTM file '{file_path}': {e}")
        sys.exit(1)

    return annotation

def evaluate_diarization(reference_rttm_path: str, hypothesis_rttm_path: str, output_json_path: str = None):
    """
    Computes Diarization Error Rate (DER) between reference and hypothesis RTTM files.

    Args:
        reference_rttm_path (str): Path to ground truth RTTM file.
        hypothesis_rttm_path (str): Path to hypothesis RTTM file.
        output_json_path (str, optional): Path to save detailed DER results as JSON.
    """
    logger.info(f"Loading reference RTTM from: {reference_rttm_path}")
    reference = read_rttm_to_annotation(reference_rttm_path)

    logger.info(f"Loading hypothesis RTTM from: {hypothesis_rttm_path}")
    hypothesis = read_rttm_to_annotation(hypothesis_rttm_path)

    der_metric = DiarizationErrorRate()

    uem = reference.get_timeline()
    der_components = der_metric(reference, hypothesis, uem=uem, detailed=True)

    overall_der = der_components['diarization error rate']

    logger.info("\n--- Diarization Evaluation ---")
    der_metric.report(display=True)
    logger.info(f"Final DER = {overall_der * 100:.2f}%")

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

    if output_json_path:
        try:
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, 'w') as f:
                json.dump(json_output_data, f, indent=4)
            logger.info(f"DER results saved to JSON file: {output_json_path}")
        except Exception as e:
            logger.error(f"Error saving DER results to JSON: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate speaker diarization using DER metric.")
    parser.add_argument("reference_rttm", type=str, help="Path to ground truth RTTM file.")
    parser.add_argument("hypothesis_rttm", type=str, help="Path to hypothesis RTTM file.")
    parser.add_argument("--output_json", type=str, default=None, help="Optional path to save detailed DER results as JSON.")

    args = parser.parse_args()

    if not os.path.exists(args.reference_rttm):
        logger.error(f"Reference RTTM file not found: {args.reference_rttm}")
        sys.exit(1)

    if not os.path.exists(args.hypothesis_rttm):
        logger.error(f"Hypothesis RTTM file not found: {args.hypothesis_rttm}")
        logger.error("Please run the diarization script first to generate hypothesis RTTM.")
        sys.exit(1)

    evaluate_diarization(args.reference_rttm, args.hypothesis_rttm, args.output_json)
    logger.info("Evaluation complete.")





