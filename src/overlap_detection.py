import torch
from pyannote.audio import Pipeline

def detect_overlaps(audio_path, hf_token):
    """
    Detect overlapping speech regions in an audio file.

    Args:
        audio_path (str): Path to audio file.
        hf_token (str): Hugging Face access token.

    Returns:
        list of tuples: Each tuple is (start_time, end_time) of overlap.
    """
    print("Loading overlapped speech detection pipeline...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/overlapped-speech-detection",
        use_auth_token=hf_token
    )
    if torch.cuda.is_available():
        pipeline.to("cuda")
    print("Pipeline loaded. Processing audio...")

    # Run pipeline on audio file path
    output = pipeline(audio_path)

    # Get timeline of overlapping speech regions
    overlap_segments = []
    for segment in output.get_timeline().support():
        overlap_segments.append((segment.start, segment.end))
        print(f"Overlap detected from {segment.start:.2f}s to {segment.end:.2f}s")

    return overlap_segments

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python overlap_detection.py <audio_path> <hf_token>")
        sys.exit(1)

    audio_file = sys.argv[1]
    token = sys.argv[2]
    overlaps = detect_overlaps(audio_file, token)
    print(f"\nTotal overlapping segments detected: {len(overlaps)}")
