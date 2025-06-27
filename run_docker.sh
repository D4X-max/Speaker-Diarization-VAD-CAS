#!/bin/bash

# Define image name
IMAGE_NAME="speaker-diarization"

# --- Build the Docker image ---
echo "Building Docker image..."
docker build -t "$IMAGE_NAME" .
if [ $? -ne 0 ]; then
    echo "Docker image build failed."
    exit 1
fi
echo "Docker image '$IMAGE_NAME' built successfully."

# --- Run the Diarization ---
echo ""
echo "Running speaker diarization in Docker..."
echo "Usage: ./run_docker.sh run <audio_file_on_host> [optional_diarization_args]"
echo "Example: ./run_docker.sh run data/my_audio.wav --min_speakers 1 --max_speakers 2"
echo ""

# Check if 'run' command is provided
if [ "$1" == "run" ]; then
    shift # Remove 'run' from arguments
    AUDIO_FILE_HOST="$1"
    shift # Remove audio file from arguments

    if [ -z "$AUDIO_FILE_HOST" ]; then
        echo "Error: Please provide an audio file path on your host machine."
        exit 1
    fi

    # Get the absolute path to the audio file and its parent directory
    AUDIO_DIR_HOST=$(dirname "$(realpath "$AUDIO_FILE_HOST")")
    AUDIO_FILENAME=$(basename "$AUDIO_FILE_HOST")

    # Determine the audio file path inside the container's /data mount
    AUDIO_FILE_CONTAINER="/data/$AUDIO_FILENAME"
    
    # Define local output directory (will be mapped to /app/outputs in container)
    LOCAL_OUTPUT_DIR="outputs"
    mkdir -p "$LOCAL_OUTPUT_DIR" # Ensure local outputs directory exists

    # Run the Docker container
    # -v: Mounts volumes (host path:container path) for data and outputs
    # -e: Passes environment variables (like Hugging Face token)
    # --rm: Removes the container after it exits
    docker run \
        --rm \
        -v "$AUDIO_DIR_HOST":/data \
        -v "$(pwd)/$LOCAL_OUTPUT_DIR":/app/outputs \
        -e HF_TOKEN="$HF_TOKEN" \
        "$IMAGE_NAME" \
        python src/diarize.py "$AUDIO_FILE_CONTAINER" "$@"

    if [ $? -ne 0 ]; then
        echo "Docker run command failed."
        exit 1
    fi
    echo "Diarization finished. Results are in the '$LOCAL_OUTPUT_DIR' directory on your host."
    echo "Remember to accept Hugging Face model conditions if this is your first run."
else
    echo "No 'run' command provided. To run diarization, use: ./run_docker.sh run <audio_file_path> [diarization_args]"
fi
