# Speaker Diarization System

## Project Overview

This project implements a comprehensive speaker diarization system capable of identifying "who spoke when" in an audio recording. It leverages the state-of-the-art `pyannote/speaker-diarization-3.1` pipeline for core diarization tasks and includes custom modules for audio preprocessing, speaker change detection, interactive visualization of results, and robust evaluation using Diarization Error Rate (DER).

The system is designed for modularity, reproducibility, and ease of use, with full Docker support.

## Features

### Core Diarization

*   **Audio Preprocessing:** Handles loading, resampling (to 16kHz), mono conversion, and RMS normalization of audio files for optimal model input.
*   **State-of-the-Art Diarization:** Utilizes the pre-trained `pyannote/speaker-diarization-3.1` pipeline from Hugging Face, which integrates VAD, speaker embedding, and clustering.
*   **Output Formats:** Generates diarization results in standard RTTM (Rich Transcription Time Mark) and CSV formats.

### Bonus Features Implemented

*   **Configurable Pipeline:** Allows runtime configuration of key diarization parameters such as minimum and maximum number of speakers, and clustering threshold.
*   **Overlapping Speech Detection:** The chosen `pyannote` pipeline intrinsically handles and accounts for overlapping speech segments.
*   **Speaker Change Detection:** A dedicated module to identify and report exact timestamps where speaker turns occur within the diarized audio.
*   **Interactive Visualization:** Generates visual plots of the audio waveform overlaid with speaker segments, offering clear insight into the diarization output. Supports both live diarization plotting and plotting from existing RTTM files.
*   **Robust Evaluation:** Computes the Diarization Error Rate (DER) against ground truth RTTM files, providing quantitative performance metrics.
*   **Docker Support:** The entire system is containerized using Docker, ensuring high reproducibility and simplified deployment across different environments.

## Project Structure
.
├── data/
│ └── sample_audio.wav # Example input audio file (add your own)
│ └── sample_audio.rttm # Corresponding ground truth RTTM for evaluation (add your own)
├── outputs/ # Directory for diarization output (CSV, RTTM) - created automatically
├── plots/ # Directory for visualization plots (PNG) - created automatically
├── src/
│ ├── preprocess.py # Audio preprocessing module
│ ├── diarize.py # Main diarization script (uses pyannote pipeline)
│ ├── speaker_change_detection.py # Speaker change detection module
│ ├── visualize.py # Speaker diarization visualization module
│ └── evaluate.py # Diarization evaluation module (calculates DER)
├── archive/ # (Optional) Contains older or experimental code (e.g., vad.py, embedding.py if not used)
├── Dockerfile # Dockerfile for containerization
├── requirements.txt # Python dependencies
└── README.md # This file

## Project Architecture / System Flow


## Setup and Installation

### Prerequisites

*   **Git:** For cloning the repository.
*   **Python 3.8+:** Recommended version.
*   **pip:** Python package installer.
*   **Docker Desktop (Optional but Recommended):** For containerized setup.

### 1. Clone the Repository

git clone https://github.com/D4X-max/Speaker-Diarization-VAD-CAS.git
cd Speaker-Diarization-VAD-CAS


### 2. Hugging Face Authentication Token

This project uses models from Hugging Face Hub (specifically `pyannote/speaker-diarization-3.1`). To download and use these models, you need a Hugging Face authentication token.

1.  **Generate a Token:** Go to [Hugging Face Settings -> Access Tokens](https://huggingface.co/settings/tokens) and create a new token with "read" role. Copy this token.
2.  **Accept User Conditions:** **This is CRUCIAL.** You must individually accept the user conditions for the models used by the `pyannote` pipeline. Visit the following links and click on "Agree and Access repository" for each:
    *   [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
    *   [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
    *   [pyannote/embedding](https://huggingface.co/pyannote/embedding)
    *(Note: The `speaker-diarization-3.1` model might internally pull `segmentation-3.0` and `embedding`. Accepting conditions for all three is a good practice.)*

### 3. Local Setup (Recommended if not using Docker)

1.  **Create a Virtual Environment (Optional but Recommended):**
    ```
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

2.  **Install Dependencies:**
    ```
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` is not yet generated, you can do so by running `pip freeze > requirements.txt` after installing all necessary libraries manually, or manually create it with libraries like `librosa`, `numpy`, `soundfile`, `pyannote.audio`, `pyannote.core`, `pyannote.metrics`, `matplotlib`, `torch`, etc.)*

### 4. Docker Setup (Alternative, Highly Recommended for Reproducibility)

1.  **Build the Docker Image:**
    This command will build the Docker image based on the `Dockerfile` in the project root.
    ```
    docker build -t speaker-diarization-app .
    ```
    *(Note: The first time you run a diarization command with Docker, pyannote will download the models to the container's volume. Subsequent runs will be faster.)*

## Usage

This project provides several scripts for different functionalities. All scripts should be run from the project's root directory.

### 1. Running Speaker Diarization

The main diarization script processes an audio file and outputs RTTM and CSV files.

python src/diarize.py <path_to_input_audio> --auth_token <your_hugging_face_token> 

[OPTIONS]


*   **`<path_to_input_audio>`:** Path to your `.wav` or `.flac` audio file (e.g., `data/sample_audio.wav`).
*   **`--auth_token <your_hugging_face_token>`:** Your Hugging Face authentication token (required).

**Options:**

*   `--min_speakers <int>`: Minimum number of speakers expected (e.g., `2`).
*   `--max_speakers <int>`: Maximum number of speakers expected (e.g., `5`).
*   `--clustering_threshold <float>`: Threshold for clustering speaker embeddings (e.g., `0.7`).

**Example:**

python src/diarize.py data/sample_audio.wav --auth_token hf_YOUR_TOKEN_HERE --min_speakers 1 --max_speakers 3


**Output:**
Diarization results will be saved in the `outputs/` directory:
*   `outputs/<audio_filename_base>_diarization.csv`
*   `outputs/<audio_filename_base>_diarization.rttm`

### 2. Detecting Speaker Changes

This script runs the diarization pipeline and then identifies and reports the timestamps of speaker changes.

python src/speaker_change_detection.py <path_to_input_audio> --auth_token <your_hugging_face_token> 

[OPTIONS]


*   **`<path_to_input_audio>`:** Path to your `.wav` or `.flac` audio file.
*   **`--auth_token <your_hugging_face_token>`:** Your Hugging Face authentication token (required).

**Options:** (Same as `diarize.py` for passing to the diarization pipeline)

**Example:**

python src/speaker_change_detection.py data/sample_audio.wav --auth_token hf_YOUR_TOKEN_HERE


**Output:**
In addition to the diarization files in `outputs/`, speaker change points will be printed to the console.

### 3. Visualizing Diarization Results

This script generates a visual plot of the diarization results overlaid on the audio waveform.

python src/visualize.py <path_to_input_audio> 

[OPTIONS]


*   **`<path_to_input_audio>`:** Path to your `.wav` or `.flac` audio file.

**Options:**

*   `--rttm_file <path_to_rttm>`: **Optional.** Path to an existing RTTM file (e.g., `outputs/sample_audio_diarization.rttm`). If provided, the script will plot this RTTM instead of performing live diarization.
*   `--auth_token <your_hugging_face_token>`: **Required if `--rttm_file` is NOT used.** Your Hugging Face authentication token for live diarization.
*   `--min_speakers`, `--max_speakers`, `--clustering_threshold`: (For live diarization only) Same options as `diarize.py`.
*   `--output_dir <dir_path>`: Directory to save the plot (default: `plots/`).
*   `--output_name <filename>`: Name of the output PNG file (default: `audio_filename_base_diarization.png`).
*   `--plot_width <float>`, `--plot_height <float>`: Dimensions of the output plot in inches.

**Examples:**

*   **Live Diarization and Plotting:**
    ```
    python src/visualize.py data/sample_audio.wav --auth_token hf_YOUR_TOKEN_HERE --output_dir my_plots --output_name my_diarization_plot.png
    ```
*   **Plotting from an Existing RTTM:**
    ```
    python src/visualize.py data/sample_audio.wav --rttm_file outputs/sample_audio_diarization.rttm --output_dir my_plots
    ```

**Output:**
A PNG image of the diarization plot will be saved in the specified output directory (default: `plots/`).

### 4. Evaluating Diarization Performance

This script calculates the Diarization Error Rate (DER) between a reference (ground truth) RTTM and a hypothesis (system output) RTTM.

python src/evaluate.py <path_to_reference_rttm> <path_to_hypothesis_rttm> 

[OPTIONS]


*   **`<path_to_reference_rttm>`:** Path to your ground truth RTTM file (e.g., `data/sample_audio.rttm`).
*   **`<path_to_hypothesis_rttm>`:** Path to the system-generated RTTM file (e.g., `outputs/sample_audio_diarization.rttm`).

**Options:**

*   `--output_json <path_to_json>`: Path to save a JSON file containing detailed DER results (e.g., `der_results.json`).

**Example:**

python src/evaluate.py data/sample_audio.rttm outputs/sample_audio_diarization.rttm --output_json outputs/sample_audio_der.json


**Output:**
DER results will be printed to the console and optionally saved to a JSON file.

---

## Running with Docker (For Consistent Environments)

You can run all the above commands within a Docker container to ensure a consistent environment and avoid local dependency conflicts.

1.  **Build the Docker Image** (if you haven't already):
    ```
    docker build -t speaker-diarization-app .
    ```

2.  **Run Commands via Docker:**
    To run any of the Python scripts, you'll use `docker run` and mount your project directory into the container. This allows the container to access your `data/` and `src/` directories and save outputs to your host machine.

    **General Docker Run Command Structure:**
    ```
    docker run --rm \
      -v "$(pwd)":/app \
      -e HF_AUTH_TOKEN="your_hugging_face_token" \ # Pass token as environment variable
      speaker-diarization-app \
      python /app/src/<script_name>.py /app/<path_to_input_audio_in_container> [OPTIONS]
    ```

    **Important:** Replace `$(pwd)` with `%cd%` on Windows in Command Prompt or `$PWD` in PowerShell. Ensure your host machine's `data/` and any generated `outputs/` or `plots/` directories are mapped correctly.

    **Example: Diarize with Docker:**
    ```
    docker run --rm \
      -v "$(pwd)":/app \
      -e HF_AUTH_TOKEN="hf_YOUR_TOKEN_HERE" \
      speaker-diarization-app \
      python /app/src/diarize.py /app/data/sample_audio.wav --min_speakers 1 --max_speakers 2
    ```

    **Example: Visualize with Docker (from existing RTTM):**
    ```
    docker run --rm \
      -v "$(pwd)":/app \
      speaker-diarization-app \
      python /app/src/visualize.py /app/data/sample_audio.wav --rttm_file /app/outputs/sample_audio_diarization.rttm --output_dir /app/plots
    ```
    *(Note: When running visualization without an RTTM file (i.e., live diarization), you'll need to pass the `--auth_token` argument directly to the python command, instead of the `HF_AUTH_TOKEN` environment variable).*

    **Example: Evaluate with Docker:**
    ```
    docker run --rm \
      -v "$(pwd)":/app \
      speaker-diarization-app \
      python /app/src/evaluate.py /app/data/sample_audio.rttm /app/outputs/sample_audio_diarization.rttm --output_json /app/outputs/sample_audio_der.json
    ```

## Contributing
Feel free to fork the repository, open issues, or submit pull requests.

## License

(Optional section, e.g., MIT License)
[License information here]

## Acknowledgements

*   **pyannote.audio:** For providing the robust speaker diarization pipeline.
*   **Hugging Face:** For hosting the models and providing authentication services.
*   **Other libraries:** librosa, numpy, matplotlib, soundfile, torch, etc.

