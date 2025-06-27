# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required by librosa and pyannote.audio
# libsndfile-dev for soundfile, ffmpeg for librosa
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
# This step is done separately to leverage Docker's caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project code into the container
COPY . .

# Set default environment variable for Hugging Face token (can be overridden at runtime)
# It's better to pass this securely via --env or .env file, but this shows the intent.
# ENV HF_HOME="/root/.cache/huggingface" # Default cache location if needed

# Expose a default command, but we'll override it when running
CMD ["python", "--version"]
