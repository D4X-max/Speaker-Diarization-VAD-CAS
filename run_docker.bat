@echo off
set IMAGE_NAME="speaker-diarization"

:: --- Build the Docker image ---
echo Building Docker image...
docker build -t %IMAGE_NAME% .
if %errorlevel% neq 0 (
    echo Docker image build failed.
    exit /b 1
)
echo Docker image %IMAGE_NAME% built successfully.

:: --- Run the Diarization ---
echo.
echo Running speaker diarization in Docker...
echo Usage: run_docker.bat run "<audio_file_on_host>" [optional_diarization_args]
echo Example: run_docker.bat run "data\sample_audio.wav" --min_speakers 1 --max_speakers 2
echo.

:: Check if 'run' command is provided and correctly parse arguments
if /I "%1"=="run" (
    shift  :: Shift past the "run" command
    
    :: %~1 expands %1 removing any surrounding quotes. This is crucial for paths with spaces.
    set "AUDIO_FILE_HOST=%~1" 
    shift  :: Shift past the audio file path
    
    :: Now %* contains all remaining optional arguments

    if "%AUDIO_FILE_HOST%"=="" (
        echo Error: Please provide an audio file path on your host machine.
        echo Usage: run_docker.bat run "<audio_file_on_host>" [optional_diarization_args]
        exit /b 1
    )

    :: Get the directory and filename for Docker mounting
    :: Using %%~dpf and %%~nxf to correctly parse path components, even with spaces
    for %%f in ("%AUDIO_FILE_HOST%") do (
        set "AUDIO_DIR_HOST=%%~dpf"
        set "AUDIO_FILENAME=%%~nxf"
    )
    
    :: Remove trailing backslash if it exists (for docker volume mount)
    if "%AUDIO_DIR_HOST:~-1%"=="\" set "AUDIO_DIR_HOST=%AUDIO_DIR_HOST:~0,-1%"

    :: Replace backslashes with forward slashes for Docker container path
    set "AUDIO_DIR_HOST=%AUDIO_DIR_HOST:\=/%"
    
    :: Define local output directory (will be mapped to /app/outputs in container)
    set "LOCAL_OUTPUT_DIR=outputs"
    if not exist "%LOCAL_OUTPUT_DIR%" mkdir "%LOCAL_OUTPUT_DIR%"

    :: Run the Docker container
    docker run ^
        --rm ^
        -v "%AUDIO_DIR_HOST%:/data" ^
        -v "%cd%/%LOCAL_OUTPUT_DIR%:/app/outputs" ^
        -e HF_TOKEN="%HF_TOKEN%" ^
        %IMAGE_NAME% ^
        python src/diarize.py "/data/%AUDIO_FILENAME%" %*

    if %errorlevel% neq 0 (
        echo Docker run command failed.
        exit /b 1
    )
    echo Diarization finished. Results are in the '%LOCAL_OUTPUT_DIR%' directory on your host.
    echo Remember to accept Hugging Face model conditions if this is your first run.
) else (
    echo No 'run' command provided. To run diarization, use: run_docker.bat run "<audio_file_on_host>" [optional_diarization_args]
)

