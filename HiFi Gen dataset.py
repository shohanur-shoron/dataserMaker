import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import librosa
import random

# to run this files first open cmd and run
# pip install numpy pandas librosa youtube_transcript_api yt-dlp

# ==================== Configuration ====================

# Specify the path to the ffmpeg executable
FFMPEG_PATH = r"C:\\ffmpeg\\ffmpeg.exe" 

# Set the directories
AUDIO_INPUT_DIR = "audios"
JSON_INPUT_DIR = "jsons"
OUTPUT_DIR = "hifi_gan_dataset"
AUDIO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "audio")
MEL_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "mel")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

# Create output directories if they don't exist
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)
os.makedirs(MEL_OUTPUT_DIR, exist_ok=True)

# Supported audio formats
SUPPORTED_AUDIO_FORMATS = ['.mp3', '.wav', '.flac', '.m4a', '.aac']

# ==================== Helper Functions ====================


def create_training_and_validation_files(csv_path, 
                                         training_output='hifi_gan_dataset\\training.txt', 
                                         validation_output='hifi_gan_dataset\\validation.txt', 
                                         train_ratio=0.9):
    """
    Reads a CSV file where each line has the format:
        path_to_mel|path_to_audio
    For example:
        hifi_gan_dataset/mel/speaker_8_segment_262.npy|hifi_gan_dataset/audio/speaker_8_segment_262.wav

    Then splits the file names (without extensions) into training and validation files.
    The train/val split ratio is by default 90% / 10%.
    """
    
    # Read all lines from the CSV
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    
    # Extract the base names without extensions
    data = []
    for line in lines:
        # Split by the '|' delimiter
        left_path, right_path = line.split('|')
        
        # The left_path (e.g. "hifi_gan_dataset/mel/speaker_8_segment_262.npy")
        # We'll only extract one name (both left and right refer to the same file base).
        filename = left_path.split('/')[-1]  # "speaker_8_segment_262.npy"
        base_name = filename.split('.')[0]   # "speaker_8_segment_262"
        
        data.append(base_name)
    
    # Shuffle data for a more random split
    random.shuffle(data)
    
    # Determine split index
    split_index = int(len(data) * train_ratio)
    
    # Split into training and validation
    training_data = data[:split_index]
    validation_data = data[split_index:]
    
    # Write training data to file
    with open(training_output, 'w', encoding='utf-8') as train_file:
        for item in training_data:
            train_file.write(f"{item}\n")
    
    # Write validation data to file
    with open(validation_output, 'w', encoding='utf-8') as val_file:
        for item in validation_data:
            val_file.write(f"{item}\n")


def convert_and_slice_audio(ffmpeg_path, input_audio_path, start_time, duration, output_wav_path):
    """
    Converts and slices the audio segment using ffmpeg.

    Args:
        ffmpeg_path (str): Path to the ffmpeg executable.
        input_audio_path (str): Path to the input audio file.
        start_time (float): Start time in seconds.
        duration (float): Duration in seconds.
        output_wav_path (str): Path to save the output sliced WAV file.
    """
    command = [
        ffmpeg_path,
        "-y",                # Overwrite output files without asking
        "-ss", str(start_time),  # Start time
        "-t", str(duration),     # Duration
        "-i", input_audio_path,  # Input file
        "-ar", "22050",          # Set audio sampling rate to 16kHz
        "-ac", "1",              # Set number of audio channels to mono
        "-acodec", "pcm_s16le",  # Set audio codec to PCM 16-bit little endian
        "-af", "volume=1.0",     # Apply volume leveling (simple normalization)
        output_wav_path          # Output file
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg processing: {e}")
        raise e

def load_json(json_path):
    """
    Loads and parses a JSON file.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        list: Parsed JSON data.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_mel_spectrogram(wav_path, n_mels=80, hop_length=256, win_length=1024):
    """
    Extracts Mel-spectrogram from a WAV file.

    Args:
        wav_path (str): Path to the WAV file.
        n_mels (int): Number of Mel bands to generate.
        hop_length (int): Number of samples between successive frames.
        win_length (int): Each frame of audio is windowed by `window` of length `win_length`.

    Returns:
        np.ndarray: Mel-spectrogram in decibels.
    """
    y, sr = librosa.load(wav_path, sr=16000)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
        win_length=win_length,
        power=1.0  # Use power=1.0 for energy
    )
    mel = librosa.power_to_db(mel, ref=np.max)
    return mel.astype(np.float32)

def sanitize_filename(filename):
    """
    Sanitizes a filename by replacing unwanted characters with underscores.

    Args:
        filename (str): Original filename.

    Returns:
        str: Sanitized filename.
    """
    return "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in filename)

# ==================== Main Processing ====================

def main():
    # Verify ffmpeg path
    if not os.path.isfile(FFMPEG_PATH):
        print(f"FFmpeg not found at specified path: {FFMPEG_PATH}")
        sys.exit(1)

    metadata_entries = []

    # List all audio files in the input directory
    audio_files = [
        file for file in os.listdir(AUDIO_INPUT_DIR)
        if os.path.splitext(file)[1].lower() in SUPPORTED_AUDIO_FORMATS
    ]

    total_audio_files = len(audio_files)
    if total_audio_files == 0:
        print(f"No supported audio files found in {AUDIO_INPUT_DIR}. Exiting.")
        return

    for audio_idx, audio_file in enumerate(audio_files, start=1):
        file_ext = os.path.splitext(audio_file)[1].lower()
        base_name = os.path.splitext(audio_file)[0]
        json_filename = base_name + ".json"

        audio_path = os.path.join(AUDIO_INPUT_DIR, audio_file)
        json_path = os.path.join(JSON_INPUT_DIR, json_filename)

        if not os.path.exists(json_path):
            print(f"JSON file not found for {audio_file}, skipping.")
            continue

        # Extract speaker label from filename (assuming format: speaker_1_audio_001)
        speaker_label_parts = base_name.split('_')
        if len(speaker_label_parts) >= 2:
            speaker_label = f"{speaker_label_parts[0]}_{speaker_label_parts[1]}"
        else:
            speaker_label = speaker_label_parts[0]

        # Load JSON data
        try:
            json_data = load_json(json_path)
        except Exception as e:
            print(f"Error loading JSON file {json_filename}: {e}")
            continue

        total_entries = len(json_data)
        if total_entries == 0:
            print(f"No segments found in {json_filename}, skipping.")
            continue

        print(f"\nProcessing '{audio_file}' ({audio_idx}/{total_audio_files}) with {total_entries} segments.")

        for idx, segment in enumerate(json_data, start=1):
            text = segment.get("text", "").strip()
            start = segment.get("start", 0.0)
            duration = segment.get("duration", 0.0)

            if duration <= 0:
                print(f"\nInvalid duration for segment {idx} in {json_filename}, skipping.")
                continue

            # Generate unique filenames
            sanitized_base = sanitize_filename(base_name)
            segment_filename = f"{sanitized_base}_segment_{idx:03d}.wav"
            segment_audio_path = os.path.join(AUDIO_OUTPUT_DIR, segment_filename)

            # Convert and slice audio using ffmpeg
            try:
                convert_and_slice_audio(
                    ffmpeg_path=FFMPEG_PATH,
                    input_audio_path=audio_path,
                    start_time=start,
                    duration=duration,
                    output_wav_path=segment_audio_path
                )
            except Exception as e:
                print(f"\nError processing segment {idx} of {audio_file}: {e}")
                continue

            # Extract Mel-spectrogram
            try:
                mel = extract_mel_spectrogram(segment_audio_path)
            except Exception as e:
                print(f"\nError extracting Mel-spectrogram for {segment_filename}: {e}")
                continue

            # Save Mel-spectrogram
            mel_filename = f"{sanitized_base}_segment_{idx:03d}.npy"
            mel_path = os.path.join(MEL_OUTPUT_DIR, mel_filename)
            try:
                np.save(mel_path, mel)
            except Exception as e:
                print(f"\nError saving Mel-spectrogram {mel_filename}: {e}")
                continue

            # Add entry to metadata
            mel_relative = os.path.relpath(mel_path, OUTPUT_DIR)
            audio_relative = os.path.relpath(segment_audio_path, OUTPUT_DIR)
            metadata_entry = f"hifi_gan_dataset/mel/{mel_filename}|hifi_gan_dataset/audio/{segment_filename}"
            metadata_entries.append(metadata_entry)

            # Calculate progress
            progress_percentage = (idx / total_entries) * 100
            sys.stdout.write(f"\rProcessing {audio_file} -> {speaker_label}: "
                             f"Progress: {progress_percentage:.2f}% | Creating clip {idx}/{total_entries}")
            sys.stdout.flush()

        print()  # Move to the next line after processing all segments of the current audio file

    # Save metadata.csv
    try:
        with open(METADATA_PATH, 'w', encoding='utf-8') as meta_file:
            for entry in metadata_entries:
                meta_file.write(entry + "\n")
        print(f"\nMetadata saved to {METADATA_PATH}")
        create_training_and_validation_files(METADATA_PATH)
    except Exception as e:
        print(f"Error saving metadata.csv: {e}")

if __name__ == "__main__":
    main()
