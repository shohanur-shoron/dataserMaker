import os
import json
import subprocess
import csv
import sys

# to run this files first open cmd and run
# pip install numpy pandas librosa youtube_transcript_api yt-dlp

def combine_text(data):
    """
    Merges each segment's text with the next one.
    Example:
      Original: [ (text="A"), (text="B"), (text="C") ]
      Combined: [ (text="A B"), (text="B C"), (text="C") ]
    """
    combined_data = []

    # Combine adjacent segments
    for i in range(len(data) - 1):
        new_text = data[i]['text'] + " " + data[i + 1]['text']
        new = {
            "text": new_text,
            "start": data[i]['start'],
            "duration": data[i]['duration']
        }
        combined_data.append(new)

    # Add the last segment as-is
    last = {
        "text": data[-1]['text'],
        "start": data[-1]['start'],
        "duration": data[-1]['duration']
    }
    combined_data.append(last)

    return combined_data


def main():
    # -------------------------------------------------------------------------
    # 1) Configure paths
    # -------------------------------------------------------------------------
    ffmpeg_path = r"C:\\ffmpeg\\ffmpeg.exe"
    audios_dir = "audios"
    jsons_dir = "jsons"
    speakers_dir = "Speakers"
    speaker_index = int(input("Enter speaker ID: "))
    output_csv = os.path.join(speakers_dir, f"metadata_speaker_{speaker_index}.csv")

    # Ensure Speakers folder exists
    os.makedirs(speakers_dir, exist_ok=True)
    
    # Prepare a list to hold all CSV rows
    csv_rows = []

    # This will track the speaker folder index
    
    # ---------------------------------------------------------------------
    # 3) Create the speaker folder
    # ---------------------------------------------------------------------
    speaker_label = f"speaker_{speaker_index}"
    speaker_folder_path = os.path.join(speakers_dir, speaker_label)

    # Check if the directory already exists
    if os.path.exists(speaker_folder_path):
        print("The folder already exists. Check the speaker_label!")
        sys.exit(1)  # Exit with a non-zero status indicating an error/stop
    else:
        # If it doesn't exist, create the directory
        os.makedirs(speaker_folder_path)
    
    print("---------------------------------------------------------------------")
    print("Creating: ", speaker_label)
    print("---------------------------------------------------------------------")
    audio_counter = 1
    # -------------------------------------------------------------------------
    # 2) Loop over audio files in `audios` folder
    # -------------------------------------------------------------------------
    audio_files = sorted(os.listdir(audios_dir))  # sort for consistent ordering, optional

    for audio_file in audio_files:

        audio_name, audio_ext = os.path.splitext(audio_file)

        # Matching JSON filename must match the audio name
        json_file = audio_name + ".json"
        json_path = os.path.join(jsons_dir, json_file)
        if not os.path.isfile(json_path):
            print(f"Warning: JSON file not found for {audio_file}, skipping.")
            continue

        with open(json_path, "r", encoding="utf-8") as f:
            try:
                segments = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON for {json_file}, skipping.")
                continue

        # ---------------------------------------------------------------------
        # 2) Combine segments using the provided function
        # ---------------------------------------------------------------------
        combined_segments = combine_text(segments)
        
        # ---------------------------------------------------------------------
        # 4) Process each combined segment
        # ---------------------------------------------------------------------
        total_entries = len(combined_segments)

        for i, seg in enumerate(combined_segments, start=1):
            # Show progress for this file's segments
            progress_percentage = (i / total_entries) * 100
            sys.stdout.write(f"\rProcessing {audio_file} -> {speaker_label}: "
                             f"Progress: {progress_percentage:.2f}% | Creating clip {i}/{total_entries}")
            sys.stdout.flush()

            # Build the output wav file name
            out_wav_name = f"{speaker_label}_audio_{audio_counter:04d}.wav"
            out_wav_path = os.path.join(speaker_folder_path, out_wav_name)

            audio_counter += 1
            
            # Extract segment info
            start_time = seg.get("start", 0.0)
            duration = seg.get("duration", 0.0)
            transcript = seg.get("text", "")

            # Use ffmpeg to cut and convert
            command = [
                ffmpeg_path,
                "-y",                # overwrite output
                "-ss", str(start_time),
                "-t", str(duration),
                "-i", os.path.join(audios_dir, audio_file),
                "-ar", "22050",      # sample rate
                "-ac", "1",          # channels (mono)
                "-acodec", "pcm_s16le",
                "-af", "volume=1.0", # simple volume leveling
                out_wav_path
            ]
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # Append to CSV data
            # Format: path|transcript|speaker_label
            rel_path = os.path.relpath(out_wav_path)
            csv_rows.append([rel_path, transcript, speaker_label])

        # End of segments for this audio - print a newline
        sys.stdout.write("\n")
        print()
        
    # -------------------------------------------------------------------------
    # 5) Write the metadata CSV
    # -------------------------------------------------------------------------
    with open(output_csv, mode="w", encoding="utf-8", newline="") as f_csv:
        writer = csv.writer(f_csv, delimiter='|', quoting=csv.QUOTE_MINIMAL)
        for row in csv_rows:
            writer.writerow(row)

    print(f"Done! Metadata saved to {output_csv}")

if __name__ == "__main__":
    main()