import os
import json
from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp

# to run this file first open cmd and run
# pip install numpy pandas librosa youtube_transcript_api yt-dlp


def download_transcript(video_id, output_path, speaker_name, language='en'):
    json_folder = os.path.join(output_path, 'json')

    os.makedirs(json_folder, exist_ok=True)

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        
        json_path = os.path.join(json_folder, f"{speaker_name}.json")

        with open(json_path, 'w', encoding='utf-8') as json_file:
            json.dump(transcript, json_file, indent=2, ensure_ascii=False)

        print(f"Transcript downloaded successfully for {speaker_name}.")
        return True
    except Exception as e:
        print(f"Failed to download transcript for {speaker_name}: {e}")
        return False

def download_audio(video_id, output_path, speaker_name, ffmpeg_path=None):
    audio_folder = os.path.join(output_path, 'audios')
    os.makedirs(audio_folder, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(audio_folder, f"{speaker_name}.%(ext)s"),
    }

    if ffmpeg_path:
        ydl_opts['ffmpeg_location'] = ffmpeg_path

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://youtu.be/{video_id}"])
        print(f"Audio downloaded successfully for {speaker_name}.")
        return True
    except Exception as e:
        print(f"Failed to download audio for {speaker_name}: {e}")
        return False

def process_video_links(input_file, output_path, language='en', ffmpeg_path=None, start = 100):
    with open(input_file, 'r', encoding='utf-8') as file:
        video_urls = [line.strip() for line in file.readlines() if line.strip()]

    for index, video_url in enumerate(video_urls, start=start):
        video_id = video_url.split('v=')[-1].split('&')[0] if 'v=' in video_url else video_url.split('/')[-1].split('?')[0]
        speaker_name = f"speaker_{index + 1}"
        print(f"Processing {speaker_name} with video ID: {video_id}")

        transcript_success = download_transcript(video_id, output_path, speaker_name, language=language)
        audio_success = download_audio(video_id, output_path, speaker_name, ffmpeg_path=ffmpeg_path)

        if transcript_success and audio_success:
            print(f"Successfully processed {speaker_name}.\n")
        else:
            print(f"Failed to process {speaker_name}.\n")

if __name__ == "__main__":
    input_file = "links\links.txt"
    output_path = "downloads"  # Root output folder
    language = "bn"  # Default language for transcripts
    ffmpeg_path = r"C:\\ffmpeg"  # Path to ffmpeg executable
    start = int(input("Enter starting speaker ID: ")) - 1

    process_video_links(input_file, output_path, language=language, ffmpeg_path=ffmpeg_path, start = start)
