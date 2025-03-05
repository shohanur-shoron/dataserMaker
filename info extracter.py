import os
import json
from collections import Counter
import string
from pathlib import Path

# to run this files first open cmd and run
# pip install numpy pandas librosa youtube_transcript_api yt-dlp

def process_json_files(folder_path, output_file):
    duration = 0
    total_lines = 0
    max_duration = float('-inf')
    min_duration = float('inf')
    
    texts = []
    
    for file_path in Path(folder_path).glob('*.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                if not isinstance(data, list):
                    print(f"Unexpected format in file: {file_path}")
                    continue
                
                duration += data[-1]['start'] + data[-1]['duration']
                
                for item in data:
                    max_duration = max(max_duration, item['duration'])
                    min_duration = min(min_duration, item['duration'])
                    if 'text' in item:
                        texts.append(item['text'])
                        total_lines += 1
                        
        except (json.JSONDecodeError, Exception) as e:
            print(f"Error processing {file_path}: {e}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(texts))
        
    return duration, total_lines, max_duration, min_duration

def analyze_words(file_path, table_file):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    
    words = text.lower().split()
    word_counts = Counter(words)
    
    with open(table_file, 'w', encoding='utf-8') as f:
        for word, count in word_counts.most_common():
            f.write(f"{word}: {count}\n")
    
    return len(words), len(word_counts)
    
    
def format_duration(total_seconds):
    """
    Convert an integer number of seconds into a
    human-readable string: Hh Mm Ss (e.g. "2h 34m 3s").
    Handles hours, minutes, and seconds.
    """
    total_seconds = int(total_seconds)
    hours = total_seconds // 3600
    remainder = total_seconds % 3600
    minutes = remainder // 60
    seconds = remainder % 60

    parts = []
    if hours > 0:
        parts.append(f"{int(hours)}h")
    if minutes > 0:
        parts.append(f"{int(minutes)}m")
    if seconds > 0:
        parts.append(f"{int(seconds)}s")

    # If total_seconds = 0 (shouldn't happen if min is 1),
    # or if there's no hours/minutes, we still return something.
    if not parts:
        return "0s"

    return " ".join(parts)

def main():
    folder = 'jsons'  # the folder name where all the json files are
    output_folder = 'info'
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, 'output.txt')
    table_file = os.path.join(output_folder, 'table.txt')
    
    duration, total_lines, max_duration, min_duration = process_json_files(folder, output_file)
    
    total_words, distinct_words = analyze_words(output_file, table_file)
    
    print('*' * 60)
    print()
    
    print(f"\033[32mTotal duration: {format_duration(duration)}\033[0m")
    print(f"\033[32mTotal Clips: {total_lines}\033[0m")
    print(f"\033[32mAvg Duration: {duration/total_lines:.4f}s\033[0m")
    print(f"\033[32mMax Duration: {max_duration}s\033[0m")
    print(f"\033[32mMin Duration: {min_duration}s\033[0m")
    print(f"\033[32mTotal words: {total_words}\033[0m")
    print(f"\033[32mTotal distinct words: {distinct_words}\033[0m")
    print()
    print('*' * 60)

if __name__ == "__main__":
    main()