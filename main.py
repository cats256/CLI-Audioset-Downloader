import argparse
import csv
import logging
import os
import re
import shutil
from dataclasses import dataclass
from functools import partial
from multiprocessing import get_context
from typing import List, Optional

import psutil
import tqdm
import yt_dlp

@dataclass
class Segment:
    ytid: str
    start_seconds: float
    end_seconds: float
    positive_labels: List[str]

temp_dir = "temp_audioset_files"
num_logical_cpus = psutil.cpu_count(logical=True)
sample_rate=16000
codec = 'wav'

# TODO:
# Make num logical cpu, sample_rate, and codec a CLI option
# Add other kwargs support for yt-dlp
# Add cookie support
# Add requirements.txt
# Turn this into a package

def get_splits_from_args():
    ALLOWED_SPLITS = {'eval', 'balanced_train', 'unbalanced_train'}

    parser = argparse.ArgumentParser(description="Get dataset splits from command line.")
    parser.add_argument(
        "--splits", 
        nargs="+",
        required=True,
        help="List of dataset splits (choices: eval, balanced_train, unbalanced_train). If not provided, download all splits"
    )
    parser.add_argument(
        "--labels", 
        nargs="+",
        required=True,
        help="List of labels to filter by. If not provided, downloads everything"
    )

    args = parser.parse_args()

    for split in args.splits:
        if split not in ALLOWED_SPLITS:
            parser.error(f"Invalid split '{split}'. Allowed values are: {', '.join(ALLOWED_SPLITS)}")

    return args.splits, args.labels

def download_audioset(segment: Segment, split: str) -> None:
    outpath = os.path.join('wav_files', split)
    output_file = os.path.join(outpath, f"{segment.ytid}.wav")

    if os.path.isfile(output_file):
        return None
    
    os.makedirs(outpath, exist_ok=True)
    log_file = os.path.join('wav_files', f"{split}_yt_dlp.log")

    logging.basicConfig(filename=log_file, level=logging.INFO)
    ytdl_logger = logging.getLogger("yt-dlp")

    url = f'https://www.youtube.com/watch?v={segment.ytid}'
    ydl_opts = {
        'logger': ytdl_logger,
        'quiet': True,
        'noprogress': True,
        'ignoreerrors': True,
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': codec
        }],
        'postprocessor_args': ['-ar', str(sample_rate)],
        'external_downloader':'ffmpeg',
        'external_downloader_args': ['-ss', str(segment.start_seconds), '-to', str(segment.end_seconds), '-loglevel', 'quiet'],
        'outtmpl': f"{temp_dir}/{segment.ytid}"
    }

    try:
        if os.path.isfile(os.path.join(outpath, f'{segment.ytid}.{codec}')):
            return None

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        downloaded_file = os.path.join(temp_dir, f"{segment.ytid}.{codec}")

        if os.path.exists(downloaded_file):
            shutil.move(downloaded_file, output_file)
        else:
            ytdl_logger.error(f"Failed to download {url}: File not found in {temp_dir}")
    except Exception as e:
        ytdl_logger.error(f"Failed to download {url}: {e}")

def download_audioset_split(split: str, allowed_labels: Optional[List[str]] = None):
    os.makedirs(os.path.join('wav_files', split), exist_ok=True)
    file = open(f'csv_files/{split}_segments.csv', 'r').read()

    with open(f'csv_files/{split}_segments.csv', 'r', newline='') as file:
        reader = csv.reader(file)
        segments: List[Segment] = [
            Segment(ytid=row[0], start_seconds=float(row[1]), end_seconds=float(row[2]), positive_labels=row[3:])  
            for row in list(reader)[3:-1]
        ]
    
    for segment in segments:
        segment.positive_labels = [positive_label.replace('"','').strip() for positive_label in segment.positive_labels]

    if allowed_labels:
        allowed_labels = set(label for label in allowed_labels)

        def label_filter(segment):
            return any(label in allowed_labels for label in segment.positive_labels)

        segments = [seg for seg in segments if label_filter(seg)]

    pool = get_context("spawn").Pool(num_logical_cpus * 4)
    download_audio_split = partial(download_audioset, split=split)

    with tqdm.tqdm(total=len(segments), leave=False) as pbar:
        for _ in pool.imap_unordered(download_audio_split, segments):
            pbar.update()

if __name__ == "__main__":
    shutil.rmtree(temp_dir, ignore_errors=True)
    os.makedirs(temp_dir, exist_ok=True)

    lines = open('csv_files/class_labels_indices.csv').readlines()[1:]
    lines = [line.split(',') for line in lines]

    mids = [line[1] for line in lines]
    labels = [line[2][1:-2] for line in lines]

    mid_label_dict = dict(zip(labels, mids))

    splits, allowed_labels = get_splits_from_args()
    print("Using splits:", splits)
    print(f"Filtering by labels: {allowed_labels}" if allowed_labels else "Downloading all labels")
    print()

    allowed_labels = [mid_label_dict[label] for label in allowed_labels]

    for split in splits:
        print(f'Downloading {split}')
        outpath = os.path.join('wav_files', split)
        os.makedirs(outpath, exist_ok=True)

        download_audioset_split(split, allowed_labels)
        print(f'Finished downloading {split}')
        print()

    shutil.rmtree(temp_dir, ignore_errors=True)
