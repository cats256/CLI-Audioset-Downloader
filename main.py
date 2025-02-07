import argparse
import logging
import os
from dataclasses import dataclass
from functools import partial
from multiprocessing import get_context
from typing import List, Optional

import psutil
import tqdm
import yt_dlp

# TODO:
# Make num logical cpu, sample_rate, and codec a CLI option
# Add other kwargs support for yt-dlp
# Add requirements.txt
# Turn this into a package

@dataclass
class Segment:
    ytid: str
    start_seconds: float
    end_seconds: float
    positive_labels: List[str]

num_logical_cpus = psutil.cpu_count(logical=True)
sample_rate=16000
codec = 'wav'

log_file = os.path.join(os.getcwd(), "audioset_download.log")
if not os.path.exists(log_file):
    open(log_file, 'w').close()

logging.basicConfig(filename=log_file, level=logging.INFO)
logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser(description="Get dataset splits from command line.")
    parser.add_argument(
        "--splits", 
        nargs="+",
        required=True,
        choices=['eval', 'balanced_train', 'unbalanced_train'],
        help="List of dataset splits (choices: eval, balanced_train, unbalanced_train)."
    )
    parser.add_argument(
        "--labels", 
        nargs="+",
        default=None,
        help="List of labels to filter by. If not provided, downloads everything."
    )
    return parser

def download_audioset(segment: Segment, split: str) -> None:
    outpath = os.path.join(f'{codec}_files', split)
    output_file = os.path.join(outpath, f"{segment.ytid}.{codec}")

    if os.path.isfile(output_file):
        return None
    
    os.makedirs(outpath, exist_ok=True)

    url = f'https://www.youtube.com/watch?v={segment.ytid}'
    ydl_opts = {
        'logger': logger,
        'quiet': True,
        'noprogress': True,
        'ignoreerrors': True,
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': codec,
            'nopostoverwrites': True
        }],
        'postprocessor_args': ['-ar', str(sample_rate)],
        'download_ranges': yt_dlp.utils.download_range_func([], [[segment.start_seconds, segment.end_seconds]]),
        'outtmpl': f"{outpath}/{segment.ytid}"
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")

def download_audioset_split(split: str, allowed_labels: Optional[List[str]] = None):
    print(f'Downloading {split}')
    os.makedirs(os.path.join(f'{codec}_files', split), exist_ok=True)

    with open(f'csv_files/{split}_segments.csv', 'r', newline='') as file:
        lines = file.readlines()[3:-1]
        segments: List[Segment] = []

        for i, line in enumerate(lines):
            parts = line.strip().split(',', 3)  
            if len(parts) < 4:
                logger.error(f"Malformed line {i}: {line}")
                continue
            
            ytid, start_seconds, end_seconds, positive_labels = parts
            segments.append(
                Segment(
                    ytid=ytid.strip(),
                    start_seconds=float(start_seconds),
                    end_seconds=float(end_seconds),
                    positive_labels=positive_labels.strip().strip('"').split(",")
                )
            )
    
    if allowed_labels:
        allowed_labels = set(allowed_labels)

        def label_filter(segment):
            return any(label in allowed_labels for label in segment.positive_labels)

        segments = [seg for seg in segments if label_filter(seg)]

    with get_context("spawn").Pool(num_logical_cpus * 2) as pool:
        download_audio_split = partial(download_audioset, split=split)

        for _ in tqdm.tqdm(pool.imap_unordered(download_audio_split, segments), total=len(segments), leave=False):
            pass

    print(f'Finished downloading {split}')
    print()

if __name__ == "__main__":
    with open('csv_files/class_labels_indices.csv') as file:
        next(file)
        mid_label_dict = {line.split(',')[2].strip().strip('"'): line.split(',')[1] for line in file}

    parser = get_parser()
    args = parser.parse_args()
    splits, allowed_labels = args.splits, args.labels

    print("Using splits:", splits)
    print(f"Filtering by labels: {allowed_labels}" if allowed_labels else "Downloading all labels")
    print()

    if allowed_labels:
        allowed_labels = [mid_label_dict[label] for label in allowed_labels]

    for split in splits:
        download_audioset_split(split, allowed_labels)