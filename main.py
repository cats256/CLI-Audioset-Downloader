import argparse
import logging
import os
from dataclasses import dataclass
from functools import partial
from multiprocessing import get_context
from typing import List, Optional, Set
from pathvalidate import sanitize_filename

import psutil
import tqdm
import yt_dlp

# TODO:
# Make num logical cpu, sample_rate, force mono audio, and codec type a CLI option
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

with open('csv_files/class_labels_indices.csv') as file:
    next(file)
    rows = [line.strip().split(',', 2) for line in file]
    label_mid_dict = {row[2].strip().strip('"'): row[1] for row in rows}
    mid_label_dict = {row[1]: row[2].strip().strip('"') for row in rows}

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

def download_audioset(segment: Segment, outpath) -> None:
    output_file = os.path.join(outpath, "unorganized", f"{segment.ytid}.{codec}")

    if os.path.isfile(output_file):
        for mid in segment.positive_labels:
            label_dir = os.path.join(outpath, "organized", sanitize_filename(mid_label_dict[mid], replacement_text="_"))
            os.makedirs(label_dir, exist_ok=True)

            hard_link_path = os.path.join(label_dir, f"{segment.ytid}.{codec}")

            if not os.path.exists(hard_link_path):
                os.link(output_file, hard_link_path)            
        return None
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

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
        'postprocessor_args': ['-ar', str(sample_rate), '-ac', '1'],
        'download_ranges': yt_dlp.utils.download_range_func(None, [(segment.start_seconds, segment.end_seconds)]),
        'outtmpl': os.path.splitext(output_file)[0],
        'force_keyframes_at_cuts': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        for mid in segment.positive_labels:
            label_dir = os.path.join(outpath, "organized", mid_label_dict[mid])
            os.makedirs(label_dir, exist_ok=True)
            
            hard_link_path = os.path.join(label_dir, f"{segment.ytid}.{codec}")

            if not os.path.exists(hard_link_path):
                os.link(output_file, hard_link_path)
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")

def download_audioset_split(split: str, allowed_labels: Optional[Set[str]] = None) -> None:
    print(f'Downloading {split}')
    outpath = os.path.join(f'{codec}_files', split)
    os.makedirs(outpath, exist_ok=True)

    with open(f'csv_files/{split}_segments.csv', 'r', newline='') as file:
        lines = file.readlines()[3:-1]
        segments: List[Segment] = []

        for i, line in enumerate(lines):
            parts = line.strip().split(',', 3)  
            if len(parts) < 4:
                logger.error(f"Malformed line {i}: {line}")
                continue
            
            segments.append(
                Segment(
                    ytid=parts[0].strip(),
                    start_seconds=float(parts[1]),
                    end_seconds=float(parts[2]),
                    positive_labels=parts[3].strip().strip('"').split(",")
                )
            )
    
    if allowed_labels:
        def label_filter(segment):
            return any(label in allowed_labels for label in segment.positive_labels)

        segments = [seg for seg in segments if label_filter(seg)]

    with get_context("spawn").Pool(num_logical_cpus * 2) as pool:
        download_audio_split = partial(download_audioset, outpath=outpath)

        for _ in tqdm.tqdm(pool.imap_unordered(download_audio_split, segments), total=len(segments), leave=False):
            pass

    print(f'Finished downloading {split}')
    print()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    splits, allowed_labels = args.splits, args.labels

    print("Using splits:", splits)
    print(f"Filtering by labels: {allowed_labels}" if allowed_labels else "Downloading all labels")
    print()

    if allowed_labels:
        allowed_labels = set([label_mid_dict[label] for label in allowed_labels])

    for split in splits:
        download_audioset_split(split, allowed_labels)