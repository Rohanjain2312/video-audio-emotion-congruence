"""
Download RAVDESS video files specifically.
Run this if you only have audio files and need videos.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.download_datasets import download_ravdess_video, verify_datasets, IS_GPU_MODE

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download RAVDESS video files")
    parser.add_argument("--data_root", type=str, default="./data/raw",
                      help="Root directory for raw data")
    parser.add_argument("--cpu_mode", action="store_true",
                      help="CPU mode (only 2 actors)")
    
    args = parser.parse_args()
    
    cpu_mode = args.cpu_mode or (not IS_GPU_MODE)
    
    print(f"Mode: {'CPU (2 actors)' if cpu_mode else 'GPU (24 actors)'}")
    
    # Download videos
    download_ravdess_video(args.data_root, cpu_mode=cpu_mode)
    
    # Verify
    verify_datasets(args.data_root)