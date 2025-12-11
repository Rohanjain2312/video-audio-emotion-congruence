"""
Automatic download scripts for RAVDESS and CREMA-D datasets.
Works in both local and Colab environments.
Supports CPU mode (limited data) and GPU mode (full data).
"""

import os
import zipfile
import requests
import torch
import shutil
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm


# Detect device globally
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_GPU_MODE = DEVICE == "cuda"

print(f"Running in {'GPU' if IS_GPU_MODE else 'CPU'} mode")
print(f"Device: {DEVICE}")


def download_file_with_progress(url, destination):
    """Download a file with progress bar."""
    print(f"Downloading to {destination}...")
    
    response = requests.get(url, stream=True, allow_redirects=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)
    
    print(f"Download complete: {destination}")


def check_kaggle_credentials():
    """Verify that the Kaggle API key exists."""
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_path):
        print("\nKaggle API key not found.")
        print("Please place kaggle.json in ~/.kaggle/kaggle.json")
        print("\nInstructions:")
        print("1. Download kaggle.json from https://www.kaggle.com/settings")
        print("2. Run: mkdir -p ~/.kaggle")
        print("3. Run: mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    print("Kaggle API credentials found.")
    return True


def download_ravdess(data_root="./data/raw", cpu_mode=None):
    """
    Download RAVDESS dataset.
    
    Args:
        data_root: Root directory for raw data
        cpu_mode: If True, download limited actors. If False, download all.
                  If None, auto-detect based on GPU availability.
    """
    if cpu_mode is None:
        cpu_mode = not IS_GPU_MODE
    
    print("\nDownloading RAVDESS dataset...")
    
    ravdess_path = Path(data_root) / "RAVDESS"
    ravdess_path.mkdir(parents=True, exist_ok=True)
    
    # Check if already extracted
    existing_files = list(ravdess_path.rglob("*.wav")) + list(ravdess_path.rglob("*.mp4"))
    if len(existing_files) > 0:
        print(f"RAVDESS already exists with {len(existing_files)} files. Skipping download.")
        return ravdess_path
    
    # RAVDESS Audio dataset (contains both audio and video)
    ravdess_url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
    
    zip_path = ravdess_path / "RAVDESS_Audio.zip"
    
    try:
        # Download audio dataset
        if not zip_path.exists():
            print("Downloading RAVDESS Audio dataset (this may take a while, ~1GB)...")
            download_file_with_progress(ravdess_url, str(zip_path))
        
        # Extract audio
        print("Extracting RAVDESS Audio...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(ravdess_path)
        
        # Clean up zip file to save space
        if zip_path.exists():
            zip_path.unlink()
            print("Cleaned up zip file.")
        
        # If CPU mode, limit to 2 actors
        if cpu_mode:
            print("\nCPU mode: Limiting to 2 actors for testing...")
            all_actors = sorted([d for d in ravdess_path.iterdir() if d.is_dir() and "Actor_" in d.name])
            
            if len(all_actors) > 2:
                # Keep only first 2 actors, remove the rest
                for actor_dir in all_actors[2:]:
                    shutil.rmtree(actor_dir)
                    print(f"Removed {actor_dir.name}")
                
                print(f"Kept {len(all_actors[:2])} actors for testing.")
        
        print("RAVDESS download and extraction complete.")
        
    except Exception as e:
        print(f"Error downloading RAVDESS: {e}")
        print("Please try manual download from: https://zenodo.org/record/1188976")
    
    return ravdess_path



def download_ravdess_video(data_root="./data/raw", cpu_mode=None):
    """
    Download RAVDESS video files separately.
    
    Args:
        data_root: Root directory for raw data
        cpu_mode: If True, download only 2 actors
    """
    if cpu_mode is None:
        cpu_mode = not IS_GPU_MODE
    
    print("\nDownloading RAVDESS video files...")
    
    ravdess_path = Path(data_root) / "RAVDESS"
    ravdess_path.mkdir(parents=True, exist_ok=True)
    
    # Check if videos already exist
    existing_videos = list(ravdess_path.rglob("*.mp4"))
    if len(existing_videos) > 0:
        print(f"Found {len(existing_videos)} video files already downloaded.")
        return ravdess_path
    
    # Individual actor video URLs
    base_url = "https://zenodo.org/record/1188976/files/Video_Speech_Actor_{:02d}.zip?download=1"
    
    num_actors = 2 if cpu_mode else 24
    print(f"Downloading videos for {num_actors} actors...")
    
    for actor_id in range(1, num_actors + 1):
        actor_url = base_url.format(actor_id)
        zip_path = ravdess_path / f"Video_Actor_{actor_id:02d}.zip"
        
        # Check if this actor's videos already exist
        actor_dir = ravdess_path / f"Actor_{actor_id:02d}"
        if actor_dir.exists():
            actor_videos = list(actor_dir.rglob("*.mp4"))
            if len(actor_videos) > 0:
                print(f"Actor {actor_id:02d} videos already exist ({len(actor_videos)} files), skipping...")
                continue
        
        try:
            if not zip_path.exists():
                print(f"Downloading videos for Actor {actor_id:02d}...")
                download_file_with_progress(actor_url, str(zip_path))
            
            print(f"Extracting videos for Actor {actor_id:02d}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(ravdess_path)
            
            # Clean up
            zip_path.unlink()
            print(f"Actor {actor_id:02d} videos complete.")
            
        except Exception as e:
            print(f"Error downloading videos for Actor {actor_id:02d}: {e}")
            continue
    
    print("RAVDESS video download complete.")
    return ravdess_path


def download_cremad(data_root="./data/raw", cpu_mode=None):
    """
    Download CREMA-D dataset from Kaggle.
    
    Args:
        data_root: Root directory for raw data
        cpu_mode: If True, limit samples after extraction. If False, use all.
                  If None, auto-detect based on GPU availability.
    """
    if cpu_mode is None:
        cpu_mode = not IS_GPU_MODE
    
    print("\nDownloading CREMA-D dataset...")
    
    cremad_path = Path(data_root) / "CREMA-D"
    cremad_path.mkdir(parents=True, exist_ok=True)
    
    # Check if already exists
    existing_video = list(cremad_path.rglob("*.flv"))
    existing_audio = list(cremad_path.rglob("*.wav"))
    
    if len(existing_video) > 0 or len(existing_audio) > 0:
        print(f"CREMA-D already exists with {len(existing_video)} videos and {len(existing_audio)} audio files.")
        
        # If CPU mode and we have too many files, limit them
        if cpu_mode and len(existing_audio) > 100:
            print("CPU mode: Limiting to 100 samples...")
            # Keep only first 100 files
            for f in existing_audio[100:]:
                f.unlink()
            for f in existing_video[100:]:
                f.unlink()
            print("Limited to 100 samples.")
        
        return cremad_path
    
    # Check Kaggle credentials
    if not check_kaggle_credentials():
        print("\nCannot download CREMA-D without Kaggle credentials.")
        print("Please set up Kaggle API and try again.")
        return cremad_path
    
    # Download using Kaggle API
    kaggle_dataset = "ejlok1/cremad"
    zip_path = cremad_path / "cremad.zip"
    
    try:
        if not zip_path.exists():
            print("Downloading CREMA-D from Kaggle...")
            print("This may take a while (dataset is ~4GB)...")
            
            # Use kaggle CLI
            cmd = ["kaggle", "datasets", "download", "-d", kaggle_dataset, "-p", str(cremad_path)]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Download complete.")
        else:
            print(f"ZIP file already exists: {zip_path}")
        
        # Extract
        print("Extracting CREMA-D...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cremad_path)
        
        # Clean up zip
        if zip_path.exists():
            zip_path.unlink()
            print("Cleaned up zip file.")
        
        # If CPU mode, limit to 100 samples
        if cpu_mode:
            print("\nCPU mode: Limiting to 100 samples for testing...")
            
            audio_files = sorted(list(cremad_path.rglob("*.wav")))
            video_files = sorted(list(cremad_path.rglob("*.flv")))
            
            # Keep only first 100
            if len(audio_files) > 100:
                for f in audio_files[100:]:
                    f.unlink()
                print(f"Limited audio files to 100")
            
            if len(video_files) > 100:
                for f in video_files[100:]:
                    f.unlink()
                print(f"Limited video files to 100")
        
        print("CREMA-D download and extraction complete.")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError downloading from Kaggle: {e}")
        print("\nTroubleshooting:")
        print("1. Verify kaggle.json is in ~/.kaggle/")
        print("2. Check permissions: chmod 600 ~/.kaggle/kaggle.json")
        print("3. Verify your Kaggle account is active")
        print("4. Manual download: https://www.kaggle.com/datasets/ejlok1/cremad")
    except Exception as e:
        print(f"Error extracting CREMA-D: {e}")
    
    return cremad_path


def verify_datasets(data_root="./data/raw"):
    """
    Verify that datasets are properly downloaded.
    
    Args:
        data_root: Root directory for raw data
    """
    ravdess_path = Path(data_root) / "RAVDESS"
    cremad_path = Path(data_root) / "CREMA-D"
    
    print("\n" + "="*70)
    print("Dataset Verification")
    print("="*70)
    
    total_files = 0
    
    # Check RAVDESS
    if ravdess_path.exists():
        video_files = list(ravdess_path.rglob("*.mp4"))
        audio_files = list(ravdess_path.rglob("*.wav"))
        total_files += len(video_files) + len(audio_files)
        
        print(f"\nRAVDESS:")
        print(f"  Video files: {len(video_files)}")
        print(f"  Audio files: {len(audio_files)}")
        
        if len(audio_files) > 0:
            print(f"  Sample: {audio_files[0].name}")
            # Count actors
            actors = set()
            for f in audio_files:
                parts = f.parts
                for part in parts:
                    if "Actor_" in part:
                        actors.add(part)
            print(f"  Actors: {sorted(actors)}")
    else:
        print("\nRAVDESS: Directory not found")
    
    # Check CREMA-D
    if cremad_path.exists():
        video_files = list(cremad_path.rglob("*.flv"))
        audio_files = list(cremad_path.rglob("*.wav"))
        total_files += len(video_files) + len(audio_files)
        
        print(f"\nCREMA-D:")
        print(f"  Video files: {len(video_files)}")
        print(f"  Audio files: {len(audio_files)}")
        
        if len(audio_files) > 0:
            print(f"  Sample: {audio_files[0].name}")
    else:
        print("\nCREMA-D: Directory not found")
    
    print("\n" + "="*70)
    
    if total_files == 0:
        print("WARNING: No dataset files found!")
        print("Please run the download script.")
    else:
        print(f"SUCCESS: Found {total_files} total files")
    
    print("="*70 + "\n")
    
    return total_files > 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download RAVDESS and CREMA-D datasets")
    parser.add_argument("--data_root", type=str, default="./data/raw",
                      help="Root directory for raw data")
    parser.add_argument("--dataset", type=str, 
                      choices=["ravdess", "cremad", "both"],
                      default="both",
                      help="Which dataset to download")
    parser.add_argument("--cpu_mode", action="store_true",
                      help="CPU mode (limited data for testing)")
    parser.add_argument("--gpu_mode", action="store_true",
                      help="GPU mode (full data)")
    
    args = parser.parse_args()
    
    # Determine mode
    if args.gpu_mode:
        cpu_mode = False
    elif args.cpu_mode:
        cpu_mode = True
    else:
        cpu_mode = not IS_GPU_MODE
    
    print(f"\nMode: {'CPU (limited data)' if cpu_mode else 'GPU (full data)'}")
    
    if args.dataset in ["ravdess", "both"]:
        download_ravdess(args.data_root, cpu_mode=cpu_mode)
    
    if args.dataset in ["cremad", "both"]:
        download_cremad(args.data_root, cpu_mode=cpu_mode)
    
    verify_datasets(args.data_root)