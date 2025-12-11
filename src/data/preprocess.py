"""
Preprocess RAVDESS and CREMA-D datasets.
- Unify emotion labels across datasets
- Create metadata CSV files
- Split into train/val/test sets
- Generate congruence labels
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

# Import config
sys.path.append(str(Path(__file__).parent.parent))
from config import config, IS_GPU_MODE


# Unified emotion mapping
EMOTION_MAPPING = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fearful': 5,
    'disgust': 6,
    'surprised': 7
}

NUM_EMOTIONS = len(EMOTION_MAPPING)


def parse_ravdess_filename(filename):
    """
    Parse RAVDESS filename to extract metadata.
    
    Format: 03-01-06-01-02-01-12.wav or 02-01-06-01-02-01-12.mp4
    - Modality (01 = full-AV, 02 = video-only, 03 = audio-only)
    - Vocal channel (01 = speech, 02 = song)
    - Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised)
    - Emotional intensity (01 = normal, 02 = strong)
    - Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door")
    - Repetition (01 = 1st repetition, 02 = 2nd repetition)
    - Actor (01 to 24, odd = male, even = female)
    """
    parts = filename.stem.split('-')
    
    if len(parts) != 7:
        return None
    
    modality = parts[0]
    vocal_channel = parts[1]
    emotion_code = int(parts[2])
    intensity = parts[3]
    statement = parts[4]
    repetition = parts[5]
    actor = int(parts[6])
    
    # Map emotion code to label
    emotion_map_ravdess = {
        1: 'neutral',
        2: 'calm',
        3: 'happy',
        4: 'sad',
        5: 'angry',
        6: 'fearful',
        7: 'disgust',
        8: 'surprised'
    }
    
    emotion = emotion_map_ravdess.get(emotion_code, 'unknown')
    gender = 'male' if actor % 2 == 1 else 'female'
    
    return {
        'filename': filename.name,
        'filepath': str(filename),
        'dataset': 'RAVDESS',
        'emotion': emotion,
        'emotion_id': EMOTION_MAPPING.get(emotion, -1),
        'actor': actor,
        'gender': gender,
        'intensity': intensity,
        'modality': modality,
        'statement': statement,
        'repetition': repetition,
        'vocal_channel': vocal_channel
    }


def match_ravdess_video_audio(audio_file, video_files_by_actor):
    """
    Match RAVDESS audio file with corresponding video file.
    
    Audio files: 03-01-06-01-02-01-12.wav (modality 03 = audio-only)
    Video files: 02-01-06-01-02-01-12.mp4 (modality 02 = video-only) or
                 01-01-06-01-02-01-12.mp4 (modality 01 = full AV)
    
    Match based on: emotion, intensity, statement, repetition, actor
    """
    audio_info = parse_ravdess_filename(audio_file)
    if not audio_info:
        return None
    
    actor = audio_info['actor']
    
    # Get videos for this actor
    if actor not in video_files_by_actor:
        return None
    
    # Look for matching video
    for video_file in video_files_by_actor[actor]:
        video_info = parse_ravdess_filename(video_file)
        if not video_info:
            continue
        
        # Match on key attributes (ignoring modality)
        if (video_info['actor'] == audio_info['actor'] and
            video_info['emotion'] == audio_info['emotion'] and
            video_info['intensity'] == audio_info['intensity'] and
            video_info['statement'] == audio_info['statement'] and
            video_info['repetition'] == audio_info['repetition'] and
            video_info['vocal_channel'] == audio_info['vocal_channel']):
            return video_file
    
    return None


def process_ravdess(data_root="./data/raw"):
    """
    Process RAVDESS dataset and create metadata.
    
    Args:
        data_root: Root directory containing raw data
    
    Returns:
        DataFrame with metadata
    """
    print("\nProcessing RAVDESS dataset...")
    
    ravdess_path = Path(data_root) / "RAVDESS"
    
    if not ravdess_path.exists():
        print("RAVDESS directory not found.")
        return pd.DataFrame()
    
    # Find all audio and video files
    audio_files = list(ravdess_path.rglob("*.wav"))
    video_files = list(ravdess_path.rglob("*.mp4"))
    
    print(f"Found {len(audio_files)} RAVDESS audio files")
    print(f"Found {len(video_files)} RAVDESS video files")
    
    # Group video files by actor for efficient matching
    video_files_by_actor = {}
    for video_file in video_files:
        video_info = parse_ravdess_filename(video_file)
        if video_info:
            actor = video_info['actor']
            if actor not in video_files_by_actor:
                video_files_by_actor[actor] = []
            video_files_by_actor[actor].append(video_file)
    
    print(f"Video files grouped by {len(video_files_by_actor)} actors")
    
    # Process audio files and match with videos
    metadata = []
    matched_count = 0
    
    for audio_file in audio_files:
        info = parse_ravdess_filename(audio_file)
        if info and info['emotion_id'] != -1:
            # Find matching video
            video_file = match_ravdess_video_audio(audio_file, video_files_by_actor)
            
            info['has_video'] = video_file is not None
            info['video_filepath'] = str(video_file) if video_file else None
            
            if info['has_video']:
                matched_count += 1
            
            metadata.append(info)
    
    df = pd.DataFrame(metadata)
    print(f"Processed {len(df)} RAVDESS samples")
    print(f"Matched audio-video pairs: {matched_count}")
    print(f"Emotion distribution:\n{df['emotion'].value_counts()}")
    
    return df


def parse_cremad_filename(filename):
    """
    Parse CREMA-D filename to extract metadata.
    
    Format: 1001_DFA_ANG_XX.wav
    - Actor ID (1001-1091)
    - Sentence (DFA, IEO, ITH, etc.)
    - Emotion (ANG, DIS, FEA, HAP, NEU, SAD)
    - Intensity (LO, MD, HI, XX)
    """
    parts = filename.stem.split('_')
    
    if len(parts) != 4:
        return None
    
    actor = parts[0]
    sentence = parts[1]
    emotion_code = parts[2]
    intensity = parts[3]
    
    # Map emotion code to label
    emotion_map_cremad = {
        'ANG': 'angry',
        'DIS': 'disgust',
        'FEA': 'fearful',
        'HAP': 'happy',
        'NEU': 'neutral',
        'SAD': 'sad'
    }
    
    emotion = emotion_map_cremad.get(emotion_code, 'unknown')
    
    return {
        'filename': filename.name,
        'filepath': str(filename),
        'dataset': 'CREMA-D',
        'emotion': emotion,
        'emotion_id': EMOTION_MAPPING.get(emotion, -1),
        'actor': actor,
        'sentence': sentence,
        'intensity': intensity
    }


def process_cremad(data_root="./data/raw", cpu_mode=None):
    """
    Process CREMA-D dataset and create metadata.
    
    Args:
        data_root: Root directory containing raw data
        cpu_mode: If True, limit to first 100 samples
    
    Returns:
        DataFrame with metadata
    """
    if cpu_mode is None:
        cpu_mode = not IS_GPU_MODE
    
    print("\nProcessing CREMA-D dataset...")
    
    cremad_path = Path(data_root) / "CREMA-D"
    
    if not cremad_path.exists():
        print("CREMA-D directory not found.")
        return pd.DataFrame()
    
    # Find all audio files
    audio_files = sorted(list(cremad_path.rglob("*.wav")))
    
    # Limit if CPU mode
    if cpu_mode and len(audio_files) > 100:
        audio_files = audio_files[:100]
        print(f"CPU mode: Limited to {len(audio_files)} samples")
    
    print(f"Found {len(audio_files)} CREMA-D audio files")
    
    metadata = []
    for audio_file in audio_files:
        info = parse_cremad_filename(audio_file)
        if info and info['emotion_id'] != -1:
            # Check if corresponding video exists
            # CREMA-D videos are .flv files with same base name
            video_file = audio_file.with_suffix('.flv')
            
            # Also check in VideoFlash folder
            if not video_file.exists():
                video_flash_dir = cremad_path / "VideoFlash"
                if video_flash_dir.exists():
                    video_file = video_flash_dir / (audio_file.stem + '.flv')
            
            info['has_video'] = video_file.exists()
            info['video_filepath'] = str(video_file) if video_file.exists() else None
            
            metadata.append(info)
    
    df = pd.DataFrame(metadata)
    print(f"Processed {len(df)} CREMA-D samples")
    print(f"Samples with video: {df['has_video'].sum()}")
    print(f"Emotion distribution:\n{df['emotion'].value_counts()}")
    
    return df


def add_congruence_labels(df):
    """
    Add congruence labels based on video and audio emotion match.
    For now, we assume video and audio have the same emotion (they do in these datasets).
    
    Args:
        df: DataFrame with metadata
    
    Returns:
        DataFrame with congruence labels added
    """
    # In RAVDESS and CREMA-D, video and audio emotions are the same
    # So all samples are congruent by default
    df['video_emotion'] = df['emotion']
    df['audio_emotion'] = df['emotion']
    df['video_emotion_id'] = df['emotion_id']
    df['audio_emotion_id'] = df['emotion_id']
    df['is_congruent'] = 1  # All are congruent in these datasets
    
    return df


def create_splits(df, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Split data into train/val/test sets with stratification.
    
    Args:
        df: DataFrame with metadata
        train_size: Proportion for training
        val_size: Proportion for validation
        test_size: Proportion for testing
        random_state: Random seed
    
    Returns:
        train_df, val_df, test_df
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    print("\nCreating train/val/test splits...")
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df['emotion'],
        random_state=random_state
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        stratify=temp_df['emotion'],
        random_state=random_state
    )
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Val set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, val_df, test_df


def save_metadata(train_df, val_df, test_df, output_dir="./data/processed"):
    """
    Save metadata to CSV files.
    
    Args:
        train_df, val_df, test_df: DataFrames with metadata
        output_dir: Directory to save CSV files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(output_path / "train_metadata.csv", index=False)
    val_df.to_csv(output_path / "val_metadata.csv", index=False)
    test_df.to_csv(output_path / "test_metadata.csv", index=False)
    
    # Save combined metadata
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_df.to_csv(output_path / "all_metadata.csv", index=False)
    
    # Save emotion mapping
    emotion_mapping_df = pd.DataFrame(
        list(EMOTION_MAPPING.items()),
        columns=['emotion', 'emotion_id']
    )
    emotion_mapping_df.to_csv(output_path / "emotion_mapping.csv", index=False)
    
    print(f"\nMetadata saved to {output_path}")
    print(f"Files: train_metadata.csv, val_metadata.csv, test_metadata.csv, all_metadata.csv")


def print_statistics(train_df, val_df, test_df):
    """Print dataset statistics."""
    print("\n" + "="*70)
    print("Dataset Statistics")
    print("="*70)
    
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    print(f"\nTotal samples: {len(all_df)}")
    print(f"  RAVDESS: {len(all_df[all_df['dataset'] == 'RAVDESS'])}")
    print(f"  CREMA-D: {len(all_df[all_df['dataset'] == 'CREMA-D'])}")
    
    print(f"\nSamples with video: {all_df['has_video'].sum()}")
    print(f"Samples with audio only: {(~all_df['has_video']).sum()}")
    
    print(f"\nEmotion distribution (all data):")
    print(all_df['emotion'].value_counts().sort_index())
    
    print(f"\nCongruent samples: {all_df['is_congruent'].sum()} ({all_df['is_congruent'].mean()*100:.1f}%)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess RAVDESS and CREMA-D datasets")
    parser.add_argument("--data_root", type=str, default="./data/raw",
                      help="Root directory for raw data")
    parser.add_argument("--output_dir", type=str, default="./data/processed",
                      help="Output directory for processed metadata")
    parser.add_argument("--cpu_mode", action="store_true",
                      help="CPU mode (limited data)")
    
    args = parser.parse_args()
    
    cpu_mode = args.cpu_mode or (not IS_GPU_MODE)
    
    print(f"Mode: {'CPU (limited data)' if cpu_mode else 'GPU (full data)'}")
    
    # Process datasets
    ravdess_df = process_ravdess(args.data_root)
    cremad_df = process_cremad(args.data_root, cpu_mode=cpu_mode)
    
    # Combine datasets
    if len(ravdess_df) == 0 and len(cremad_df) == 0:
        print("\nError: No data found. Please download datasets first.")
        exit(1)
    
    all_df = pd.concat([ravdess_df, cremad_df], ignore_index=True)
    
    # Add congruence labels
    all_df = add_congruence_labels(all_df)
    
    # Create splits
    train_df, val_df, test_df = create_splits(all_df)
    
    # Save metadata
    save_metadata(train_df, val_df, test_df, args.output_dir)
    
    # Print statistics
    print_statistics(train_df, val_df, test_df)
    
    print("\nPreprocessing complete!")