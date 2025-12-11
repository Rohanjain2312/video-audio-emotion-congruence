"""
Check the actual file structure of downloaded datasets.
"""

from pathlib import Path

def check_ravdess_structure(data_root="./data/raw"):
    """Check RAVDESS file structure."""
    ravdess_path = Path(data_root) / "RAVDESS"
    
    if not ravdess_path.exists():
        print("RAVDESS not found")
        return
    
    print("\n" + "="*70)
    print("RAVDESS Structure")
    print("="*70)
    
    # Find all files
    audio_files = list(ravdess_path.rglob("*.wav"))
    video_files = list(ravdess_path.rglob("*.mp4"))
    
    print(f"\nTotal audio files: {len(audio_files)}")
    print(f"Total video files: {len(video_files)}")
    
    if audio_files:
        print(f"\nSample audio paths:")
        for f in audio_files[:3]:
            print(f"  {f.relative_to(ravdess_path)}")
    
    if video_files:
        print(f"\nSample video paths:")
        for f in video_files[:3]:
            print(f"  {f.relative_to(ravdess_path)}")
    
    # Check for matching pairs
    print("\nChecking for audio-video pairs...")
    audio_stems = {f.stem: f for f in audio_files}
    video_stems = {f.stem: f for f in video_files}
    
    matching = set(audio_stems.keys()) & set(video_stems.keys())
    print(f"Files with matching audio+video: {len(matching)}")
    
    if matching:
        sample = list(matching)[0]
        print(f"\nExample pair:")
        print(f"  Audio: {audio_stems[sample].relative_to(ravdess_path)}")
        print(f"  Video: {video_stems[sample].relative_to(ravdess_path)}")


def check_cremad_structure(data_root="./data/raw"):
    """Check CREMA-D file structure."""
    cremad_path = Path(data_root) / "CREMA-D"
    
    if not cremad_path.exists():
        print("\nCREMA-D not found")
        return
    
    print("\n" + "="*70)
    print("CREMA-D Structure")
    print("="*70)
    
    audio_files = list(cremad_path.rglob("*.wav"))
    video_files = list(cremad_path.rglob("*.flv"))
    
    print(f"\nTotal audio files: {len(audio_files)}")
    print(f"Total video files: {len(video_files)}")
    
    if audio_files:
        print(f"\nSample audio paths:")
        for f in audio_files[:3]:
            print(f"  {f.relative_to(cremad_path)}")
    
    if video_files:
        print(f"\nSample video paths:")
        for f in video_files[:3]:
            print(f"  {f.relative_to(cremad_path)}")


if __name__ == "__main__":
    check_ravdess_structure()
    check_cremad_structure()