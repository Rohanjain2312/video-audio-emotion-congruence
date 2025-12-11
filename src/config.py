"""
Global configuration for CPU vs GPU mode.
All scripts will import from here.
"""

import torch


class Config:
    """Global configuration object."""
    
    # Device detection
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    IS_GPU_MODE = DEVICE == "cuda"
    
    # CPU mode settings (limited/testing)
    CPU_CONFIG = {
        "num_actors_ravdess": 2,           # Download only 2 actors
        "num_samples_cremad": 100,          # Use only 100 samples
        "batch_size": 2,                    # Small batch size (reduced from 4)
        "num_epochs": 2,                    # Quick testing epochs
        "num_workers": 0,                   # No multiprocessing
        "max_frames": 16,                   # 16 frames (required by VideoMAE)
        "train_subset_size": 50,            # Limit training samples
        "val_subset_size": 20,              # Limit validation samples
        "learning_rate": 1e-4,
    }
    
    # GPU mode settings (full training)
    GPU_CONFIG = {
        "num_actors_ravdess": 24,           # All actors
        "num_samples_cremad": None,         # All samples
        "batch_size": 16,                   # Full batch size
        "num_epochs": 20,                   # Full training
        "num_workers": 4,                   # Parallel data loading
        "max_frames": 16,                   # 16 frames per video
        "train_subset_size": None,          # Use all data
        "val_subset_size": None,            # Use all data
        "learning_rate": 1e-4,
    }
    
    @classmethod
    def get_config(cls):
        """Return appropriate config based on device."""
        if cls.IS_GPU_MODE:
            return cls.GPU_CONFIG
        else:
            return cls.CPU_CONFIG
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        mode = "GPU" if cls.IS_GPU_MODE else "CPU"
        config = cls.get_config()
        
        print(f"\n{'='*50}")
        print(f"Running in {mode} mode")
        print(f"Device: {cls.DEVICE}")
        print(f"{'='*50}")
        
        for key, value in config.items():
            print(f"{key:25s}: {value}")
        
        print(f"{'='*50}\n")


# Create global config instance
config = Config.get_config()
DEVICE = Config.DEVICE
IS_GPU_MODE = Config.IS_GPU_MODE


if __name__ == "__main__":
    Config.print_config()