"""
Training script for video-only baseline model.
Trains only on video frames for emotion classification.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import config, DEVICE, IS_GPU_MODE
from data.dataset_loaders import get_dataloaders
from models.video_backbone import VideoBackbone
from models.fusion_module import EmotionClassifier


class VideoOnlyModel(nn.Module):
    """Video-only emotion classification model."""
    
    def __init__(self, num_emotions=8, freeze_backbone=True):
        super(VideoOnlyModel, self).__init__()
        
        print("Initializing Video-Only Model...")
        
        # Video backbone
        self.video_backbone = VideoBackbone(
            model_name="MCG-NJU/videomae-base-short",
            freeze_backbone=freeze_backbone
        )
        video_dim = self.video_backbone.get_output_dim()
        
        # Emotion classifier
        self.emotion_classifier = EmotionClassifier(
            input_dim=video_dim,
            num_emotions=num_emotions
        )
        
        print("Video-Only Model initialized!\n")
    
    def forward(self, video):
        video_features = self.video_backbone(video)
        emotion_logits = self.emotion_classifier(video_features)
        return emotion_logits


class Trainer:
    """Trainer for video-only model."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        num_epochs=20,
        learning_rate=1e-4,
        checkpoint_dir="./checkpoints",
        log_dir="./outputs/logs"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        self.best_val_loss = float('inf')
        
        print(f"Trainer initialized:")
        print(f"  Device: {DEVICE}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning rate: {learning_rate}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            video = batch['video'].to(DEVICE)
            labels = batch['emotion_label'].to(DEVICE)
            
            # Forward pass
            logits = self.model(video)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            batch_size = video.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Accuracy
            pred = torch.argmax(logits, dim=1)
            correct += (pred == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total_samples
            })
        
        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                video = batch['video'].to(DEVICE)
                labels = batch['emotion_label'].to(DEVICE)
                
                # Forward pass
                logits = self.model(video)
                loss = self.criterion(logits, labels)
                
                # Track metrics
                batch_size = video.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Accuracy
                pred = torch.argmax(logits, dim=1)
                correct += (pred == labels).sum().item()
        
        avg_loss = total_loss / total_samples
        accuracy = correct / total_samples
        
        return avg_loss, accuracy
    
    def train(self):
        """Full training loop."""
        print("\nStarting training...")
        print("="*70)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-"*70)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Validate
            val_loss, val_acc = self.validate()
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if new_lr != current_lr:
                print(f"Learning rate reduced: {current_lr} -> {new_lr}")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Save checkpoint if best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                print(f"Saved best model (val_loss: {val_loss:.4f})")
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print("\n" + "="*70)
        print("Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final history
        self.save_history()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        if is_best:
            path = self.checkpoint_dir / "best_model.pth"
        else:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = self.log_dir / "training_history_video_only.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train video-only emotion model")
    parser.add_argument("--data_dir", type=str, default="./data/processed",
                       help="Directory with processed metadata")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/video_only",
                       help="Directory to save checkpoints")
    parser.add_argument("--num_epochs", type=int, default=None,
                       help="Number of epochs (default from config)")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (default from config)")
    parser.add_argument("--learning_rate", type=float, default=None,
                       help="Learning rate (default from config)")
    
    args = parser.parse_args()
    
    # Get config values
    num_epochs = args.num_epochs if args.num_epochs else config['num_epochs']
    batch_size = args.batch_size if args.batch_size else config['batch_size']
    learning_rate = args.learning_rate if args.learning_rate else config['learning_rate']
    
    print(f"\nConfiguration:")
    print(f"  Mode: {'CPU (limited)' if not IS_GPU_MODE else 'GPU (full)'}")
    print(f"  Device: {DEVICE}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    
    # Load data
    train_path = Path(args.data_dir) / "train_metadata.csv"
    val_path = Path(args.data_dir) / "val_metadata.csv"
    test_path = Path(args.data_dir) / "test_metadata.csv"
    
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        str(train_path),
        str(val_path),
        str(test_path),
        batch_size=batch_size,
        mode='video'
    )
    
    # Create model
    print("\nCreating model...")
    model = VideoOnlyModel(num_emotions=8, freeze_backbone=True)
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Train
    trainer.train()