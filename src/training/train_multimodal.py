"""
Training script for the full multimodal model.
Trains on video + audio with emotion and congruence objectives.
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
from models.full_model import MultimodalEmotionModel


class Trainer:
    """Trainer for multimodal emotion model."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        num_epochs=20,
        learning_rate=1e-4,
        emotion_weight=1.0,
        congruence_weight=0.5,
        checkpoint_dir="./checkpoints",
        log_dir="./outputs/logs"
    ):
        """
        Args:
            model: MultimodalEmotionModel instance
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            emotion_weight: Weight for emotion loss
            congruence_weight: Weight for congruence loss
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory to save logs
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.emotion_weight = emotion_weight
        self.congruence_weight = congruence_weight
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss functions
        self.emotion_criterion = nn.CrossEntropyLoss()
        self.congruence_criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate
        )
        
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
            'train_emotion_acc': [],
            'train_congruence_acc': [],
            'val_loss': [],
            'val_emotion_acc': [],
            'val_congruence_acc': []
        }
        
        self.best_val_loss = float('inf')
        
        print(f"Trainer initialized:")
        print(f"  Device: {DEVICE}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Emotion weight: {emotion_weight}")
        print(f"  Congruence weight: {congruence_weight}")
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        emotion_correct = 0
        congruence_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            video = batch['video'].to(DEVICE)
            audio = batch['audio'].to(DEVICE)
            emotion_labels = batch['emotion_label'].to(DEVICE)
            congruence_labels = batch['congruence_label'].to(DEVICE)
            
            # Forward pass
            outputs = self.model(video, audio)
            
            # Compute losses
            emotion_loss = self.emotion_criterion(
                outputs['emotion_logits'],
                emotion_labels
            )
            
            congruence_loss = self.congruence_criterion(
                outputs['congruence_logits'],
                congruence_labels
            )
            
            # Combined loss
            loss = (self.emotion_weight * emotion_loss + 
                   self.congruence_weight * congruence_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            batch_size = video.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Accuracy
            emotion_pred = torch.argmax(outputs['emotion_logits'], dim=1)
            emotion_correct += (emotion_pred == emotion_labels).sum().item()
            
            congruence_pred = torch.argmax(outputs['congruence_logits'], dim=1)
            congruence_correct += (congruence_pred == congruence_labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'emotion_acc': emotion_correct / total_samples,
                'cong_acc': congruence_correct / total_samples
            })
        
        avg_loss = total_loss / total_samples
        emotion_acc = emotion_correct / total_samples
        congruence_acc = congruence_correct / total_samples
        
        return avg_loss, emotion_acc, congruence_acc
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0
        emotion_correct = 0
        congruence_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                video = batch['video'].to(DEVICE)
                audio = batch['audio'].to(DEVICE)
                emotion_labels = batch['emotion_label'].to(DEVICE)
                congruence_labels = batch['congruence_label'].to(DEVICE)
                
                # Forward pass
                outputs = self.model(video, audio)
                
                # Compute losses
                emotion_loss = self.emotion_criterion(
                    outputs['emotion_logits'],
                    emotion_labels
                )
                
                congruence_loss = self.congruence_criterion(
                    outputs['congruence_logits'],
                    congruence_labels
                )
                
                # Combined loss
                loss = (self.emotion_weight * emotion_loss + 
                       self.congruence_weight * congruence_loss)
                
                # Track metrics
                batch_size = video.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Accuracy
                emotion_pred = torch.argmax(outputs['emotion_logits'], dim=1)
                emotion_correct += (emotion_pred == emotion_labels).sum().item()
                
                congruence_pred = torch.argmax(outputs['congruence_logits'], dim=1)
                congruence_correct += (congruence_pred == congruence_labels).sum().item()
        
        avg_loss = total_loss / total_samples
        emotion_acc = emotion_correct / total_samples
        congruence_acc = congruence_correct / total_samples
        
        return avg_loss, emotion_acc, congruence_acc
    
    def train(self):
        """Full training loop."""
        print("\nStarting training...")
        print("="*70)
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-"*70)
            
            # Train
            train_loss, train_emotion_acc, train_cong_acc = self.train_epoch()
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Train Emotion Acc: {train_emotion_acc:.4f}")
            print(f"Train Congruence Acc: {train_cong_acc:.4f}")
            
            # Validate
            val_loss, val_emotion_acc, val_cong_acc = self.validate()
            
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Emotion Acc: {val_emotion_acc:.4f}")
            print(f"Val Congruence Acc: {val_cong_acc:.4f}")
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            
            if new_lr != current_lr:
                print(f"Learning rate reduced: {current_lr} -> {new_lr}")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_emotion_acc'].append(train_emotion_acc)
            self.history['train_congruence_acc'].append(train_cong_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_emotion_acc'].append(val_emotion_acc)
            self.history['val_congruence_acc'].append(val_cong_acc)
            
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
        history_path = self.log_dir / "training_history_multimodal.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train multimodal emotion model")
    parser.add_argument("--data_dir", type=str, default="./data/processed",
                       help="Directory with processed metadata")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/multimodal",
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
        mode='both'
    )
    
    # Create model
    print("\nCreating model...")
    model = MultimodalEmotionModel(
        num_emotions=8,
        freeze_backbones=True
    )
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