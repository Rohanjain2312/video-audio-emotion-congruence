"""
Comprehensive evaluation script for emotion models.
Computes accuracy, precision, recall, F1, and confusion matrices.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, balanced_accuracy_score
)
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import config, DEVICE, IS_GPU_MODE
from data.dataset_loaders import get_dataloaders
from models.full_model import MultimodalEmotionModel


# Emotion mapping
EMOTION_NAMES = [
    'neutral', 'calm', 'happy', 'sad', 
    'angry', 'fearful', 'disgust', 'surprised'
]


class ModelEvaluator:
    """Evaluator for emotion recognition models."""
    
    def __init__(
        self,
        model,
        test_loader,
        emotion_names=EMOTION_NAMES,
        output_dir="./outputs/metrics",
        model_name="model"
    ):
        """
        Args:
            model: Trained model
            test_loader: Test data loader
            emotion_names: List of emotion class names
            output_dir: Directory to save outputs
            model_name: Name for saving outputs
        """
        self.model = model
        self.test_loader = test_loader
        self.emotion_names = emotion_names
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Evaluator initialized:")
        print(f"  Model: {model_name}")
        print(f"  Device: {DEVICE}")
        print(f"  Output dir: {output_dir}")
    
    def evaluate_emotion(self):
        """Evaluate emotion classification performance."""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        all_logits = []
        
        print("\nEvaluating emotion classification...")
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                # Get inputs based on model type
                if hasattr(self.model, 'video_backbone') and hasattr(self.model, 'audio_backbone'):
                    # Multimodal model
                    video = batch['video'].to(DEVICE)
                    audio = batch['audio'].to(DEVICE)
                    outputs = self.model(video, audio)
                    logits = outputs['emotion_logits']
                elif hasattr(self.model, 'video_backbone'):
                    # Video-only model
                    video = batch['video'].to(DEVICE)
                    logits = self.model(video)
                elif hasattr(self.model, 'audio_backbone'):
                    # Audio-only model
                    audio = batch['audio'].to(DEVICE)
                    logits = self.model(audio)
                else:
                    raise ValueError("Unknown model type")
                
                labels = batch['emotion_label'].to(DEVICE)
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                
                # Store results
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        
        # Compute metrics
        metrics = self.compute_emotion_metrics(all_labels, all_preds)
        
        # Generate visualizations
        self.plot_confusion_matrix(all_labels, all_preds, "emotion")
        self.plot_per_class_metrics(all_labels, all_preds)
        
        return metrics, all_preds, all_labels, all_logits
    
    def evaluate_congruence(self):
        """Evaluate congruence classification performance (multimodal only)."""
        if not (hasattr(self.model, 'video_backbone') and hasattr(self.model, 'audio_backbone')):
            print("Congruence evaluation only available for multimodal models")
            return None, None, None
        
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        print("\nEvaluating congruence classification...")
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                video = batch['video'].to(DEVICE)
                audio = batch['audio'].to(DEVICE)
                outputs = self.model(video, audio)
                
                logits = outputs['congruence_logits']
                labels = batch['congruence_label'].to(DEVICE)
                
                # Get predictions
                preds = torch.argmax(logits, dim=1)
                
                # Store results
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        metrics = self.compute_congruence_metrics(all_labels, all_preds)
        
        # Generate visualizations
        self.plot_confusion_matrix(all_labels, all_preds, "congruence")
        
        return metrics, all_preds, all_labels
    
    def compute_emotion_metrics(self, labels, preds):
        """Compute comprehensive emotion metrics."""
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'balanced_accuracy': balanced_accuracy_score(labels, preds),
            'macro_precision': precision_score(labels, preds, average='macro', zero_division=0),
            'macro_recall': recall_score(labels, preds, average='macro', zero_division=0),
            'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
            'weighted_precision': precision_score(labels, preds, average='weighted', zero_division=0),
            'weighted_recall': recall_score(labels, preds, average='weighted', zero_division=0),
            'weighted_f1': f1_score(labels, preds, average='weighted', zero_division=0)
        }
        
        # Per-class metrics
        per_class_precision = precision_score(labels, preds, average=None, zero_division=0)
        per_class_recall = recall_score(labels, preds, average=None, zero_division=0)
        per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
        
        metrics['per_class'] = {}
        for i, emotion in enumerate(self.emotion_names):
            metrics['per_class'][emotion] = {
                'precision': float(per_class_precision[i]),
                'recall': float(per_class_recall[i]),
                'f1': float(per_class_f1[i])
            }
        
        return metrics
    
    def compute_congruence_metrics(self, labels, preds):
        """Compute congruence classification metrics."""
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'precision': precision_score(labels, preds, average='binary', zero_division=0),
            'recall': recall_score(labels, preds, average='binary', zero_division=0),
            'f1': f1_score(labels, preds, average='binary', zero_division=0)
        }
        
        return metrics
    
    def plot_confusion_matrix(self, labels, preds, task="emotion"):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(labels, preds)
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        if task == "emotion":
            class_names = self.emotion_names
        else:
            class_names = ['Incongruent', 'Congruent']
        
        sns.heatmap(
            cm_normalized,
            annot=cm,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Normalized Count'}
        )
        
        plt.title(f'{task.capitalize()} Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / f"{self.model_name}_{task}_confusion_matrix.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to {save_path}")
    
    def plot_per_class_metrics(self, labels, preds):
        """Plot per-class precision, recall, F1."""
        # Compute per-class metrics
        precision = precision_score(labels, preds, average=None, zero_division=0)
        recall = recall_score(labels, preds, average=None, zero_division=0)
        f1 = f1_score(labels, preds, average=None, zero_division=0)
        
        # Create dataframe
        df = pd.DataFrame({
            'Emotion': self.emotion_names,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(self.emotion_names))
        width = 0.25
        
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Emotion')
        ax.set_ylabel('Score')
        ax.set_title(f'Per-Class Metrics - {self.model_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(self.emotion_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save
        save_path = self.output_dir / f"{self.model_name}_per_class_metrics.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Per-class metrics plot saved to {save_path}")
    
    def save_metrics(self, emotion_metrics, congruence_metrics=None):
        """Save metrics to JSON file."""
        results = {
            'model_name': self.model_name,
            'emotion_metrics': emotion_metrics
        }
        
        if congruence_metrics is not None:
            results['congruence_metrics'] = congruence_metrics
        
        # Save to JSON
        save_path = self.output_dir / f"{self.model_name}_metrics.json"
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Metrics saved to {save_path}")
    
    def print_report(self, emotion_metrics, congruence_metrics=None):
        """Print evaluation report."""
        print("\n" + "="*70)
        print(f"Evaluation Report - {self.model_name}")
        print("="*70)
        
        print("\nEmotion Classification Metrics:")
        print("-"*70)
        print(f"Accuracy:           {emotion_metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy:  {emotion_metrics['balanced_accuracy']:.4f}")
        print(f"Macro F1:           {emotion_metrics['macro_f1']:.4f}")
        print(f"Weighted F1:        {emotion_metrics['weighted_f1']:.4f}")
        
        print("\nPer-Class Performance:")
        print("-"*70)
        for emotion, metrics in emotion_metrics['per_class'].items():
            print(f"{emotion:12s}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
        
        if congruence_metrics is not None:
            print("\nCongruence Classification Metrics:")
            print("-"*70)
            print(f"Accuracy:   {congruence_metrics['accuracy']:.4f}")
            print(f"Precision:  {congruence_metrics['precision']:.4f}")
            print(f"Recall:     {congruence_metrics['recall']:.4f}")
            print(f"F1:         {congruence_metrics['f1']:.4f}")
        
        print("\n" + "="*70)


def load_model(checkpoint_path, model_type='multimodal', num_emotions=8):
    """Load a trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    if model_type == 'multimodal':
        from models.full_model import MultimodalEmotionModel
        model = MultimodalEmotionModel(num_emotions=num_emotions, freeze_backbones=True)
    elif model_type == 'video_only':
        from training.train_video_only import VideoOnlyModel
        model = VideoOnlyModel(num_emotions=num_emotions, freeze_backbone=True)
    elif model_type == 'audio_only':
        from training.train_audio_only import AudioOnlyModel
        model = AudioOnlyModel(num_emotions=num_emotions, freeze_backbone=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained emotion model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--model_type", type=str, default="multimodal",
                       choices=['multimodal', 'video_only', 'audio_only'],
                       help="Type of model")
    parser.add_argument("--data_dir", type=str, default="./data/processed",
                       help="Directory with processed metadata")
    parser.add_argument("--output_dir", type=str, default="./outputs/metrics",
                       help="Directory to save evaluation outputs")
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Batch size (default from config)")
    
    args = parser.parse_args()
    
    # Get batch size
    batch_size = args.batch_size if args.batch_size else config['batch_size']
    
    print(f"\nConfiguration:")
    print(f"  Device: {DEVICE}")
    print(f"  Model type: {args.model_type}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Batch size: {batch_size}")
    
    # Determine data mode
    if args.model_type == 'multimodal':
        data_mode = 'both'
    elif args.model_type == 'video_only':
        data_mode = 'video'
    else:
        data_mode = 'audio'
    
    # Load data
    train_path = Path(args.data_dir) / "train_metadata.csv"
    val_path = Path(args.data_dir) / "val_metadata.csv"
    test_path = Path(args.data_dir) / "test_metadata.csv"
    
    print("\nLoading test data...")
    _, _, test_loader = get_dataloaders(
        str(train_path),
        str(val_path),
        str(test_path),
        batch_size=batch_size,
        mode=data_mode
    )
    
    # Load model
    model = load_model(args.checkpoint, args.model_type)
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        test_loader=test_loader,
        output_dir=args.output_dir,
        model_name=args.model_type
    )
    
    # Evaluate emotion classification
    emotion_metrics, _, _, _ = evaluator.evaluate_emotion()
    
    # Evaluate congruence (if multimodal)
    congruence_metrics = None
    if args.model_type == 'multimodal':
        congruence_metrics, _, _ = evaluator.evaluate_congruence()
    
    # Save metrics
    evaluator.save_metrics(emotion_metrics, congruence_metrics)
    
    # Print report
    evaluator.print_report(emotion_metrics, congruence_metrics)