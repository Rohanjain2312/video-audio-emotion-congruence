"""
Compare performance across video-only, audio-only, and multimodal models.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def load_metrics(metrics_path):
    """Load metrics from JSON file."""
    if not os.path.exists(metrics_path):
        print(f"Warning: {metrics_path} not found")
        return None
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


def compare_models(metrics_dir="./outputs/metrics", output_dir="./outputs/comparisons"):
    """
    Compare all model variants.
    
    Args:
        metrics_dir: Directory containing individual model metrics
        output_dir: Directory to save comparison outputs
    """
    metrics_dir = Path(metrics_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nLoading model metrics...")
    
    # Load metrics for each model
    multimodal_metrics = load_metrics(metrics_dir / "multimodal_metrics.json")
    video_metrics = load_metrics(metrics_dir / "video_only_metrics.json")
    audio_metrics = load_metrics(metrics_dir / "audio_only_metrics.json")
    
    # Check which models are available
    available_models = []
    if multimodal_metrics:
        available_models.append(('Multimodal', multimodal_metrics))
    if video_metrics:
        available_models.append(('Video Only', video_metrics))
    if audio_metrics:
        available_models.append(('Audio Only', audio_metrics))
    
    if len(available_models) == 0:
        print("Error: No model metrics found!")
        print(f"Expected files in {metrics_dir}:")
        print("  - multimodal_metrics.json")
        print("  - video_only_metrics.json")
        print("  - audio_only_metrics.json")
        return
    
    print(f"Found metrics for {len(available_models)} model(s): {[m[0] for m in available_models]}")
    
    # Create comparison tables
    create_overall_comparison(available_models, output_dir)
    create_per_class_comparison(available_models, output_dir)
    
    # Create visualizations
    plot_overall_metrics(available_models, output_dir)
    plot_per_class_comparison(available_models, output_dir)
    
    # Print summary
    print_comparison_summary(available_models)
    
    print(f"\nComparison results saved to {output_dir}")


def create_overall_comparison(models, output_dir):
    """Create overall metrics comparison table."""
    
    data = []
    for model_name, metrics in models:
        emotion_metrics = metrics['emotion_metrics']
        
        row = {
            'Model': model_name,
            'Accuracy': emotion_metrics['accuracy'],
            'Balanced Accuracy': emotion_metrics['balanced_accuracy'],
            'Macro Precision': emotion_metrics['macro_precision'],
            'Macro Recall': emotion_metrics['macro_recall'],
            'Macro F1': emotion_metrics['macro_f1'],
            'Weighted F1': emotion_metrics['weighted_f1']
        }
        
        # Add congruence metrics if available
        if 'congruence_metrics' in metrics:
            cong = metrics['congruence_metrics']
            row['Congruence Acc'] = cong['accuracy']
            row['Congruence F1'] = cong['f1']
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = output_dir / "overall_comparison.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Overall comparison saved to {csv_path}")
    
    return df


def create_per_class_comparison(models, output_dir):
    """Create per-class metrics comparison table."""
    
    # Get emotion names from first model
    emotion_names = list(models[0][1]['emotion_metrics']['per_class'].keys())
    
    data = []
    for emotion in emotion_names:
        row = {'Emotion': emotion}
        
        for model_name, metrics in models:
            per_class = metrics['emotion_metrics']['per_class'].get(emotion, {})
            row[f'{model_name} Precision'] = per_class.get('precision', 0)
            row[f'{model_name} Recall'] = per_class.get('recall', 0)
            row[f'{model_name} F1'] = per_class.get('f1', 0)
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    csv_path = output_dir / "per_class_comparison.csv"
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f"Per-class comparison saved to {csv_path}")
    
    return df


def plot_overall_metrics(models, output_dir):
    """Plot overall metrics comparison."""
    
    metrics_to_plot = ['Accuracy', 'Macro F1', 'Weighted F1']
    
    data = {metric: [] for metric in metrics_to_plot}
    model_names = []
    
    for model_name, metrics in models:
        model_names.append(model_name)
        emotion_metrics = metrics['emotion_metrics']
        
        data['Accuracy'].append(emotion_metrics['accuracy'])
        data['Macro F1'].append(emotion_metrics['macro_f1'])
        data['Weighted F1'].append(emotion_metrics['weighted_f1'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(model_names))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x + i * width, data[metric], width, label=metric, alpha=0.8)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Overall Performance Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for i, metric in enumerate(metrics_to_plot):
        for j, value in enumerate(data[metric]):
            ax.text(j + i * width, value + 0.02, f'{value:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save
    save_path = output_dir / "overall_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Overall comparison plot saved to {save_path}")


def plot_per_class_comparison(models, output_dir):
    """Plot per-class F1 scores comparison."""
    
    # Get emotion names
    emotion_names = list(models[0][1]['emotion_metrics']['per_class'].keys())
    
    # Prepare data
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(len(emotion_names))
    width = 0.8 / len(models)
    
    for i, (model_name, metrics) in enumerate(models):
        f1_scores = []
        for emotion in emotion_names:
            per_class = metrics['emotion_metrics']['per_class'].get(emotion, {})
            f1_scores.append(per_class.get('f1', 0))
        
        ax.bar(x + i * width, f1_scores, width, label=model_name, alpha=0.8)
    
    ax.set_xlabel('Emotion')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score Comparison')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(emotion_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    
    # Save
    save_path = output_dir / "per_class_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Per-class comparison plot saved to {save_path}")


def plot_improvement_analysis(models, output_dir):
    """Analyze multimodal improvement over unimodal baselines."""
    
    if len(models) < 3:
        print("Skipping improvement analysis (need all 3 models)")
        return
    
    # Find multimodal model
    multimodal = None
    video_only = None
    audio_only = None
    
    for name, metrics in models:
        if name == 'Multimodal':
            multimodal = metrics
        elif name == 'Video Only':
            video_only = metrics
        elif name == 'Audio Only':
            audio_only = metrics
    
    if not (multimodal and video_only and audio_only):
        print("Skipping improvement analysis (missing models)")
        return
    
    # Get emotion names
    emotion_names = list(multimodal['emotion_metrics']['per_class'].keys())
    
    # Calculate improvements
    improvements_over_video = []
    improvements_over_audio = []
    
    for emotion in emotion_names:
        mm_f1 = multimodal['emotion_metrics']['per_class'][emotion]['f1']
        v_f1 = video_only['emotion_metrics']['per_class'][emotion]['f1']
        a_f1 = audio_only['emotion_metrics']['per_class'][emotion]['f1']
        
        improvements_over_video.append(mm_f1 - v_f1)
        improvements_over_audio.append(mm_f1 - a_f1)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(emotion_names))
    width = 0.35
    
    ax.bar(x - width/2, improvements_over_video, width, 
           label='Improvement over Video-Only', alpha=0.8)
    ax.bar(x + width/2, improvements_over_audio, width,
           label='Improvement over Audio-Only', alpha=0.8)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Emotion')
    ax.set_ylabel('F1 Score Improvement')
    ax.set_title('Multimodal Fusion Improvement Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(emotion_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    save_path = output_dir / "improvement_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Improvement analysis plot saved to {save_path}")


def print_comparison_summary(models):
    """Print comparison summary to console."""
    
    print("\n" + "="*70)
    print("Model Comparison Summary")
    print("="*70)
    
    for model_name, metrics in models:
        emotion_metrics = metrics['emotion_metrics']
        
        print(f"\n{model_name}:")
        print(f"  Accuracy:         {emotion_metrics['accuracy']:.4f}")
        print(f"  Macro F1:         {emotion_metrics['macro_f1']:.4f}")
        print(f"  Weighted F1:      {emotion_metrics['weighted_f1']:.4f}")
        
        if 'congruence_metrics' in metrics:
            cong = metrics['congruence_metrics']
            print(f"  Congruence Acc:   {cong['accuracy']:.4f}")
            print(f"  Congruence F1:    {cong['f1']:.4f}")
    
    # Find best model
    best_model = max(models, key=lambda x: x[1]['emotion_metrics']['macro_f1'])
    print(f"\nBest Model (by Macro F1): {best_model[0]}")
    
    # Calculate improvements if all models present
    if len(models) == 3:
        multimodal = next((m for m in models if m[0] == 'Multimodal'), None)
        video_only = next((m for m in models if m[0] == 'Video Only'), None)
        audio_only = next((m for m in models if m[0] == 'Audio Only'), None)
        
        if multimodal and video_only and audio_only:
            mm_f1 = multimodal[1]['emotion_metrics']['macro_f1']
            v_f1 = video_only[1]['emotion_metrics']['macro_f1']
            a_f1 = audio_only[1]['emotion_metrics']['macro_f1']
            
            print(f"\nMultimodal Fusion Gains:")
            print(f"  vs Video-Only: {(mm_f1 - v_f1)*100:+.2f}% F1")
            print(f"  vs Audio-Only: {(mm_f1 - a_f1)*100:+.2f}% F1")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare baseline models")
    parser.add_argument("--metrics_dir", type=str, default="./outputs/metrics",
                       help="Directory containing model metrics")
    parser.add_argument("--output_dir", type=str, default="./outputs/comparisons",
                       help="Directory to save comparison outputs")
    
    args = parser.parse_args()
    
    compare_models(args.metrics_dir, args.output_dir)