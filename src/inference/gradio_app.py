"""
Gradio web application for interactive emotion recognition demo.
Allows users to upload videos or select from sample videos.
"""

import os
import sys
from pathlib import Path
import gradio as gr
import json
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from inference_pipeline import EmotionInference, EMOTION_NAMES


class GradioDemo:
    """Gradio demo application."""
    
    def __init__(
        self,
        checkpoint_path="./checkpoints/multimodal/best_model.pth",
        sample_videos_dir="./data/samples"
    ):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            sample_videos_dir: Directory containing sample videos
        """
        self.checkpoint_path = checkpoint_path
        self.sample_videos_dir = Path(sample_videos_dir)
        
        # Initialize inference pipeline
        print("Initializing inference pipeline...")
        self.pipeline = EmotionInference(checkpoint_path=checkpoint_path)
        
        # Get sample videos
        self.sample_videos = self.get_sample_videos()
        
        print(f"Demo initialized with {len(self.sample_videos)} sample videos")
    
    def parse_ravdess_filename(self, filename):
        """
        Parse RAVDESS filename to create readable name.
        Format: XX-XX-XX-XX-XX-XX-XX.mp4
        """
        try:
            parts = filename.stem.split('-')
            if len(parts) < 7:
                return filename.stem
            
            # Emotion mapping (position 2)
            emotions = {
                '01': 'Neutral',
                '02': 'Calm', 
                '03': 'Happy',
                '04': 'Sad',
                '05': 'Angry',
                '06': 'Fearful',
                '07': 'Disgust',
                '08': 'Surprised'
            }
            
            # Intensity (position 3)
            intensity = {
                '01': 'Normal',
                '02': 'Strong'
            }
            
            # Actor gender (position 6 - odd=male, even=female)
            actor_num = int(parts[6])
            gender = 'Male' if actor_num % 2 == 1 else 'Female'
            
            emotion = emotions.get(parts[2], 'Unknown')
            intens = intensity.get(parts[3], '')
            
            return f"{emotion} ({intens} - {gender} Actor {actor_num})"
        
        except:
            return filename.stem
    
    def get_sample_videos(self):
        """Get list of sample videos with readable names."""
        samples = []
        
        # Check if sample directory exists
        if not self.sample_videos_dir.exists():
            print(f"Sample videos directory not found: {self.sample_videos_dir}")
            print("Using dataset videos as samples...")
            
            # Use some videos from the dataset as samples
            dataset_path = Path("./data/raw/RAVDESS")
            if dataset_path.exists():
                video_files = list(dataset_path.rglob("*.mp4"))[:10]
                for video_file in video_files:
                    readable_name = self.parse_ravdess_filename(video_file)
                    samples.append({
                        'name': readable_name,
                        'path': str(video_file)
                    })
        else:
            # Get videos from sample directory (limit to 10)
            video_files = []
            for ext in ['*.mp4', '*.avi', '*.mov', '*.flv']:
                video_files.extend(list(self.sample_videos_dir.glob(ext)))
            
            # Limit to 10
            video_files = video_files[:10]
            
            for video_file in video_files:
                readable_name = self.parse_ravdess_filename(video_file)
                samples.append({
                    'name': readable_name,
                    'path': str(video_file)
                })
        
        return samples
    
    def predict_video(self, video_path, progress=gr.Progress()):
        """
        Run prediction on video.
        
        Args:
            video_path: Path to video file
        
        Returns:
            Tuple of (emotion_text, congruence_text, emotion_plot, details_json)
        """
        if video_path is None:
            return (
                "Please upload or select a video",
                "",
                None,
                "No video provided"
            )
        
        try:
            progress(0.3, desc="Extracting features...")
            
            # Run inference
            results = self.pipeline.predict(video_path)
            
            progress(0.8, desc="Generating results...")
            
            # Format emotion results
            emotion_text = self.format_emotion_results(results)
            
            # Format congruence results
            congruence_text = self.format_congruence_results(results)
            
            # Create probability plot
            emotion_plot = self.create_emotion_plot(results)
            
            # Format detailed JSON
            details_json = json.dumps(results, indent=2)
            
            progress(1.0, desc="Complete!")
            
            return emotion_text, congruence_text, emotion_plot, details_json
        
        except Exception as e:
            error_msg = f"Error processing video: {str(e)}"
            print(error_msg)
            return error_msg, "", None, error_msg
    
    def format_emotion_results(self, results):
        """Format emotion prediction results as text."""
        emotion_data = results['emotion']
        
        text = f"## Predicted Emotion: {emotion_data['predicted'].upper()}\n\n"
        text += f"**Confidence:** {emotion_data['confidence']:.2%}\n\n"
        text += "### All Emotion Probabilities:\n\n"
        
        # Sort by probability
        sorted_emotions = sorted(
            emotion_data['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for emotion, prob in sorted_emotions:
            text += f"- **{emotion.capitalize()}**: {prob:.2%}\n"
        
        return text
    
    def format_congruence_results(self, results):
        """Format congruence prediction results as text."""
        congruence_data = results['congruence']
        
        pred = congruence_data['predicted']
        conf = congruence_data['confidence']
        
        if pred == 'congruent':
            text = f"## CONGRUENT\n\n"
            text += "Video and audio emotions **match**\n\n"
        else:
            text = f"## INCONGRUENT\n\n"
            text += "Video and audio emotions **conflict**\n\n"
        
        text += f"**Confidence:** {conf:.2%}\n"
        
        return text
    
    def create_emotion_plot(self, results):
        """Create emotion probabilities bar chart."""
        import matplotlib.pyplot as plt
        
        emotion_data = results['emotion']
        
        # Sort by probability
        sorted_emotions = sorted(
            emotion_data['probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        emotions = [e.capitalize() for e, _ in sorted_emotions]
        probs = [p for _, p in sorted_emotions]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2ecc71' if e.lower() == emotion_data['predicted'] else '#3498db' 
                  for e, _ in sorted_emotions]
        
        bars = ax.barh(emotions, probs, color=colors, alpha=0.8)
        
        # Add value labels
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            ax.text(prob + 0.01, i, f'{prob:.2%}', 
                   va='center', fontsize=10)
        
        ax.set_xlabel('Probability', fontsize=12)
        ax.set_title('Emotion Predictions', fontsize=14, fontweight='bold')
        ax.set_xlim([0, max(probs) * 1.15])
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def load_sample_video(self, sample_name):
        """Load a sample video by name."""
        for sample in self.sample_videos:
            if sample['name'] == sample_name:
                return sample['path']
        return None
    
    def create_interface(self):
        """Create Gradio interface."""
        
        with gr.Blocks(title="Video-Audio Emotion Recognition", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # Video-Audio Emotion Congruence Detector
                
                Upload a video or select a sample to detect emotions from both video and audio,
                and determine if they are congruent (matching) or incongruent (conflicting).
                
                **Features:**
                - Visual emotion detection from facial expressions
                - Audio emotion detection from speech
                - Multimodal fusion for overall emotion
                - Congruence detection (match vs conflict)
                """
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Input")
                    
                    # Video upload
                    video_input = gr.Video(
                        label="Upload Video",
                        sources=["upload"]
                    )
                    
                    # Sample video selector
                    if len(self.sample_videos) > 0:
                        sample_dropdown = gr.Dropdown(
                            choices=[s['name'] for s in self.sample_videos],
                            label="Or select a sample video",
                            value=None
                        )
                        
                        load_sample_btn = gr.Button("Load Sample")
                    
                    # Predict button
                    predict_btn = gr.Button("Analyze Video", variant="primary")
                
                with gr.Column(scale=2):
                    gr.Markdown("### Results")
                    
                    with gr.Row():
                        with gr.Column():
                            emotion_output = gr.Markdown(label="Emotion Prediction")
                        
                        with gr.Column():
                            congruence_output = gr.Markdown(label="Congruence")
                    
                    emotion_plot = gr.Plot(label="Emotion Probabilities")
                    
                    with gr.Accordion("Detailed Results (JSON)", open=False):
                        details_output = gr.Textbox(
                            label="Raw Results",
                            lines=10,
                            max_lines=20
                        )
            
            # Event handlers
            predict_btn.click(
                fn=self.predict_video,
                inputs=[video_input],
                outputs=[emotion_output, congruence_output, emotion_plot, details_output]
            )
            
            if len(self.sample_videos) > 0:
                load_sample_btn.click(
                    fn=self.load_sample_video,
                    inputs=[sample_dropdown],
                    outputs=[video_input]
                )
            
            # About and Links section
            gr.Markdown(
                """
                ---
                
                ## About This Project
                
                This demo uses a multimodal deep learning model combining:
                - **VideoMAE** (MCG-NJU/videomae-base-short) for visual emotion recognition
                - **Wav2Vec2** (wav2vec2-lg-xlsr-en-speech-emotion-recognition) for audio emotion recognition  
                - **Fusion Network** for multimodal integration
                
                **Emotions detected:** Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised
                
                **Training Datasets:** RAVDESS and CREMA-D
                
                ### Links
                
                - **GitHub Repository:** [https://github.com/Rohanjain2312/video-audio-emotion-congruence](https://github.com/Rohanjain2312/video-audio-emotion-congruence)
                - **Developer:** [Rohan Jain - LinkedIn](https://www.linkedin.com/in/jaroh23/)
                
                ### Architecture
                
                The model uses frozen pretrained backbones for feature extraction and trains only the fusion module
                and classification heads, achieving efficient multimodal emotion recognition with ~403M total parameters
                (1.4M trainable).
                
                ---
                
                Built with PyTorch, Transformers, and Gradio | University of Maryland - MS in Machine Learning
                """
            )
        
        return demo
    
    def launch(self, share=False, server_port=7860):
        """Launch the Gradio app."""
        demo = self.create_interface()
        demo.launch(share=share, server_port=server_port)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch Gradio demo app")
    parser.add_argument("--checkpoint", type=str,
                       default="./checkpoints/multimodal/best_model.pth",
                       help="Path to model checkpoint")
    parser.add_argument("--samples", type=str,
                       default="./data/samples",
                       help="Directory with sample videos")
    parser.add_argument("--share", action="store_true",
                       help="Create public share link")
    parser.add_argument("--port", type=int, default=7860,
                       help="Server port")
    
    args = parser.parse_args()
    
    # Create and launch demo
    demo_app = GradioDemo(
        checkpoint_path=args.checkpoint,
        sample_videos_dir=args.samples
    )
    
    print("\nLaunching Gradio demo...")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Sample videos: {args.samples}")
    
    demo_app.launch(share=args.share, server_port=args.port)