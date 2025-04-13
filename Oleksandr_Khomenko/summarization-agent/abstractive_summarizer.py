"""Abstractive summarization module using OpenAI's multi-modal LLMs."""

import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from openai import OpenAI
from typing import Dict, Any, Optional, List, Tuple, Union

class AbstractiveSummarizer:
    """Probabilistic abstractive summarization class using multi-modal LLMs."""
    
    def __init__(self, model: str = "gpt-4o", max_tokens: int = 500):
        """Initialize the abstractive summarizer.
        
        Args:
            model: The OpenAI model to use
            max_tokens: Maximum tokens for the generated summary
        """
        self.model = model
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
    def _encode_image(self, image_path: str) -> str:
        """Encode an image to base64 for API submission.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64-encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
            
    def _create_visualization(self, text: str) -> Tuple[str, Image.Image]:
        """Create a visualization of the text's key metrics.
        
        Args:
            text: The input text to visualize
            
        Returns:
            Tuple of (path to saved image, PIL Image object)
        """
        # Calculate text metrics
        words = text.split()
        sentences = text.split('.')
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Word length distribution
        word_lengths = [len(word) for word in words]
        ax1.hist(word_lengths, bins=range(1, 20), alpha=0.7)
        ax1.set_title('Word Length Distribution')
        ax1.set_xlabel('Word Length')
        ax1.set_ylabel('Frequency')
        
        # Text statistics
        metrics = ['Words', 'Sentences', 'Avg Word Length']
        values = [len(words), len(sentences), avg_word_length]
        
        ax2.barh(metrics, values, color='skyblue')
        ax2.set_title('Text Statistics')
        ax2.set_xlabel('Count')
        
        plt.tight_layout()
        
        # Save figure to a BytesIO object
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100)
        buf.seek(0)
        
        # Create PIL Image
        img = Image.open(buf)
        
        # Also save to temporary file
        temp_path = 'temp_visualization.png'
        fig.savefig(temp_path, format='png', dpi=100)
        plt.close(fig)
        
        return temp_path, img
    
    def summarize(self, text: str, use_visualization: bool = True) -> Dict[str, Any]:
        """Generate an abstractive summary from the input text.
        
        Args:
            text: The text to summarize
            use_visualization: Whether to include text visualization
            
        Returns:
            Dictionary containing the summary and metadata
        """
        messages = [
            {"role": "system", "content": "You are an expert summarizer that creates concise, informative, and coherent summaries of text. For product reviews, focus on the key points about product quality, user experience, and notable features."},
            {"role": "user", "content": []}
        ]
        
        # Add the text to summarize
        messages[1]["content"].append({
            "type": "text",
            "text": f"Create a concise abstractive summary of the following text in your own words. Focus on the main points and sentiment:\n\n{text}"
        })
        
        # Add visualization if requested
        if use_visualization:
            try:
                image_path, image_obj = self._create_visualization(text)
                base64_image = self._encode_image(image_path)
                
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })
                
                messages[1]["content"].append({
                    "type": "text",
                    "text": "Use the visualization to help understand the text characteristics. The visualization shows word length distribution and basic text statistics."
                })
            except Exception as e:
                print(f"Error creating visualization: {e}")
        
        # Generate the summary
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0.3,  # Lower temperature for more deterministic output
        )
        
        summary = response.choices[0].message.content
        
        return {
            'summary': summary,
            'model': self.model,
            'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else None,
            'visualization_used': use_visualization
        }
    
    def get_multi_modal_feedback(self, original_text: str, extractive_summary: str, abstractive_summary: str) -> Dict[str, Any]:
        """Generate multi-modal feedback comparing the original text with both summaries.
        
        Args:
            original_text: The original text
            extractive_summary: The extractive summary
            abstractive_summary: The abstractive summary generated by this class
            
        Returns:
            Dictionary containing the feedback and metadata
        """
        messages = [
            {"role": "system", "content": "You are an expert at evaluating text summaries. You can analyze the quality of summaries and provide detailed feedback on their strengths and weaknesses."},
            {"role": "user", "content": []}
        ]
        
        # Create a visualization comparing the summaries
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate key metrics
        orig_words = len(original_text.split())
        ext_words = len(extractive_summary.split())
        abs_words = len(abstractive_summary.split())
        
        orig_chars = len(original_text)
        ext_chars = len(extractive_summary)
        abs_chars = len(abstractive_summary)
        
        # Compression ratios
        ext_compression = ext_words / orig_words if orig_words > 0 else 0
        abs_compression = abs_words / orig_words if orig_words > 0 else 0
        
        # Plot data
        categories = ['Word Count', 'Character Count', 'Compression Ratio (%)']
        extractive_values = [ext_words, ext_chars, ext_compression * 100]
        abstractive_values = [abs_words, abs_chars, abs_compression * 100]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, extractive_values, width, label='Extractive')
        ax.bar(x + width/2, abstractive_values, width, label='Abstractive')
        
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.set_title('Comparison of Summarization Methods')
        ax.legend()
        
        # Add value labels on top of bars
        for i, v in enumerate(extractive_values):
            ax.text(i - width/2, v + 0.1, f'{v:.1f}', ha='center')
        
        for i, v in enumerate(abstractive_values):
            ax.text(i + width/2, v + 0.1, f'{v:.1f}', ha='center')
        
        plt.tight_layout()
        
        # Save figure to temporary file
        temp_path = 'summary_comparison.png'
        fig.savefig(temp_path, format='png', dpi=100)
        plt.close(fig)
        
        # Add the text prompt
        messages[1]["content"].append({
            "type": "text",
            "text": f"Analyze and compare these two summaries of the original text. Provide detailed feedback on their effectiveness, accuracy, and readability.\n\nORIGINAL TEXT:\n{original_text}\n\nEXTRACTIVE SUMMARY:\n{extractive_summary}\n\nABSTRACTIVE SUMMARY:\n{abstractive_summary}"
        })
        
        # Add the comparison visualization
        try:
            base64_image = self._encode_image(temp_path)
            
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
            })
            
            messages[1]["content"].append({
                "type": "text",
                "text": "Use the visualization above to help with your analysis. It shows key metrics comparing the two summarization methods."
            })
        except Exception as e:
            print(f"Error adding comparison visualization: {e}")
        
        # Generate the feedback
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens * 2,  # Allow more tokens for detailed feedback
            temperature=0.3,
        )
        
        feedback = response.choices[0].message.content
        
        return {
            'feedback': feedback,
            'model': self.model,
            'tokens_used': response.usage.total_tokens if hasattr(response, 'usage') else None,
            'visualization_path': temp_path
        }
