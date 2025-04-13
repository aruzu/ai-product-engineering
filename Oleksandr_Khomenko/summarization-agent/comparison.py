"""Module for comparing extractive and abstractive summarization methods."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

class SummaryComparison:
    """Class for comparing and analyzing different summarization methods."""
    
    def __init__(self, output_dir: str = "outputs"):
        """Initialize the comparison module.
        
        Args:
            output_dir: Directory to save comparison outputs
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def compare_summaries(self, original_text: str, extractive_summary: Dict[str, Any], 
                         abstractive_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Compare extractive and abstractive summaries.
        
        Args:
            original_text: The original text
            extractive_summary: Results from the extractive summarizer
            abstractive_summary: Results from the abstractive summarizer
            
        Returns:
            Dictionary containing comparison metrics and analysis
        """
        # Extract the summary texts
        ext_text = extractive_summary['summary']
        abs_text = abstractive_summary['summary']
        
        # Calculate basic metrics
        original_length = len(original_text)
        original_words = len(original_text.split())
        
        ext_length = len(ext_text)
        ext_words = len(ext_text.split())
        
        abs_length = len(abs_text)
        abs_words = len(abs_text.split())
        
        # Calculate compression ratios
        ext_compression_ratio = ext_length / original_length if original_length > 0 else 0
        abs_compression_ratio = abs_length / original_length if original_length > 0 else 0
        
        ext_word_ratio = ext_words / original_words if original_words > 0 else 0
        abs_word_ratio = abs_words / original_words if original_words > 0 else 0
        
        # Create a comparison report
        comparison = {
            'original_text': {
                'length': original_length,
                'word_count': original_words
            },
            'extractive_summary': {
                'length': ext_length,
                'word_count': ext_words,
                'compression_ratio': ext_compression_ratio,
                'word_ratio': ext_word_ratio,
                **extractive_summary  # Include all extractive summary metadata
            },
            'abstractive_summary': {
                'length': abs_length,
                'word_count': abs_words,
                'compression_ratio': abs_compression_ratio,
                'word_ratio': abs_word_ratio,
                **abstractive_summary  # Include all abstractive summary metadata
            }
        }
        
        return comparison
    
    def generate_report(self, comparison_results: Dict[str, Any], sample_id: str = "sample") -> str:
        """Generate a detailed comparison report and visualizations.
        
        Args:
            comparison_results: Results from compare_summaries method
            sample_id: Identifier for the current sample
            
        Returns:
            Path to the saved report
        """
        # Extract key metrics for visualization
        metrics = {
            'Compression Ratio': {
                'Extractive': comparison_results['extractive_summary']['compression_ratio'],
                'Abstractive': comparison_results['abstractive_summary']['compression_ratio']
            },
            'Word Ratio': {
                'Extractive': comparison_results['extractive_summary']['word_ratio'],
                'Abstractive': comparison_results['abstractive_summary']['word_ratio']
            },
            'Character Count': {
                'Extractive': comparison_results['extractive_summary']['length'],
                'Abstractive': comparison_results['abstractive_summary']['length'],
                'Original': comparison_results['original_text']['length']
            },
            'Word Count': {
                'Extractive': comparison_results['extractive_summary']['word_count'],
                'Abstractive': comparison_results['abstractive_summary']['word_count'],
                'Original': comparison_results['original_text']['word_count']
            }
        }
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Summary Comparison Report - Sample {sample_id}', fontsize=16)
        
        # Plot compression metrics
        self._plot_comparison_bars(axes[0, 0], 
                                  [metrics['Compression Ratio']['Extractive'], metrics['Compression Ratio']['Abstractive']],
                                  ['Extractive', 'Abstractive'],
                                  'Compression Ratio (lower is better)',
                                  'Ratio')
        
        self._plot_comparison_bars(axes[0, 1], 
                                  [metrics['Word Ratio']['Extractive'], metrics['Word Ratio']['Abstractive']],
                                  ['Extractive', 'Abstractive'],
                                  'Word Ratio (lower is better)',
                                  'Ratio')
        
        # Plot counts with original included
        self._plot_comparison_bars(axes[1, 0], 
                                  [metrics['Character Count']['Original'], 
                                   metrics['Character Count']['Extractive'], 
                                   metrics['Character Count']['Abstractive']],
                                  ['Original', 'Extractive', 'Abstractive'],
                                  'Character Count',
                                  'Characters')
        
        self._plot_comparison_bars(axes[1, 1], 
                                  [metrics['Word Count']['Original'], 
                                   metrics['Word Count']['Extractive'], 
                                   metrics['Word Count']['Abstractive']],
                                  ['Original', 'Extractive', 'Abstractive'],
                                  'Word Count',
                                  'Words')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for title
        
        # Save the visualization
        report_path = os.path.join(self.output_dir, f'summary_comparison_{sample_id}.png')
        fig.savefig(report_path, dpi=300)
        plt.close(fig)
        
        # Generate text report
        text_report_path = os.path.join(self.output_dir, f'summary_comparison_{sample_id}.txt')
        with open(text_report_path, 'w') as f:
            f.write(f"SUMMARY COMPARISON REPORT - SAMPLE {sample_id}\n")
            f.write("="*50 + "\n\n")
            
            f.write("ORIGINAL TEXT:\n")
            f.write(f"Length: {comparison_results['original_text']['length']} characters\n")
            f.write(f"Word Count: {comparison_results['original_text']['word_count']} words\n\n")
            
            f.write("EXTRACTIVE SUMMARY:\n")
            f.write(f"Length: {comparison_results['extractive_summary']['length']} characters")
            f.write(f" ({comparison_results['extractive_summary']['compression_ratio']:.2%} of original)\n")
            f.write(f"Word Count: {comparison_results['extractive_summary']['word_count']} words")
            f.write(f" ({comparison_results['extractive_summary']['word_ratio']:.2%} of original)\n")
            if 'method' in comparison_results['extractive_summary']:
                f.write(f"Method: {comparison_results['extractive_summary']['method']}\n")
            f.write("\n")
            
            f.write("ABSTRACTIVE SUMMARY:\n")
            f.write(f"Length: {comparison_results['abstractive_summary']['length']} characters")
            f.write(f" ({comparison_results['abstractive_summary']['compression_ratio']:.2%} of original)\n")
            f.write(f"Word Count: {comparison_results['abstractive_summary']['word_count']} words")
            f.write(f" ({comparison_results['abstractive_summary']['word_ratio']:.2%} of original)\n")
            if 'model' in comparison_results['abstractive_summary']:
                f.write(f"Model: {comparison_results['abstractive_summary']['model']}\n")
            if 'tokens_used' in comparison_results['abstractive_summary']:
                f.write(f"Tokens Used: {comparison_results['abstractive_summary']['tokens_used']}\n")
            f.write("\n")
            
            f.write("COMPARISON:\n")
            ext_ratio = comparison_results['extractive_summary']['compression_ratio']
            abs_ratio = comparison_results['abstractive_summary']['compression_ratio']
            ratio_diff = abs(ext_ratio - abs_ratio)
            f.write(f"Compression Ratio Difference: {ratio_diff:.2%}\n")
            f.write(f"More Concise Method: {'Abstractive' if abs_ratio < ext_ratio else 'Extractive'}\n\n")
            
            # Add the actual summaries
            f.write("SUMMARY TEXTS:\n\n")
            f.write("Extractive Summary:\n")
            f.write(comparison_results['extractive_summary']['summary'] + "\n\n")
            f.write("Abstractive Summary:\n")
            f.write(comparison_results['abstractive_summary']['summary'] + "\n")
        
        return report_path
    
    def _plot_comparison_bars(self, ax, values, labels, title, ylabel):
        """Helper method to create comparison bar charts.
        
        Args:
            ax: Matplotlib axis to plot on
            values: Values to plot
            labels: Labels for x-axis
            title: Plot title
            ylabel: Y-axis label
        """
        colors = ['#3498db', '#2ecc71', '#e74c3c'][:len(values)]
        bars = ax.bar(np.arange(len(values)), values, color=colors)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        ax.set_xticks(np.arange(len(values)))
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)

    def save_aggregate_report(self, all_comparisons: List[Dict[str, Any]], output_file: str = "aggregate_report.csv") -> str:
        """Save an aggregate report of multiple comparisons.
        
        Args:
            all_comparisons: List of comparison results
            output_file: Filename for the aggregate report
            
        Returns:
            Path to the saved report
        """
        # Prepare data for the report
        data = []
        
        for i, comp in enumerate(all_comparisons):
            data.append({
                'sample_id': i,
                'original_length': comp['original_text']['length'],
                'original_words': comp['original_text']['word_count'],
                'extractive_length': comp['extractive_summary']['length'],
                'extractive_words': comp['extractive_summary']['word_count'],
                'extractive_compression': comp['extractive_summary']['compression_ratio'],
                'abstractive_length': comp['abstractive_summary']['length'],
                'abstractive_words': comp['abstractive_summary']['word_count'],
                'abstractive_compression': comp['abstractive_summary']['compression_ratio'],
                'more_concise': 'Abstractive' if comp['abstractive_summary']['compression_ratio'] < comp['extractive_summary']['compression_ratio'] else 'Extractive'
            })
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        report_path = os.path.join(self.output_dir, output_file)
        df.to_csv(report_path, index=False)
        
        # Generate summary statistics
        summary_path = os.path.join(self.output_dir, "summary_statistics.txt")
        with open(summary_path, 'w') as f:
            f.write("SUMMARY COMPARISON - AGGREGATE STATISTICS\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Number of samples: {len(all_comparisons)}\n\n")
            
            # Compression ratio stats
            ext_compression = [comp['extractive_summary']['compression_ratio'] for comp in all_comparisons]
            abs_compression = [comp['abstractive_summary']['compression_ratio'] for comp in all_comparisons]
            
            f.write("COMPRESSION RATIO STATISTICS:\n")
            f.write(f"Extractive - Mean: {np.mean(ext_compression):.4f}, Median: {np.median(ext_compression):.4f}, ")
            f.write(f"Min: {np.min(ext_compression):.4f}, Max: {np.max(ext_compression):.4f}\n")
            
            f.write(f"Abstractive - Mean: {np.mean(abs_compression):.4f}, Median: {np.median(abs_compression):.4f}, ")
            f.write(f"Min: {np.min(abs_compression):.4f}, Max: {np.max(abs_compression):.4f}\n\n")
            
            # Count which method was more concise
            more_concise_counts = df['more_concise'].value_counts()
            f.write("MORE CONCISE METHOD COUNTS:\n")
            for method, count in more_concise_counts.items():
                f.write(f"{method}: {count} ({count/len(df):.2%})\n")
        
        return report_path
