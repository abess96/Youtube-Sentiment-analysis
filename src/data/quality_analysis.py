"""
Data Quality Analysis Script

This script generates comprehensive quality analysis reports from the enhanced
data preprocessing pipeline results.
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger('quality_analysis')
logger.setLevel('INFO')

def load_quality_reports():
    """Load quality reports from enhanced preprocessing."""
    try:
        with open('data/interim/train_quality_report.json', 'r') as f:
            train_report = json.load(f)
        
        with open('data/interim/test_quality_report.json', 'r') as f:
            test_report = json.load(f)
        
        return train_report, test_report
    except FileNotFoundError as e:
        logger.error(f"Quality reports not found: {e}")
        return None, None

def generate_quality_metrics(train_report, test_report):
    """Generate consolidated quality metrics."""
    metrics = {
        'train_metrics': {
            'original_records': train_report['processing_summary']['original_records'],
            'filtered_records': train_report['processing_summary']['filtered_records'],
            'filter_rate': train_report['processing_summary']['filter_rate'],
            'avg_quality_score': train_report['quality_metrics']['average_quality_score'],
            'spam_rate': train_report['spam_detection']['spam_rate']
        },
        'test_metrics': {
            'original_records': test_report['processing_summary']['original_records'],
            'filtered_records': test_report['processing_summary']['filtered_records'],
            'filter_rate': test_report['processing_summary']['filter_rate'],
            'avg_quality_score': test_report['quality_metrics']['average_quality_score'],
            'spam_rate': test_report['spam_detection']['spam_rate']
        }
    }
    
    return metrics

def create_quality_visualizations(train_report, test_report):
    """Create quality analysis visualizations."""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Data Quality Analysis Report', fontsize=16, fontweight='bold')
    
    # Plot 1: Filter rates comparison
    datasets = ['Training', 'Test']
    filter_rates = [
        train_report['processing_summary']['filter_rate'] * 100,
        test_report['processing_summary']['filter_rate'] * 100
    ]
    
    axes[0, 0].bar(datasets, filter_rates, color=['skyblue', 'lightcoral'])
    axes[0, 0].set_title('Data Filter Rates (%)')
    axes[0, 0].set_ylabel('Filter Rate (%)')
    
    # Add value labels on bars
    for i, v in enumerate(filter_rates):
        axes[0, 0].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom')
    
    return fig

def generate_html_report(metrics, output_path):
    """Generate HTML report from quality metrics."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Quality Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .metrics {{ display: flex; justify-content: space-between; margin: 20px 0; }}
            .metric-box {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; width: 45%; }}
            .metric-title {{ font-weight: bold; color: #2c3e50; }}
            .metric-value {{ font-size: 1.2em; color: #3498db; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Data Quality Analysis Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="metrics">
            <div class="metric-box">
                <div class="metric-title">Training Data</div>
                <p>Original Records: <span class="metric-value">{metrics['train_metrics']['original_records']:,}</span></p>
                <p>Filtered Records: <span class="metric-value">{metrics['train_metrics']['filtered_records']:,}</span></p>
                <p>Filter Rate: <span class="metric-value">{metrics['train_metrics']['filter_rate']:.2%}</span></p>
                <p>Avg Quality Score: <span class="metric-value">{metrics['train_metrics']['avg_quality_score']:.3f}</span></p>
                <p>Spam Rate: <span class="metric-value">{metrics['train_metrics']['spam_rate']:.2%}</span></p>
            </div>
            
            <div class="metric-box">
                <div class="metric-title">Test Data</div>
                <p>Original Records: <span class="metric-value">{metrics['test_metrics']['original_records']:,}</span></p>
                <p>Filtered Records: <span class="metric-value">{metrics['test_metrics']['filtered_records']:,}</span></p>
                <p>Filter Rate: <span class="metric-value">{metrics['test_metrics']['filter_rate']:.2%}</span></p>
                <p>Avg Quality Score: <span class="metric-value">{metrics['test_metrics']['avg_quality_score']:.3f}</span></p>
                <p>Spam Rate: <span class="metric-value">{metrics['test_metrics']['spam_rate']:.2%}</span></p>
            </div>
        </div>
        
        <h2>Summary</h2>
        <table>
            <tr>
                <th>Dataset</th>
                <th>Original Records</th>
                <th>Filtered Records</th>
                <th>Filter Rate</th>
                <th>Quality Score</th>
                <th>Spam Rate</th>
            </tr>
            <tr>
                <td>Training</td>
                <td>{metrics['train_metrics']['original_records']:,}</td>
                <td>{metrics['train_metrics']['filtered_records']:,}</td>
                <td>{metrics['train_metrics']['filter_rate']:.2%}</td>
                <td>{metrics['train_metrics']['avg_quality_score']:.3f}</td>
                <td>{metrics['train_metrics']['spam_rate']:.2%}</td>
            </tr>
            <tr>
                <td>Test</td>
                <td>{metrics['test_metrics']['original_records']:,}</td>
                <td>{metrics['test_metrics']['filtered_records']:,}</td>
                <td>{metrics['test_metrics']['filter_rate']:.2%}</td>
                <td>{metrics['test_metrics']['avg_quality_score']:.3f}</td>
                <td>{metrics['test_metrics']['spam_rate']:.2%}</td>
            </tr>
        </table>
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def main():
    """Main function to generate quality analysis."""
    try:
        logger.info("Starting data quality analysis...")
        
        # Create output directory
        output_dir = Path('data/reports')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load quality reports
        train_report, test_report = load_quality_reports()
        
        if train_report is None or test_report is None:
            logger.warning("Quality reports not available. Creating minimal report.")
            # Create minimal HTML report when quality reports are missing
            minimal_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Data Quality Analysis Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .warning { background-color: #fff3cd; padding: 20px; border-radius: 5px; border-left: 4px solid #ffc107; }
                </style>
            </head>
            <body>
                <h1>Data Quality Analysis Report</h1>
                <div class="warning">
                    <h3>⚠️ Quality Reports Not Available</h3>
                    <p>The enhanced data preprocessing pipeline quality reports were not found.</p>
                    <p>This may be because the enhanced pipeline is not enabled or the quality reports were not generated.</p>
                    <p>Generated on: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
                </div>
            </body>
            </html>
            """
            
            with open(output_dir / 'quality_analysis.html', 'w', encoding='utf-8') as f:
                f.write(minimal_html)
            
            # Create minimal metrics file
            minimal_metrics = {
                "status": "quality_reports_not_available",
                "message": "Enhanced preprocessing quality reports not found",
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            with open(output_dir / 'quality_metrics.json', 'w') as f:
                json.dump(minimal_metrics, f, indent=2)
            
            logger.info("Minimal quality analysis report created")
            return
        
        # Generate metrics
        metrics = generate_quality_metrics(train_report, test_report)
        
        # Save metrics
        with open(output_dir / 'quality_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Generate HTML report
        generate_html_report(metrics, output_dir / 'quality_analysis.html')
        
        logger.info("Quality analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in quality analysis: {e}")
        raise

if __name__ == "__main__":
    main()