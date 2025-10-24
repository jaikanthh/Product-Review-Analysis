"""
Dashboard Configuration Module

This module provides configuration settings and utilities for the analytics dashboard.
"""

from dataclasses import dataclass
from typing import Dict, List, Any
import plotly.express as px

@dataclass
class ChartConfig:
    """Configuration for chart styling and behavior."""
    
    height: int = 400
    width: int = None
    color_palette: List[str] = None
    template: str = "plotly_white"
    font_family: str = "Arial, sans-serif"
    font_size: int = 12
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = px.colors.qualitative.Set3

@dataclass
class DashboardTheme:
    """Theme configuration for the dashboard."""
    
    primary_color: str = "#1f77b4"
    secondary_color: str = "#ff7f0e"
    success_color: str = "#28a745"
    warning_color: str = "#ffc107"
    error_color: str = "#dc3545"
    background_color: str = "#ffffff"
    sidebar_color: str = "#f0f2f6"
    
    # CSS styles
    custom_css: str = """
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 8px 8px 0px 0px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    
    .quality-score-high {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .quality-score-medium {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .quality-score-low {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2em;
    }
    
    .pipeline-status {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .pipeline-healthy {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .pipeline-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    
    .pipeline-error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    .feature-importance {
        background: linear-gradient(90deg, #1f77b4 0%, #e8f4f8 100%);
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        margin: 0.1rem 0;
    }
    
    .data-quality-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        margin: 0.25rem 0;
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        border-left: 3px solid #1f77b4;
    }
    
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
    </style>
    """

@dataclass
class DashboardConfig:
    """Main configuration class for the dashboard."""
    
    # Page configuration
    page_title: str = "Review Insights Platform Dashboard"
    page_icon: str | None = None
    layout: str = "wide"
    sidebar_state: str = "expanded"
    
    # Data settings
    default_num_reviews: int = 1000
    max_reviews: int = 10000
    auto_refresh: bool = False
    refresh_interval: int = 30  # seconds
    
    # Chart configuration
    chart_config: ChartConfig = None
    
    # Theme configuration
    theme: DashboardTheme = None
    
    # Feature flags
    enable_real_time: bool = False
    enable_advanced_analytics: bool = True
    enable_data_export: bool = True
    enable_pipeline_monitoring: bool = True
    
    # Performance settings
    max_display_rows: int = 1000
    chart_animation: bool = True
    lazy_loading: bool = True
    
    def __post_init__(self):
        if self.chart_config is None:
            self.chart_config = ChartConfig()
        if self.theme is None:
            self.theme = DashboardTheme()

class DashboardMetrics:
    """Utility class for calculating dashboard metrics."""
    
    @staticmethod
    def calculate_sentiment_metrics(data):
        """Calculate sentiment-related metrics."""
        if 'sentiment_score' not in data.columns:
            return {}
        
        return {
            'avg_sentiment': data['sentiment_score'].mean(),
            'sentiment_std': data['sentiment_score'].std(),
            'positive_ratio': (data['sentiment_score'] > 0.1).mean(),
            'negative_ratio': (data['sentiment_score'] < -0.1).mean(),
            'neutral_ratio': (abs(data['sentiment_score']) <= 0.1).mean()
        }
    
    @staticmethod
    def calculate_rating_metrics(data):
        """Calculate rating-related metrics."""
        if 'rating' not in data.columns:
            return {}
        
        return {
            'avg_rating': data['rating'].mean(),
            'rating_std': data['rating'].std(),
            'rating_distribution': data['rating'].value_counts().to_dict(),
            'high_rating_ratio': (data['rating'] >= 4).mean(),
            'low_rating_ratio': (data['rating'] <= 2).mean()
        }
    
    @staticmethod
    def calculate_quality_metrics(quality_report):
        """Calculate data quality metrics."""
        if not quality_report:
            return {}
        
        return {
            'overall_score': quality_report.overall_score,
            'total_checks': quality_report.total_checks,
            'passed_checks': quality_report.passed_checks,
            'failed_checks': len(quality_report.failed_checks),
            'critical_issues': len([c for c in quality_report.failed_checks 
                                  if c.severity.value == 'critical']),
            'warning_issues': len([c for c in quality_report.failed_checks 
                                 if c.severity.value == 'warning'])
        }
    
    @staticmethod
    def calculate_pipeline_metrics(storage_stats, transformation_result):
        """Calculate pipeline performance metrics."""
        metrics = {
            'data_processed': 0,
            'processing_time': 0,
            'storage_efficiency': 0,
            'transformation_success_rate': 0
        }
        
        if storage_stats:
            metrics['data_processed'] = sum(
                stats.get('total_records', 0) 
                for stats in storage_stats.values()
            )
        
        if transformation_result:
            metrics['processing_time'] = getattr(transformation_result, 'processing_time', 0)
            metrics['transformation_success_rate'] = 1.0  # Assume success if result exists
        
        return metrics

class DashboardUtils:
    """Utility functions for the dashboard."""
    
    @staticmethod
    def format_number(value, precision=2):
        """Format numbers for display."""
        if value >= 1_000_000:
            return f"{value/1_000_000:.{precision}f}M"
        elif value >= 1_000:
            return f"{value/1_000:.{precision}f}K"
        else:
            return f"{value:.{precision}f}"
    
    @staticmethod
    def get_status_color(status):
        """Get color for status indicators."""
        status_colors = {
            'healthy': '#28a745',
            'warning': '#ffc107',
            'error': '#dc3545',
            'unknown': '#6c757d'
        }
        return status_colors.get(status.lower(), status_colors['unknown'])
    
    @staticmethod
    def get_sentiment_color(sentiment_score):
        """Get color for sentiment scores."""
        if sentiment_score > 0.1:
            return '#28a745'  # Positive - green
        elif sentiment_score < -0.1:
            return '#dc3545'  # Negative - red
        else:
            return '#6c757d'  # Neutral - gray
    
    @staticmethod
    def get_quality_score_class(score):
        """Get CSS class for quality scores."""
        if score >= 80:
            return 'quality-score-high'
        elif score >= 60:
            return 'quality-score-medium'
        else:
            return 'quality-score-low'
    
    @staticmethod
    def truncate_text(text, max_length=100):
        """Truncate text for display."""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."
    
    @staticmethod
    def create_download_link(data, filename, file_format='csv'):
        """Create a download link for data."""
        if file_format.lower() == 'csv':
            csv_data = data.to_csv(index=False)
            return csv_data
        elif file_format.lower() == 'json':
            json_data = data.to_json(orient='records', indent=2)
            return json_data
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

# Global configuration instance
dashboard_config = DashboardConfig()