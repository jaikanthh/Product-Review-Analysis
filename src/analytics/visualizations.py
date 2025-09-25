"""
Advanced Visualizations Module

This module provides sophisticated visualization components for the analytics dashboard,
including interactive charts, statistical plots, and business intelligence visualizations.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class VisualizationConfig:
    """Configuration for visualization styling and behavior."""
    
    color_palette: List[str] = None
    template: str = "plotly_white"
    height: int = 400
    width: int = None
    font_family: str = "Arial, sans-serif"
    font_size: int = 12
    show_legend: bool = True
    animation: bool = True
    
    def __post_init__(self):
        if self.color_palette is None:
            self.color_palette = px.colors.qualitative.Set3

class SentimentVisualizations:
    """Specialized visualizations for sentiment analysis."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
    
    def sentiment_distribution_pie(self, data: pd.DataFrame) -> go.Figure:
        """Create a pie chart showing sentiment distribution."""
        if data is None or 'sentiment_label' not in data.columns:
            return self._create_empty_chart("Sentiment data not available")
        
        sentiment_counts = data['sentiment_label'].value_counts()
        
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="Sentiment Distribution",
            color_discrete_sequence=self.config.color_palette,
            template=self.config.template
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(
            height=self.config.height,
            font_family=self.config.font_family,
            font_size=self.config.font_size
        )
        
        return fig
    
    def sentiment_score_histogram(self, data: pd.DataFrame) -> go.Figure:
        """Create a histogram of sentiment scores."""
        if data is None or 'sentiment_score' not in data.columns:
            return self._create_empty_chart("Sentiment score data not available")
        
        fig = px.histogram(
            data,
            x='sentiment_score',
            title="Sentiment Score Distribution",
            nbins=50,
            color_discrete_sequence=self.config.color_palette,
            template=self.config.template
        )
        
        # Add vertical lines for sentiment boundaries
        fig.add_vline(x=-0.1, line_dash="dash", line_color="red", 
                     annotation_text="Negative Threshold")
        fig.add_vline(x=0.1, line_dash="dash", line_color="green", 
                     annotation_text="Positive Threshold")
        
        fig.update_layout(
            height=self.config.height,
            xaxis_title="Sentiment Score",
            yaxis_title="Frequency",
            font_family=self.config.font_family,
            font_size=self.config.font_size
        )
        
        return fig
    
    def sentiment_over_time(self, data: pd.DataFrame) -> go.Figure:
        """Create a time series plot of sentiment trends."""
        if data is None or 'timestamp' not in data.columns or 'sentiment_score' not in data.columns:
            return self._create_empty_chart("Timestamp or sentiment data not available")
        
        # Prepare time series data
        data_copy = data.copy()
        data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
        data_copy['date'] = data_copy['timestamp'].dt.date
        
        # Calculate daily sentiment metrics
        daily_sentiment = data_copy.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count']
        }).round(3)
        
        daily_sentiment.columns = ['avg_sentiment', 'sentiment_std', 'review_count']
        daily_sentiment = daily_sentiment.reset_index()
        
        # Create subplot with secondary y-axis
        fig = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=["Sentiment Trends Over Time"]
        )
        
        # Add sentiment line
        fig.add_trace(
            go.Scatter(
                x=daily_sentiment['date'],
                y=daily_sentiment['avg_sentiment'],
                mode='lines+markers',
                name='Average Sentiment',
                line=dict(color=self.config.color_palette[0], width=3),
                hovertemplate='<b>Date:</b> %{x}<br><b>Avg Sentiment:</b> %{y:.3f}<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Add confidence band
        fig.add_trace(
            go.Scatter(
                x=daily_sentiment['date'],
                y=daily_sentiment['avg_sentiment'] + daily_sentiment['sentiment_std'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_sentiment['date'],
                y=daily_sentiment['avg_sentiment'] - daily_sentiment['sentiment_std'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.2)',
                name='Confidence Band',
                hoverinfo='skip'
            ),
            secondary_y=False
        )
        
        # Add review count bars
        fig.add_trace(
            go.Bar(
                x=daily_sentiment['date'],
                y=daily_sentiment['review_count'],
                name='Review Count',
                opacity=0.3,
                marker_color=self.config.color_palette[1],
                hovertemplate='<b>Date:</b> %{x}<br><b>Reviews:</b> %{y}<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Average Sentiment Score", secondary_y=False)
        fig.update_yaxes(title_text="Number of Reviews", secondary_y=True)
        
        fig.update_layout(
            height=self.config.height,
            template=self.config.template,
            font_family=self.config.font_family,
            font_size=self.config.font_size,
            hovermode='x unified'
        )
        
        return fig
    
    def sentiment_by_rating(self, data: pd.DataFrame) -> go.Figure:
        """Create a box plot showing sentiment distribution by rating."""
        if data is None or 'rating' not in data.columns or 'sentiment_score' not in data.columns:
            return self._create_empty_chart("Rating or sentiment data not available")
        
        fig = px.box(
            data,
            x='rating',
            y='sentiment_score',
            title="Sentiment Score Distribution by Rating",
            color='rating',
            color_discrete_sequence=self.config.color_palette,
            template=self.config.template
        )
        
        fig.update_layout(
            height=self.config.height,
            xaxis_title="Rating",
            yaxis_title="Sentiment Score",
            font_family=self.config.font_family,
            font_size=self.config.font_size,
            showlegend=False
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=self.config.height,
            template=self.config.template,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

class RatingVisualizations:
    """Specialized visualizations for rating analysis."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
    
    def rating_distribution_bar(self, data: pd.DataFrame) -> go.Figure:
        """Create a bar chart showing rating distribution."""
        if data is None or 'rating' not in data.columns:
            return self._create_empty_chart("Rating data not available")
        
        rating_counts = data['rating'].value_counts().sort_index()
        
        fig = px.bar(
            x=rating_counts.index,
            y=rating_counts.values,
            title="Rating Distribution",
            labels={'x': 'Rating', 'y': 'Number of Reviews'},
            color=rating_counts.index,
            color_continuous_scale='RdYlGn',
            template=self.config.template
        )
        
        # Add percentage annotations
        total_reviews = rating_counts.sum()
        for i, (rating, count) in enumerate(rating_counts.items()):
            percentage = (count / total_reviews) * 100
            fig.add_annotation(
                x=rating,
                y=count,
                text=f"{percentage:.1f}%",
                showarrow=False,
                yshift=10
            )
        
        fig.update_layout(
            height=self.config.height,
            font_family=self.config.font_family,
            font_size=self.config.font_size,
            showlegend=False
        )
        
        return fig
    
    def rating_trends_over_time(self, data: pd.DataFrame) -> go.Figure:
        """Create a time series plot of rating trends."""
        if data is None or 'timestamp' not in data.columns or 'rating' not in data.columns:
            return self._create_empty_chart("Timestamp or rating data not available")
        
        # Prepare time series data
        data_copy = data.copy()
        data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
        data_copy['date'] = data_copy['timestamp'].dt.date
        
        # Calculate daily rating metrics
        daily_ratings = data_copy.groupby('date').agg({
            'rating': ['mean', 'count', 'std']
        }).round(3)
        
        daily_ratings.columns = ['avg_rating', 'review_count', 'rating_std']
        daily_ratings = daily_ratings.reset_index()
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Average Rating Over Time", "Review Volume Over Time"],
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Add average rating line
        fig.add_trace(
            go.Scatter(
                x=daily_ratings['date'],
                y=daily_ratings['avg_rating'],
                mode='lines+markers',
                name='Average Rating',
                line=dict(color=self.config.color_palette[0], width=3),
                hovertemplate='<b>Date:</b> %{x}<br><b>Avg Rating:</b> %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add review count bars
        fig.add_trace(
            go.Bar(
                x=daily_ratings['date'],
                y=daily_ratings['review_count'],
                name='Review Count',
                marker_color=self.config.color_palette[1],
                hovertemplate='<b>Date:</b> %{x}<br><b>Reviews:</b> %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=self.config.height * 1.5,
            template=self.config.template,
            font_family=self.config.font_family,
            font_size=self.config.font_size,
            showlegend=True
        )
        
        return fig
    
    def rating_by_category(self, data: pd.DataFrame) -> go.Figure:
        """Create a horizontal bar chart of average ratings by category."""
        if data is None or 'category' not in data.columns or 'rating' not in data.columns:
            return self._create_empty_chart("Category or rating data not available")
        
        category_ratings = data.groupby('category').agg({
            'rating': ['mean', 'count', 'std']
        }).round(3)
        
        category_ratings.columns = ['avg_rating', 'review_count', 'rating_std']
        category_ratings = category_ratings.sort_values('avg_rating', ascending=True)
        
        fig = go.Figure()
        
        # Add bars with error bars
        fig.add_trace(
            go.Bar(
                y=category_ratings.index,
                x=category_ratings['avg_rating'],
                orientation='h',
                error_x=dict(
                    type='data',
                    array=category_ratings['rating_std'],
                    visible=True
                ),
                marker_color=self.config.color_palette[0],
                hovertemplate='<b>%{y}</b><br>Avg Rating: %{x:.2f}<br>Reviews: %{customdata}<extra></extra>',
                customdata=category_ratings['review_count']
            )
        )
        
        fig.update_layout(
            title="Average Rating by Category",
            xaxis_title="Average Rating",
            yaxis_title="Category",
            height=max(self.config.height, len(category_ratings) * 30),
            template=self.config.template,
            font_family=self.config.font_family,
            font_size=self.config.font_size
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=self.config.height,
            template=self.config.template,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

class QualityVisualizations:
    """Specialized visualizations for data quality analysis."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
    
    def quality_score_gauge(self, overall_score: float) -> go.Figure:
        """Create a gauge chart for overall quality score."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=overall_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Overall Data Quality Score"},
            delta={'reference': 80, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': self._get_quality_color(overall_score)},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=self.config.height,
            template=self.config.template,
            font_family=self.config.font_family,
            font_size=self.config.font_size
        )
        
        return fig
    
    def quality_scores_by_type(self, scores_by_type: Dict[str, float]) -> go.Figure:
        """Create a radar chart showing quality scores by check type."""
        if not scores_by_type:
            return self._create_empty_chart("Quality scores not available")
        
        categories = list(scores_by_type.keys())
        values = list(scores_by_type.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Quality Scores',
            line_color=self.config.color_palette[0],
            fillcolor=f'rgba(31, 119, 180, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title="Data Quality Scores by Check Type",
            height=self.config.height,
            template=self.config.template,
            font_family=self.config.font_family,
            font_size=self.config.font_size
        )
        
        return fig
    
    def quality_issues_breakdown(self, failed_checks: List) -> go.Figure:
        """Create a breakdown of quality issues by severity."""
        if not failed_checks:
            return self._create_empty_chart("No quality issues found")
        
        # Count issues by severity
        severity_counts = {}
        for check in failed_checks:
            severity = check.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Define colors for severities
        severity_colors = {
            'critical': '#dc3545',
            'warning': '#ffc107',
            'info': '#17a2b8'
        }
        
        fig = px.bar(
            x=list(severity_counts.keys()),
            y=list(severity_counts.values()),
            title="Quality Issues by Severity",
            labels={'x': 'Severity', 'y': 'Number of Issues'},
            color=list(severity_counts.keys()),
            color_discrete_map=severity_colors,
            template=self.config.template
        )
        
        fig.update_layout(
            height=self.config.height,
            font_family=self.config.font_family,
            font_size=self.config.font_size,
            showlegend=False
        )
        
        return fig
    
    def _get_quality_color(self, score: float) -> str:
        """Get color based on quality score."""
        if score >= 80:
            return "green"
        elif score >= 60:
            return "yellow"
        else:
            return "red"
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=self.config.height,
            template=self.config.template,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

class FeatureVisualizations:
    """Specialized visualizations for feature engineering analysis."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
    
    def feature_correlation_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Create a correlation heatmap for numerical features."""
        if data is None:
            return self._create_empty_chart("Data not available for correlation analysis")
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return self._create_empty_chart("Insufficient numerical features for correlation analysis")
        
        correlation_matrix = data[numeric_columns].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto',
            template=self.config.template
        )
        
        fig.update_layout(
            height=max(self.config.height, len(numeric_columns) * 25),
            font_family=self.config.font_family,
            font_size=self.config.font_size
        )
        
        return fig
    
    def feature_importance_bar(self, feature_importance: Dict[str, float]) -> go.Figure:
        """Create a bar chart showing feature importance scores."""
        if not feature_importance:
            return self._create_empty_chart("Feature importance data not available")
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_features[:20])  # Top 20 features
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title="Top 20 Feature Importance Scores",
            labels={'x': 'Importance Score', 'y': 'Feature'},
            color=importance,
            color_continuous_scale='Viridis',
            template=self.config.template
        )
        
        fig.update_layout(
            height=max(self.config.height, len(features) * 25),
            font_family=self.config.font_family,
            font_size=self.config.font_size,
            showlegend=False
        )
        
        return fig
    
    def feature_distribution_comparison(self, data: pd.DataFrame, features: List[str]) -> go.Figure:
        """Create distribution plots for selected features."""
        if data is None or not features or not all(f in data.columns for f in features):
            return self._create_empty_chart("Selected features not available in data")
        
        # Limit to first 4 features for readability
        features = features[:4]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=features,
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, feature in enumerate(features):
            if i >= 4:
                break
            
            row, col = positions[i]
            
            fig.add_trace(
                go.Histogram(
                    x=data[feature],
                    name=feature,
                    nbinsx=30,
                    marker_color=self.config.color_palette[i % len(self.config.color_palette)],
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Feature Distribution Comparison",
            height=self.config.height * 1.5,
            template=self.config.template,
            font_family=self.config.font_family,
            font_size=self.config.font_size
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=self.config.height,
            template=self.config.template,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

class BusinessIntelligenceVisualizations:
    """Business intelligence and KPI visualizations."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
    
    def kpi_dashboard(self, metrics: Dict[str, float]) -> go.Figure:
        """Create a KPI dashboard with key metrics."""
        if not metrics:
            return self._create_empty_chart("KPI metrics not available")
        
        # Create a 2x2 grid of KPI indicators
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]],
            vertical_spacing=0.3
        )
        
        kpi_list = list(metrics.items())[:4]  # Take first 4 KPIs
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, (kpi_name, value) in enumerate(kpi_list):
            row, col = positions[i]
            
            fig.add_trace(
                go.Indicator(
                    mode="number+gauge",
                    value=value,
                    title={"text": kpi_name.replace('_', ' ').title()},
                    gauge={
                        'axis': {'range': [None, max(100, value * 1.2)]},
                        'bar': {'color': self.config.color_palette[i % len(self.config.color_palette)]},
                        'steps': [{'range': [0, value * 0.8], 'color': "lightgray"}],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': value * 0.9}
                    }
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Key Performance Indicators",
            height=self.config.height * 1.5,
            template=self.config.template,
            font_family=self.config.font_family,
            font_size=self.config.font_size
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=self.config.height,
            template=self.config.template,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig