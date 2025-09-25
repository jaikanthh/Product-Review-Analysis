"""
Analytics Dashboard for Product Review Analysis

This module provides an interactive web-based dashboard using Streamlit
to visualize and analyze the complete data engineering pipeline results.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os
import random
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_generation.generate_reviews import ReviewDataGenerator
from storage.storage_manager import StorageManager, StorageConfig
from transformation.transformation_orchestrator import TransformationOrchestrator, TransformationConfig
from ingestion.ingestion_orchestrator import IngestionOrchestrator, IngestionConfig

class DashboardConfig:
    """Configuration for the analytics dashboard."""
    
    def __init__(self):
        self.page_title = "Product Review Analytics Dashboard"
        self.page_icon = "📊"
        self.layout = "wide"
        self.sidebar_state = "expanded"
        
        # Data refresh settings
        self.auto_refresh = True
        self.refresh_interval = 30  # seconds
        
        # Chart settings
        self.color_palette = px.colors.qualitative.Set3
        self.chart_height = 400

class DataPipelineMonitor:
    """Monitor and display data pipeline status and metrics."""
    
    def __init__(self):
        self.storage_manager = None
        self.transformation_orchestrator = None
        self.ingestion_orchestrator = None
    
    def initialize_components(self):
        """Initialize data pipeline components."""
        try:
            # Initialize storage manager
            storage_config = StorageConfig(
                enable_sqlite=True,
                enable_mongodb=False,  # Disable for demo
                enable_data_lake=True
            )
            self.storage_manager = StorageManager(storage_config)
            
            # Initialize transformation orchestrator
            transformation_config = TransformationConfig()
            self.transformation_orchestrator = TransformationOrchestrator(transformation_config)
            
            # Initialize ingestion orchestrator
            ingestion_config = IngestionConfig()
            self.ingestion_orchestrator = IngestionOrchestrator(ingestion_config)
            
            return True
        except Exception as e:
            st.error(f"Failed to initialize pipeline components: {str(e)}")
            return False
    
    def get_pipeline_status(self):
        """Get current pipeline status and health metrics."""
        status = {
            'storage': {'status': 'healthy', 'details': {}},
            'transformation': {'status': 'healthy', 'details': {}},
            'ingestion': {'status': 'healthy', 'details': {}}
        }
        
        try:
            if self.storage_manager:
                health_check = self.storage_manager.health_check()
                status['storage']['details'] = health_check
                status['storage']['status'] = 'healthy' if all(health_check.values()) else 'warning'
        except Exception as e:
            status['storage']['status'] = 'error'
            status['storage']['details'] = {'error': str(e)}
        
        return status

class ReviewAnalyticsDashboard:
    """Main dashboard class for review analytics."""
    
    def __init__(self):
        self.config = DashboardConfig()
        self.monitor = DataPipelineMonitor()
        self.sample_data = None
        # Data will be loaded by sidebar based on slider value
        
    def setup_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title=self.config.page_title,
            page_icon=self.config.page_icon,
            layout=self.config.layout,
            initial_sidebar_state=self.config.sidebar_state
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f77b4;
        }
        .status-healthy {
            color: #28a745;
        }
        .status-warning {
            color: #ffc107;
        }
        .status-error {
            color: #dc3545;
        }
        </style>
        """, unsafe_allow_html=True)

    def load_existing_data(self, num_reviews=1000):
        """Load existing sample data if available."""
        try:
            # Try to load existing CSV data
            data_path = Path(__file__).parent.parent.parent / "data" / "raw" / "reviews" / "reviews.csv"
            if data_path.exists():
                data = pd.read_csv(data_path)
                # Sample the requested number of reviews
                if len(data) > num_reviews:
                    data = data.sample(n=num_reviews, random_state=42)
                
                self.sample_data = data
                st.session_state['sample_data'] = data
                
                # Also try to load transformed data if available
                transformed_path = Path(__file__).parent.parent.parent / "data" / "transformed" / "reviews_transformed.parquet"
                if transformed_path.exists():
                    transformed_data = pd.read_parquet(transformed_path)
                    # Sample transformed data to match
                    if len(transformed_data) > num_reviews:
                        transformed_data = transformed_data.sample(n=num_reviews, random_state=42)
                    st.session_state['transformed_data'] = transformed_data
                    
                    # Create a mock transformation result
                    from types import SimpleNamespace
                    result = SimpleNamespace()
                    result.transformed_data = transformed_data
                    result.quality_report = None
                    st.session_state['transformation_result'] = result
                    
        except Exception as e:
            # If loading fails, continue without data
            pass

    def load_uploaded_data(self, uploaded_file):
        """Load and validate uploaded dataset."""
        try:
            # Read the uploaded file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload CSV or Excel files.")
                return False
            
            # Validate and standardize column names
            df = self._validate_and_standardize_data(df)
            if df is None:
                return False
            
            # Clear any existing transformed data
            if 'transformed_data' in st.session_state:
                del st.session_state['transformed_data']
            if 'transformation_result' in st.session_state:
                del st.session_state['transformation_result']
            
            # Store the raw data
            self.sample_data = df
            st.session_state['sample_data'] = df
            st.session_state['data_source'] = 'uploaded'
            st.session_state['uploaded_filename'] = uploaded_file.name
            
            # Perform sentiment analysis on uploaded data
            with st.spinner("🔍 Performing sentiment analysis..."):
                try:
                    # Initialize transformation orchestrator if not already done
                    if not hasattr(self, 'transformation_orchestrator') or self.transformation_orchestrator is None:
                        transformation_config = TransformationConfig()
                        self.transformation_orchestrator = TransformationOrchestrator(transformation_config)
                    
                    # Transform the data (includes sentiment analysis)
                    result = self.transformation_orchestrator.transform_dataset(
                        df, 
                        dataset_name="uploaded_reviews"
                    )
                    
                    # Store transformed data in session state
                    st.session_state['transformation_result'] = result
                    st.session_state['transformed_data'] = result.transformed_data
                    
                    st.success(f"✅ Successfully loaded and analyzed {len(df)} reviews from {uploaded_file.name}")
                    
                except Exception as e:
                    st.warning(f"⚠️ Data loaded but sentiment analysis failed: {str(e)}")
                    st.info("📊 Basic analytics will still be available without sentiment analysis.")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            return False
    
    def _validate_and_standardize_data(self, df):
        """Validate and standardize the uploaded dataset."""
        try:
            # Check if dataframe is empty
            if df.empty:
                st.error("The uploaded file is empty.")
                return None
            
            # Column mapping for common variations
            column_mapping = {
                # Review text variations
                'text': 'review_text',
                'review': 'review_text',
                'comment': 'review_text',
                'content': 'review_text',
                'description': 'review_text',
                
                # Date variations
                'date': 'created_at',
                'timestamp': 'created_at',
                'review_date': 'created_at',
                'date_created': 'created_at',
                
                # User variations
                'user': 'user_id',
                'customer_id': 'user_id',
                'reviewer_id': 'user_id',
                
                # Product variations
                'product': 'product_id',
                'item_id': 'product_id',
                'asin': 'product_id',
                
                # Rating variations
                'score': 'rating',
                'stars': 'rating',
                'review_rating': 'rating'
            }
            
            # Apply column mapping
            df.columns = df.columns.str.lower().str.strip()
            df = df.rename(columns=column_mapping)
            
            # Check for required columns
            required_columns = ['review_text', 'rating']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                st.info("Please ensure your dataset has columns for review text and rating.")
                return None
            
            # Validate rating column
            if not pd.api.types.is_numeric_dtype(df['rating']):
                try:
                    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
                except:
                    st.error("Rating column must contain numeric values.")
                    return None
            
            # Filter valid ratings (1-5)
            valid_ratings = df['rating'].between(1, 5, inclusive='both')
            if not valid_ratings.all():
                invalid_count = (~valid_ratings).sum()
                st.warning(f"Removed {invalid_count} rows with invalid ratings (not between 1-5).")
                df = df[valid_ratings]
            
            # Remove rows with empty review text
            df = df.dropna(subset=['review_text'])
            df = df[df['review_text'].str.strip() != '']
            
            # Add missing columns with default values
            if 'user_id' not in df.columns:
                df['user_id'] = [f"user_{i+1}" for i in range(len(df))]
            
            if 'product_id' not in df.columns:
                df['product_id'] = [f"product_{i+1}" for i in range(len(df))]
            
            if 'created_at' not in df.columns:
                # Generate random dates within the last year
                start_date = datetime.now() - timedelta(days=365)
                df['created_at'] = pd.to_datetime([
                    start_date + timedelta(days=random.randint(0, 365))
                    for _ in range(len(df))
                ])
            else:
                # Try to parse existing date column
                try:
                    df['created_at'] = pd.to_datetime(df['created_at'])
                except:
                    st.warning("Could not parse date column. Using current date.")
                    df['created_at'] = datetime.now()
            
            if 'title' not in df.columns:
                df['title'] = "Review Title"
            
            if 'verified_purchase' not in df.columns:
                df['verified_purchase'] = True
            
            # Add helpful votes columns if missing
            if 'helpful_votes' not in df.columns:
                df['helpful_votes'] = np.random.randint(0, 20, size=len(df))
            
            if 'total_votes' not in df.columns:
                df['total_votes'] = df['helpful_votes'] + np.random.randint(0, 10, size=len(df))
            
            # Reset index
            df = df.reset_index(drop=True)
            
            # Show data summary
            st.success(f"✅ Successfully loaded {len(df)} reviews from {st.session_state.get('uploaded_filename', 'uploaded file')}")
            
            return df
            
        except Exception as e:
            st.error(f"Error validating data: {str(e)}")
            return None

    def render_header(self):
        """Render the dashboard header."""
        st.markdown('<h1 class="main-header">📊 Product Review Analytics Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        **Welcome to the comprehensive Product Review Analysis Dashboard!**
        
        This dashboard demonstrates a complete data engineering pipeline including:
        - 🔄 **Data Generation & Ingestion**: Simulated e-commerce review data
        - 💾 **Multi-Storage Architecture**: SQLite, MongoDB, and Data Lake
        - 🔧 **Data Transformation**: Sentiment analysis, feature engineering, and quality checks
        - 📈 **Real-time Analytics**: Interactive visualizations and insights
        """)
        
        st.divider()
    
    def render_sidebar(self):
        """Render the sidebar with controls and filters."""
        st.sidebar.header("🎛️ Dashboard Controls")
        
        # Data source selection
        st.sidebar.subheader("📊 Data Source")
        data_source = st.sidebar.radio(
            "Choose Data Source:",
            ["Generated Data", "Upload Dataset"],
            help="Select whether to use generated sample data or upload your own dataset"
        )
        
        if data_source == "Generated Data":
            # Data generation controls
            st.sidebar.subheader("Data Generation")
            num_reviews = st.sidebar.slider("Number of Reviews", 100, 5000, 1000, 100)
            
            # Check if slider value changed and reload data accordingly
            if 'current_num_reviews' not in st.session_state:
                st.session_state['current_num_reviews'] = num_reviews
                self.load_existing_data(num_reviews)
            elif st.session_state['current_num_reviews'] != num_reviews:
                st.session_state['current_num_reviews'] = num_reviews
                self.load_existing_data(num_reviews)
                st.rerun()
            
            if st.sidebar.button("🔄 Generate New Data", type="primary"):
                self.generate_sample_data(num_reviews)
                st.rerun()
        
        else:  # Upload Dataset
            st.sidebar.subheader("📁 Upload Dataset")
            uploaded_file = st.sidebar.file_uploader(
                "Choose a file",
                type=['csv', 'xlsx', 'xls'],
                help="Upload a CSV or Excel file with review data"
            )
            
            if uploaded_file is not None:
                if st.sidebar.button("📤 Load Uploaded Data", type="primary"):
                    success = self.load_uploaded_data(uploaded_file)
                    if success:
                        st.sidebar.success("✅ Data loaded successfully!")
                        st.rerun()
                    else:
                        st.sidebar.error("❌ Failed to load data. Please check file format.")
            
            # Show data requirements
            with st.sidebar.expander("📋 Data Format Requirements"):
                st.write("""
                **Required columns:**
                - `review_text` or `text`: Review content
                - `rating`: Numeric rating (1-5)
                
                **Optional columns:**
                - `user_id`: User identifier
                - `product_id`: Product identifier
                - `created_at` or `date`: Review date
                - `title`: Review title
                - `verified_purchase`: Boolean
                """)
            
            num_reviews = len(st.session_state.get('sample_data', [])) if 'sample_data' in st.session_state else 0
        
        # Pipeline controls
        st.sidebar.subheader("Pipeline Operations")
        
        if st.sidebar.button("🚀 Run Full Pipeline"):
            self.run_full_pipeline()
        
        if st.sidebar.button("🔍 Check Pipeline Health"):
            self.check_pipeline_health()
        
        # Filters
        st.sidebar.subheader("📊 Analytics Filters")
        
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )
        
        rating_filter = st.sidebar.multiselect(
            "Rating Filter",
            options=[1, 2, 3, 4, 5],
            default=[1, 2, 3, 4, 5]
        )
        
        return {
            'num_reviews': num_reviews,
            'date_range': date_range,
            'rating_filter': rating_filter
        }
    
    def generate_sample_data(self, num_reviews):
        """Generate sample review data."""
        with st.spinner(f"Generating {num_reviews} sample reviews..."):
            try:
                # Set a different random seed each time to ensure truly different data
                import random
                import time
                random.seed(int(time.time()))
                
                generator = ReviewDataGenerator()
                
                # Generate users and products first (smaller datasets)
                users_df = generator.generate_users(num_users=min(1000, num_reviews // 5))
                products_df = generator.generate_products(num_products=min(500, num_reviews // 10))
                
                # Generate the specified number of reviews
                reviews_df = generator.generate_reviews(users_df, products_df, num_reviews=num_reviews)
                
                self.sample_data = reviews_df
                st.success(f"✅ Generated {len(self.sample_data)} sample reviews!")
                
                # Clear old data and store new data in session state
                st.session_state['sample_data'] = self.sample_data
                
                # Clear any existing transformed data so all tabs use the new generated data
                if 'transformed_data' in st.session_state:
                    del st.session_state['transformed_data']
                if 'transformation_result' in st.session_state:
                    del st.session_state['transformation_result']
                
            except Exception as e:
                st.error(f"❌ Failed to generate data: {str(e)}")
    
    def run_full_pipeline(self):
        """Run the complete data pipeline."""
        if 'sample_data' not in st.session_state:
            st.warning("⚠️ Please generate sample data first!")
            return
        
        with st.spinner("Running full data pipeline..."):
            try:
                # Initialize components
                if not self.monitor.initialize_components():
                    return
                
                data = st.session_state['sample_data']
                
                # Store data
                self.monitor.storage_manager.store_data(
                    data, 
                    dataset_name="reviews", 
                    storage_type="auto"
                )
                
                # Transform data
                result = self.monitor.transformation_orchestrator.transform_dataset(
                    data, 
                    dataset_name="reviews"
                )
                
                # Store transformed data in session state
                st.session_state['transformation_result'] = result
                st.session_state['transformed_data'] = result.transformed_data
                
                st.success("✅ Pipeline completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Pipeline failed: {str(e)}")
    
    def check_pipeline_health(self):
        """Check and display pipeline health status."""
        if not self.monitor.initialize_components():
            return
        
        status = self.monitor.get_pipeline_status()
        
        st.subheader("🏥 Pipeline Health Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            storage_status = status['storage']['status']
            st.metric(
                "Storage System",
                storage_status.title(),
                delta=None,
                delta_color="normal"
            )
            
        with col2:
            transformation_status = status['transformation']['status']
            st.metric(
                "Transformation",
                transformation_status.title(),
                delta=None,
                delta_color="normal"
            )
            
        with col3:
            ingestion_status = status['ingestion']['status']
            st.metric(
                "Ingestion",
                ingestion_status.title(),
                delta=None,
                delta_color="normal"
            )
    
    def render_overview_metrics(self):
        """Render overview metrics and KPIs."""
        st.header("📈 Overview Metrics")
        
        if 'sample_data' in st.session_state and st.session_state['sample_data'] is not None:
            data = st.session_state['sample_data']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Reviews",
                    f"{len(data):,}",
                    delta=f"+{len(data)}" if len(data) > 0 else None
                )
            
            with col2:
                avg_rating = data['rating'].mean()
                st.metric(
                    "Average Rating",
                    f"{avg_rating:.2f}",
                    delta=f"{avg_rating - 3:.2f}" if avg_rating != 3 else None
                )
            
            with col3:
                unique_products = data['product_id'].nunique()
                st.metric(
                    "Unique Products",
                    f"{unique_products:,}",
                    delta=f"+{unique_products}"
                )
            
            with col4:
                unique_users = data['user_id'].nunique()
                st.metric(
                    "Unique Users",
                    f"{unique_users:,}",
                    delta=f"+{unique_users}"
                )
        else:
            st.info("📝 Generate sample data to see overview metrics")
    
    def render_sentiment_analysis(self):
        """Render sentiment analysis visualizations."""
        st.header("😊 Sentiment Analysis")
        
        # Try to use transformed data first, then fall back to sample data
        data = None
        if 'transformed_data' in st.session_state and st.session_state['transformed_data'] is not None:
            data = st.session_state['transformed_data']
        elif 'sample_data' in st.session_state and st.session_state['sample_data'] is not None:
            data = st.session_state['sample_data']
        
        if data is None:
            st.warning("No data available for sentiment analysis.")
            return
        
        # Create sentiment labels if they don't exist
        if 'sentiment_score' in data.columns and 'sentiment_label' not in data.columns:
            data = data.copy()
            data['sentiment_label'] = data['sentiment_score'].apply(
                lambda x: 'Positive' if x > 0.1 else 'Negative' if x < -0.1 else 'Neutral'
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment distribution
            if 'sentiment_label' in data.columns:
                sentiment_counts = data['sentiment_label'].value_counts()
                
                fig = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Distribution",
                    color_discrete_sequence=self.config.color_palette
                )
                fig.update_layout(height=self.config.chart_height)
                st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Sentiment score distribution
            if 'sentiment_score' in data.columns:
                fig = px.histogram(
                    data,
                    x='sentiment_score',
                    title="Sentiment Score Distribution",
                    nbins=30,
                    color_discrete_sequence=self.config.color_palette
                )
                fig.update_layout(height=self.config.chart_height)
                st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment over time
        time_col = None
        for col in ['timestamp', 'review_date', 'created_at']:
            if col in data.columns:
                time_col = col
                break
                
        if time_col and 'sentiment_score' in data.columns:
            try:
                data_copy = data.copy()
                data_copy['date'] = pd.to_datetime(data_copy[time_col]).dt.date
                daily_sentiment = data_copy.groupby('date')['sentiment_score'].mean().reset_index()
                
                fig = px.line(
                    daily_sentiment,
                    x='date',
                    y='sentiment_score',
                    title="Average Sentiment Over Time",
                    color_discrete_sequence=self.config.color_palette
                )
                fig.update_layout(height=self.config.chart_height)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create time series chart: {str(e)}")
        
        # Sentiment by rating
        if 'rating' in data.columns and 'sentiment_score' in data.columns:
            rating_sentiment = data.groupby('rating')['sentiment_score'].mean().reset_index()
            
            fig = px.bar(
                rating_sentiment,
                x='rating',
                y='sentiment_score',
                title="Average Sentiment by Rating",
                color_discrete_sequence=self.config.color_palette
            )
            fig.update_layout(height=self.config.chart_height)
            st.plotly_chart(fig, use_container_width=True)
    
    def render_rating_analysis(self):
        """Render rating analysis visualizations."""
        st.header("⭐ Rating Analysis")
        
        if 'sample_data' in st.session_state and st.session_state['sample_data'] is not None:
            data = st.session_state['sample_data']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Rating distribution
                rating_counts = data['rating'].value_counts().sort_index()
                
                fig = px.bar(
                    x=rating_counts.index,
                    y=rating_counts.values,
                    title="Rating Distribution",
                    labels={'x': 'Rating', 'y': 'Count'},
                    color_discrete_sequence=self.config.color_palette
                )
                fig.update_layout(height=self.config.chart_height)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Average rating by product category
                if 'category' in data.columns:
                    category_ratings = data.groupby('category')['rating'].mean().sort_values(ascending=False)
                    
                    fig = px.bar(
                        x=category_ratings.values,
                        y=category_ratings.index,
                        orientation='h',
                        title="Average Rating by Category",
                        labels={'x': 'Average Rating', 'y': 'Category'},
                        color_discrete_sequence=self.config.color_palette
                    )
                    fig.update_layout(height=self.config.chart_height)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📝 Generate sample data to see rating analysis")
    
    def render_data_quality_report(self):
        """Render data quality analysis."""
        st.header("🔍 Data Quality Report")
        
        # Try to use transformed data first, then fall back to sample data
        data = None
        if 'transformation_result' in st.session_state and st.session_state['transformation_result'] is not None:
            result = st.session_state['transformation_result']
            if hasattr(result, 'quality_report') and result.quality_report:
                # Use existing quality report
                quality_report = result.quality_report
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Quality Scores")
                    
                    # Overall quality score
                    overall_score = quality_report.overall_score
                    st.metric("Overall Quality Score", f"{overall_score:.2f}%")
                    
                    # Individual scores
                    for check_type, score in quality_report.scores_by_type.items():
                        st.metric(check_type.replace('_', ' ').title(), f"{score:.2f}%")
                
                with col2:
                    st.subheader("Quality Issues")
                    
                    if quality_report.failed_checks:
                        for check in quality_report.failed_checks:
                            severity_color = {
                                'critical': '🔴',
                                'warning': '🟡',
                                'info': '🔵'
                            }.get(check.severity.value, '⚪')
                            
                            st.write(f"{severity_color} **{check.rule.name}**: {check.message}")
                    else:
                        st.success("✅ No quality issues found!")
                return
        
        # Fall back to basic quality analysis
        if 'sample_data' in st.session_state and st.session_state['sample_data'] is not None:
            data = st.session_state['sample_data']
        
        if data is None:
            st.warning("No data available for quality analysis.")
            return
        
        # Perform basic quality checks
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Data Quality Metrics")
            
            # Basic metrics
            total_rows = len(data)
            total_cols = len(data.columns)
            
            st.metric("Total Rows", f"{total_rows:,}")
            st.metric("Total Columns", f"{total_cols}")
            
            # Missing values
            missing_values = data.isnull().sum().sum()
            missing_percentage = (missing_values / (total_rows * total_cols)) * 100
            st.metric("Missing Values", f"{missing_values:,} ({missing_percentage:.2f}%)")
            
            # Duplicate rows
            duplicates = data.duplicated().sum()
            duplicate_percentage = (duplicates / total_rows) * 100
            st.metric("Duplicate Rows", f"{duplicates:,} ({duplicate_percentage:.2f}%)")
        
        with col2:
            st.subheader("Column Quality")
            
            # Missing values by column
            missing_by_col = data.isnull().sum()
            missing_by_col = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
            
            if len(missing_by_col) > 0:
                st.write("**Columns with Missing Values:**")
                for col, count in missing_by_col.head(10).items():
                    percentage = (count / total_rows) * 100
                    st.write(f"• {col}: {count:,} ({percentage:.1f}%)")
            else:
                st.success("✅ No missing values found!")
        
        # Data type analysis
        st.subheader("Data Types")
        dtype_counts = data.dtypes.value_counts()
        
        fig = px.pie(
            values=dtype_counts.values,
            names=dtype_counts.index.astype(str),
            title="Distribution of Data Types",
            color_discrete_sequence=self.config.color_palette
        )
        fig.update_layout(height=self.config.chart_height)
        st.plotly_chart(fig, use_container_width=True)
        
        # Outlier detection for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.subheader("Outlier Analysis")
            
            selected_col = st.selectbox("Select column for outlier analysis:", numeric_cols)
            
            if selected_col:
                Q1 = data[selected_col].quantile(0.25)
                Q3 = data[selected_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = data[(data[selected_col] < lower_bound) | (data[selected_col] > upper_bound)]
                outlier_percentage = (len(outliers) / total_rows) * 100
                
                st.metric(f"Outliers in {selected_col}", f"{len(outliers):,} ({outlier_percentage:.2f}%)")
                
                # Box plot
                fig = px.box(data, y=selected_col, title=f"Box Plot: {selected_col}")
                fig.update_layout(height=self.config.chart_height)
                st.plotly_chart(fig, use_container_width=True)
    
    def render_feature_engineering_results(self):
        """Render feature engineering analysis."""
        st.header("🔧 Feature Engineering Results")
        
        # Try to use transformed data first, then fall back to sample data
        data = None
        if 'transformed_data' in st.session_state and st.session_state['transformed_data'] is not None:
            data = st.session_state['transformed_data']
        elif 'sample_data' in st.session_state and st.session_state['sample_data'] is not None:
            data = st.session_state['sample_data']
        
        if data is None:
            st.warning("No data available for feature engineering analysis.")
            return
        
        # Create basic engineered features if they don't exist
        data_copy = data.copy()
        
        # Add some basic features
        if 'helpful_votes' in data_copy.columns and 'total_votes' in data_copy.columns:
            # Ensure numeric data types for calculation
            try:
                data_copy['helpful_votes'] = pd.to_numeric(data_copy['helpful_votes'], errors='coerce').fillna(0)
                data_copy['total_votes'] = pd.to_numeric(data_copy['total_votes'], errors='coerce').fillna(0)
                data_copy['helpfulness_ratio'] = data_copy['helpful_votes'] / (data_copy['total_votes'] + 1)
            except Exception as e:
                st.warning(f"Could not calculate helpfulness ratio: {str(e)}")
                # Add default columns if calculation fails
                data_copy['helpful_votes'] = 0
                data_copy['total_votes'] = 0
                data_copy['helpfulness_ratio'] = 0
        
        if 'word_count' in data_copy.columns:
            data_copy['review_length_category'] = pd.cut(
                data_copy['word_count'], 
                bins=[0, 10, 25, 50, float('inf')], 
                labels=['Short', 'Medium', 'Long', 'Very Long']
            )
        
        if 'sentiment_score' in data_copy.columns:
            data_copy['sentiment_magnitude'] = abs(data_copy['sentiment_score'])
        
        # Feature correlation heatmap
        numeric_columns = data_copy.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            correlation_matrix = data_copy[numeric_columns].corr()
            
            fig = px.imshow(
                correlation_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu',
                aspect='auto'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Display available features
        st.subheader("Available Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Numeric Features:**")
            numeric_features = data_copy.select_dtypes(include=[np.number]).columns.tolist()
            for feature in numeric_features:
                st.write(f"• {feature}")
        
        with col2:
            st.write("**Categorical Features:**")
            categorical_features = data_copy.select_dtypes(include=['object', 'category']).columns.tolist()
            for feature in categorical_features[:10]:  # Limit to first 10
                st.write(f"• {feature}")
        
        # Feature distributions
        if 'helpfulness_ratio' in data_copy.columns:
            st.subheader("Feature Distributions")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    data_copy,
                    x='helpfulness_ratio',
                    title="Helpfulness Ratio Distribution",
                    nbins=30,
                    color_discrete_sequence=self.config.color_palette
                )
                fig.update_layout(height=self.config.chart_height)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'sentiment_magnitude' in data_copy.columns:
                    fig = px.histogram(
                        data_copy,
                        x='sentiment_magnitude',
                        title="Sentiment Magnitude Distribution",
                        nbins=30,
                        color_discrete_sequence=self.config.color_palette
                    )
                    fig.update_layout(height=self.config.chart_height)
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_raw_data_explorer(self):
        """Render raw data exploration interface."""
        st.header("🔍 Data Explorer")
        
        if 'sample_data' in st.session_state and st.session_state['sample_data'] is not None:
            data = st.session_state['sample_data']
            
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(data.head(100), use_container_width=True)
            
            # Data info
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dataset Info")
                st.write(f"**Shape**: {data.shape}")
                st.write(f"**Memory Usage**: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
            with col2:
                st.subheader("Column Types")
                st.write(data.dtypes.to_frame('Type'))
        else:
            st.info("📝 Generate sample data to explore the dataset")
    
    def run(self):
        """Run the main dashboard application."""
        self.setup_page()
        self.render_header()
        
        # Sidebar controls
        filters = self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Overview", 
            "😊 Sentiment", 
            "⭐ Ratings", 
            "🔍 Quality", 
            "🔧 Features", 
            "📊 Data Explorer"
        ])
        
        with tab1:
            self.render_overview_metrics()
        
        with tab2:
            self.render_sentiment_analysis()
        
        with tab3:
            self.render_rating_analysis()
        
        with tab4:
            self.render_data_quality_report()
        
        with tab5:
            self.render_feature_engineering_results()
        
        with tab6:
            self.render_raw_data_explorer()

def main():
    """Main function to run the dashboard."""
    dashboard = ReviewAnalyticsDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()