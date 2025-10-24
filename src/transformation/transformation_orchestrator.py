"""
Transformation Orchestrator
Coordinates all data transformation processes including sentiment analysis,
feature engineering, and data quality checks in a unified pipeline
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

# Import transformation modules
from .sentiment_analyzer import SentimentAnalyzer, SentimentConfig
from .feature_engineering import FeatureEngineer, FeatureConfig
from .data_quality import DataQualityChecker, DataCleaner, DataQualityReport

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TransformationConfig:
    """Configuration for the transformation pipeline"""
    # Data quality settings
    enable_quality_checks: bool = True
    enable_data_cleaning: bool = True
    quality_threshold: float = 0.8
    
    # Sentiment analysis settings
    enable_sentiment_analysis: bool = True
    sentiment_config: SentimentConfig = field(default_factory=SentimentConfig)
    
    # Feature engineering settings
    enable_feature_engineering: bool = True
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    
    # Output settings
    save_intermediate_results: bool = True
    output_directory: str = "data/transformed"
    
    # Processing settings
    batch_size: int = 1000
    parallel_processing: bool = True
    
    # Column mappings
    text_column: str = "review_text"
    user_column: str = "user_id"
    product_column: str = "product_id"
    rating_column: str = "rating"
    date_column: str = "created_at"


@dataclass
class TransformationResult:
    """Result of the transformation pipeline"""
    original_shape: Tuple[int, int]
    final_shape: Tuple[int, int]
    transformed_data: Optional[pd.DataFrame] = None
    quality_report: Optional[DataQualityReport] = None
    sentiment_stats: Dict[str, Any] = field(default_factory=dict)
    feature_stats: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    output_files: List[str] = field(default_factory=list)


class TransformationOrchestrator:
    """Main orchestrator for all data transformation processes"""
    
    def __init__(self, config: TransformationConfig = None):
        self.config = config or TransformationConfig()
        
        # Initialize components
        self.quality_checker = DataQualityChecker()
        self.data_cleaner = DataCleaner()
        self.sentiment_analyzer = SentimentAnalyzer(self.config.sentiment_config)
        self.feature_engineer = FeatureEngineer(self.config.feature_config)
        
        # Create output directory
        os.makedirs(self.config.output_directory, exist_ok=True)
        
        logger.info("Transformation Orchestrator initialized")
    
    def transform_dataset(self, df: pd.DataFrame, dataset_name: str = "dataset") -> TransformationResult:
        """Execute the complete transformation pipeline"""
        logger.info(f"🚀 Starting transformation pipeline for {dataset_name}")
        start_time = datetime.now()
        
        try:
            result = TransformationResult(
                original_shape=df.shape,
                final_shape=df.shape
            )
            
            transformed_df = df.copy()
            
            # Step 1: Initial Data Quality Assessment
            if self.config.enable_quality_checks:
                logger.info("📊 Step 1: Initial data quality assessment")
                initial_quality_report = self.quality_checker.check_data_quality(
                    transformed_df, f"{dataset_name}_initial"
                )
                
                if initial_quality_report.overall_score < self.config.quality_threshold:
                    logger.warning(f"Initial data quality score ({initial_quality_report.overall_score:.2f}) "
                                 f"below threshold ({self.config.quality_threshold})")
                
                if self.config.save_intermediate_results:
                    self._save_quality_report(initial_quality_report, f"{dataset_name}_initial_quality")
            
            # Step 2: Data Cleaning
            if self.config.enable_data_cleaning:
                logger.info("🧹 Step 2: Data cleaning")
                transformed_df = self._clean_data(transformed_df, dataset_name)
                
                if self.config.save_intermediate_results:
                    self._save_dataframe(transformed_df, f"{dataset_name}_cleaned")
            
            # Step 3: Sentiment Analysis
            if self.config.enable_sentiment_analysis:
                logger.info("😊 Step 3: Sentiment analysis")
                transformed_df, sentiment_stats = self._analyze_sentiment(transformed_df, dataset_name)
                result.sentiment_stats = sentiment_stats
                
                if self.config.save_intermediate_results:
                    self._save_dataframe(transformed_df, f"{dataset_name}_with_sentiment")
            
            # Step 4: Feature Engineering
            if self.config.enable_feature_engineering:
                logger.info("🔧 Step 4: Feature engineering")
                transformed_df, feature_stats = self._engineer_features(transformed_df, dataset_name)
                result.feature_stats = feature_stats
                
                if self.config.save_intermediate_results:
                    self._save_dataframe(transformed_df, f"{dataset_name}_with_features")
            
            # Step 5: Final Data Quality Check
            if self.config.enable_quality_checks:
                logger.info("✅ Step 5: Final data quality assessment")
                final_quality_report = self.quality_checker.check_data_quality(
                    transformed_df, f"{dataset_name}_final"
                )
                result.quality_report = final_quality_report
                
                if self.config.save_intermediate_results:
                    self._save_quality_report(final_quality_report, f"{dataset_name}_final_quality")
            
            # Step 6: Save Final Results
            logger.info("💾 Step 6: Saving final results")
            final_output_path = self._save_dataframe(transformed_df, f"{dataset_name}_transformed")
            result.output_files.append(final_output_path)
            
            # Update result
            result.final_shape = transformed_df.shape
            result.transformed_data = transformed_df
            result.processing_time = (datetime.now() - start_time).total_seconds()
            result.success = True
            
            logger.info(f"✅ Transformation pipeline completed successfully in {result.processing_time:.2f} seconds")
            logger.info(f"📊 Dataset transformed from {result.original_shape} to {result.final_shape}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Transformation pipeline failed: {e}")
            result = TransformationResult(
                original_shape=df.shape,
                final_shape=df.shape,
                processing_time=(datetime.now() - start_time).total_seconds(),
                success=False,
                error_message=str(e)
            )
            return result
    
    def _clean_data(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Clean the dataset"""
        cleaning_config = {
            'remove_duplicates': True,
            'handle_missing': True,
            'clean_text': True,
            'standardize_formats': True,
            'missing_strategy': {
                self.config.rating_column: 'median',
                self.config.text_column: 'drop',
                'user_email': 'drop'
            }
        }
        
        cleaned_df = self.data_cleaner.clean_dataset(df, cleaning_config)
        
        logger.info(f"Data cleaning: {len(df)} → {len(cleaned_df)} rows "
                   f"({len(df) - len(cleaned_df)} removed)")
        
        return cleaned_df
    
    def _analyze_sentiment(self, df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform sentiment analysis"""
        if self.config.text_column not in df.columns:
            logger.warning(f"Text column '{self.config.text_column}' not found. Skipping sentiment analysis.")
            return df, {}
        
        # Determine batch size and parallel flags
        batch_size = max(500, int(self.config.batch_size)) if hasattr(self.config, 'batch_size') else 1000
        parallel = getattr(self.config, 'parallel_processing', True)
        
        # Determine optimal number of workers from CPU cores
        import os
        max_workers = os.cpu_count() or 8
        # Cap to a reasonable upper bound
        if max_workers > 16:
            max_workers = 16
        
        df_with_sentiment = df
        try:
            df_with_sentiment = self.sentiment_analyzer.analyze_dataframe_chunked(
                df,
                text_column=self.config.text_column,
                batch_size=batch_size,
                parallel=parallel,
                progress_cb=None,
                max_workers=max_workers
            )
        except Exception as e:
            logger.warning(f"Chunked sentiment analysis failed ({e}); falling back to standard method.")
            df_with_sentiment = self.sentiment_analyzer.analyze_dataframe(
                df,
                text_column=self.config.text_column
            )
        
        # Calculate sentiment statistics
        sentiment_stats = {}
        if 'sentiment_label' in df_with_sentiment.columns:
            sentiment_distribution = df_with_sentiment['sentiment_label'].value_counts()
            sentiment_stats['distribution'] = sentiment_distribution.to_dict()
            sentiment_stats['total_analyzed'] = len(df_with_sentiment)
            
            if 'sentiment_score' in df_with_sentiment.columns:
                sentiment_stats['average_score'] = float(df_with_sentiment['sentiment_score'].mean())
        
        return df_with_sentiment, sentiment_stats
    
    def _engineer_features(self, df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Perform feature engineering"""
        original_columns = set(df.columns)
        
        # Engineer features
        df_with_features = self.feature_engineer.engineer_features(
            df,
            text_column=self.config.text_column,
            user_column=self.config.user_column,
            product_column=self.config.product_column,
            rating_column=self.config.rating_column,
            date_column=self.config.date_column
        )
        
        # Calculate feature statistics
        new_columns = set(df_with_features.columns) - original_columns
        feature_stats = {
            'original_features': len(original_columns),
            'new_features': len(new_columns),
            'total_features': len(df_with_features.columns),
            'feature_categories': self._categorize_features(list(new_columns))
        }
        
        # Feature importance analysis
        if self.config.rating_column in df_with_features.columns:
            try:
                importance_df = self.feature_engineer.get_feature_importance(
                    df_with_features, self.config.rating_column
                )
                if not importance_df.empty:
                    feature_stats['top_features'] = importance_df.head(10)['feature'].tolist()
                    feature_stats['avg_importance'] = importance_df['importance_score'].mean()
            except Exception as e:
                logger.warning(f"Could not calculate feature importance: {e}")
        
        logger.info(f"Feature engineering: {len(original_columns)} → {len(df_with_features.columns)} features "
                   f"({len(new_columns)} added)")
        
        return df_with_features, feature_stats
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, int]:
        """Categorize features by type"""
        categories = {
            'text_features': 0,
            'temporal_features': 0,
            'user_features': 0,
            'product_features': 0,
            'interaction_features': 0,
            'statistical_features': 0,
            'other_features': 0
        }
        
        for feature in feature_names:
            feature_lower = feature.lower()
            
            if any(keyword in feature_lower for keyword in ['text_', 'word_', 'sentence_', 'readability', 'tfidf_', 'topic_']):
                categories['text_features'] += 1
            elif any(keyword in feature_lower for keyword in ['year', 'month', 'day', 'since', 'temporal', '_sin', '_cos']):
                categories['temporal_features'] += 1
            elif 'user_' in feature_lower:
                categories['user_features'] += 1
            elif 'product_' in feature_lower:
                categories['product_features'] += 1
            elif any(keyword in feature_lower for keyword in ['_diff', '_ratio', '_interaction']):
                categories['interaction_features'] += 1
            elif any(keyword in feature_lower for keyword in ['_zscore', '_percentile', '_outlier', 'rolling_']):
                categories['statistical_features'] += 1
            else:
                categories['other_features'] += 1
        
        return categories
    
    def _save_dataframe(self, df: pd.DataFrame, filename: str) -> str:
        """Save DataFrame to file"""
        output_path = os.path.join(self.config.output_directory, f"{filename}.parquet")
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved DataFrame to {output_path}")
        return output_path
    
    def _save_quality_report(self, report: DataQualityReport, filename: str) -> str:
        """Save quality report to JSON file"""
        output_path = os.path.join(self.config.output_directory, f"{filename}.json")
        
        # Convert report to dictionary
        report_dict = {
            'dataset_name': report.dataset_name,
            'total_rows': report.total_rows,
            'total_columns': report.total_columns,
            'overall_score': report.overall_score,
            'critical_issues': report.critical_issues,
            'high_issues': report.high_issues,
            'medium_issues': report.medium_issues,
            'low_issues': report.low_issues,
            'timestamp': report.timestamp.isoformat(),
            'check_results': [
                {
                    'rule_name': result.rule_name,
                    'check_type': result.check_type.value,
                    'severity': result.severity.value,
                    'passed': result.passed,
                    'pass_rate': result.pass_rate,
                    'failed_count': result.failed_count,
                    'total_count': result.total_count,
                    'details': result.details
                }
                for result in report.check_results
            ]
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        logger.info(f"Saved quality report to {output_path}")
        return output_path
    
    def get_transformation_summary(self, result: TransformationResult) -> Dict[str, Any]:
        """Generate a comprehensive transformation summary"""
        summary = {
            'transformation_overview': {
                'success': result.success,
                'processing_time_seconds': result.processing_time,
                'original_shape': result.original_shape,
                'final_shape': result.final_shape,
                'rows_change': result.final_shape[0] - result.original_shape[0],
                'columns_added': result.final_shape[1] - result.original_shape[1]
            },
            'data_quality': {},
            'sentiment_analysis': result.sentiment_stats,
            'feature_engineering': result.feature_stats,
            'output_files': result.output_files
        }
        
        if result.quality_report:
            summary['data_quality'] = {
                'overall_score': result.quality_report.overall_score,
                'critical_issues': result.quality_report.critical_issues,
                'high_issues': result.quality_report.high_issues,
                'medium_issues': result.quality_report.medium_issues,
                'low_issues': result.quality_report.low_issues,
                'total_checks': len(result.quality_report.check_results),
                'passed_checks': sum(1 for r in result.quality_report.check_results if r.passed)
            }
        
        if not result.success:
            summary['error'] = result.error_message
        
        return summary
    
    def batch_transform_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, TransformationResult]:
        """Transform multiple datasets in batch"""
        logger.info(f"🔄 Starting batch transformation for {len(datasets)} datasets")
        
        results = {}
        
        for dataset_name, df in datasets.items():
            logger.info(f"Processing dataset: {dataset_name}")
            try:
                result = self.transform_dataset(df, dataset_name)
                results[dataset_name] = result
                
                if result.success:
                    logger.info(f"✅ {dataset_name} transformed successfully")
                else:
                    logger.error(f"❌ {dataset_name} transformation failed: {result.error_message}")
                    
            except Exception as e:
                logger.error(f"❌ Error processing {dataset_name}: {e}")
                results[dataset_name] = TransformationResult(
                    original_shape=df.shape,
                    final_shape=df.shape,
                    success=False,
                    error_message=str(e)
                )
        
        # Generate batch summary
        successful_transforms = sum(1 for r in results.values() if r.success)
        logger.info(f"🎯 Batch transformation completed: {successful_transforms}/{len(datasets)} successful")
        
        return results


def main():
    """Demonstrate transformation orchestrator functionality"""
    logger.info("🎭 Demonstrating Transformation Orchestrator")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    
    sample_data = []
    for i in range(n_samples):
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
        sentiment_words = {
            1: ["terrible", "awful", "hate", "worst"],
            2: ["bad", "poor", "disappointing", "not good"],
            3: ["okay", "average", "decent", "fine"],
            4: ["good", "nice", "great", "like"],
            5: ["amazing", "excellent", "love", "perfect"]
        }
        
        words = sentiment_words[rating]
        review_text = f"This product is {np.random.choice(words)}! " + \
                     f"{'I really recommend it. ' if rating >= 4 else 'Not worth the money. '}" + \
                     f"{'Quality is great. ' * np.random.randint(0, 3)}"
        
        sample_data.append({
            'review_id': i + 1,
            'user_id': np.random.randint(1, 101),
            'product_id': np.random.randint(1, 51),
            'rating': rating,
            'review_text': review_text,
            'created_at': datetime.now() - timedelta(days=np.random.randint(0, 365)),
            'helpful_votes': np.random.randint(0, 20)
        })
    
    df = pd.DataFrame(sample_data)
    
    print(f"\n📊 Sample dataset created:")
    print(f"  • Shape: {df.shape}")
    print(f"  • Columns: {list(df.columns)}")
    print(f"  • Rating distribution:")
    rating_dist = df['rating'].value_counts().sort_index()
    for rating, count in rating_dist.items():
        print(f"    - {rating} stars: {count} ({count/len(df)*100:.1f}%)")
    
    # Configure transformation pipeline
    config = TransformationConfig(
        enable_quality_checks=True,
        enable_data_cleaning=True,
        enable_sentiment_analysis=True,
        enable_feature_engineering=True,
        save_intermediate_results=True,
        output_directory="data/demo_transformed",
        sentiment_config=SentimentConfig(
            enable_vader=True,
            enable_ml_model=False,  # Skip ML for demo
            enable_emotion_detection=True
        ),
        feature_config=FeatureConfig(
            enable_text_features=True,
            enable_tfidf_features=False,  # Skip TF-IDF for demo
            enable_topic_modeling=False,  # Skip topic modeling for demo
            enable_temporal_features=True,
            enable_user_features=True,
            enable_product_features=True,
            enable_interaction_features=True,
            enable_statistical_features=True
        )
    )
    
    # Initialize orchestrator
    orchestrator = TransformationOrchestrator(config)
    
    # Transform dataset
    print(f"\n🚀 Starting transformation pipeline...")
    result = orchestrator.transform_dataset(df, "demo_reviews")
    
    # Display results
    print(f"\n📊 Transformation Results:")
    print(f"  • Success: {'✅ Yes' if result.success else '❌ No'}")
    print(f"  • Processing time: {result.processing_time:.2f} seconds")
    print(f"  • Shape change: {result.original_shape} → {result.final_shape}")
    print(f"  • Columns added: {result.final_shape[1] - result.original_shape[1]}")
    
    if result.quality_report:
        print(f"\n📋 Data Quality:")
        print(f"  • Overall score: {result.quality_report.overall_score:.2f}")
        print(f"  • Critical issues: {result.quality_report.critical_issues}")
        print(f"  • High issues: {result.quality_report.high_issues}")
        print(f"  • Medium issues: {result.quality_report.medium_issues}")
    
    if result.sentiment_stats:
        print(f"\n😊 Sentiment Analysis:")
        if 'distribution' in result.sentiment_stats:
            for label, count in result.sentiment_stats['distribution'].items():
                pct = count / result.sentiment_stats['total_analyzed'] * 100
                print(f"  • {label}: {count} ({pct:.1f}%)")
        
        if 'avg_score' in result.sentiment_stats:
            print(f"  • Average sentiment score: {result.sentiment_stats['avg_score']:.3f}")
    
    if result.feature_stats:
        print(f"\n🔧 Feature Engineering:")
        print(f"  • Original features: {result.feature_stats['original_features']}")
        print(f"  • New features: {result.feature_stats['new_features']}")
        print(f"  • Total features: {result.feature_stats['total_features']}")
        
        if 'feature_categories' in result.feature_stats:
            print(f"  • Feature categories:")
            for category, count in result.feature_stats['feature_categories'].items():
                if count > 0:
                    print(f"    - {category.replace('_', ' ').title()}: {count}")
        
        if 'top_features' in result.feature_stats:
            print(f"  • Top important features:")
            for i, feature in enumerate(result.feature_stats['top_features'][:5], 1):
                print(f"    {i}. {feature}")
    
    print(f"\n💾 Output Files:")
    for file_path in result.output_files:
        print(f"  • {file_path}")
    
    # Generate comprehensive summary
    summary = orchestrator.get_transformation_summary(result)
    
    print(f"\n📈 Transformation Summary:")
    print(f"  • Data processing efficiency: {result.final_shape[0]/result.original_shape[0]*100:.1f}% rows retained")
    print(f"  • Feature expansion: {(result.final_shape[1]/result.original_shape[1]-1)*100:.1f}% more features")
    print(f"  • Processing rate: {result.original_shape[0]/result.processing_time:.0f} rows/second")
    
    print("\n✅ Transformation orchestrator demonstration completed!")


if __name__ == "__main__":
    main()