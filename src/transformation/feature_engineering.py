"""
Feature Engineering Module
Extracts meaningful features from product review data
Implements text features, temporal features, user behavior features, and product features
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
from collections import Counter
import math

# Text processing
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    # Text features
    enable_text_features: bool = True
    enable_tfidf_features: bool = True
    enable_topic_modeling: bool = True
    max_tfidf_features: int = 1000
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    num_topics: int = 10
    
    # Temporal features
    enable_temporal_features: bool = True
    temporal_window_days: int = 30
    
    # User behavior features
    enable_user_features: bool = True
    min_user_reviews: int = 2
    
    # Product features
    enable_product_features: bool = True
    min_product_reviews: int = 5
    
    # Advanced features
    enable_interaction_features: bool = True
    enable_statistical_features: bool = True


class TextFeatureExtractor:
    """Extract features from review text"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        # Initialize vectorizers
        self.tfidf_vectorizer = None
        self.topic_model = None
        self.is_fitted = False
    
    def extract_basic_text_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Extract basic text features"""
        logger.info("Extracting basic text features...")
        
        result_df = df.copy()
        
        # Text length features
        result_df['text_length'] = df[text_column].str.len()
        result_df['word_count'] = df[text_column].str.split().str.len()
        result_df['sentence_count'] = df[text_column].apply(self._count_sentences)
        result_df['avg_word_length'] = df[text_column].apply(self._avg_word_length)
        
        # Punctuation features
        result_df['exclamation_count'] = df[text_column].str.count('!')
        result_df['question_count'] = df[text_column].str.count(r'\?')
        result_df['capital_ratio'] = df[text_column].apply(self._capital_ratio)
        
        # Special character features
        result_df['digit_count'] = df[text_column].str.count(r'\d')
        result_df['special_char_count'] = df[text_column].apply(self._special_char_count)
        
        # Readability features
        result_df['readability_score'] = df[text_column].apply(self._flesch_reading_ease)
        
        # Linguistic features
        result_df['unique_word_ratio'] = df[text_column].apply(self._unique_word_ratio)
        result_df['stopword_ratio'] = df[text_column].apply(self._stopword_ratio)
        
        logger.info(f"Extracted {len([col for col in result_df.columns if col not in df.columns])} basic text features")
        return result_df
    
    def extract_tfidf_features(self, df: pd.DataFrame, text_column: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Extract TF-IDF features"""
        if not self.config.enable_tfidf_features:
            return df, np.array([])
        
        logger.info("Extracting TF-IDF features...")
        
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.config.max_tfidf_features,
            ngram_range=self.config.tfidf_ngram_range,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Fit and transform
        texts = df[text_column].fillna("").tolist()
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # Get feature names
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Create DataFrame with TF-IDF features
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{name}' for name in feature_names],
            index=df.index
        )
        
        # Combine with original DataFrame
        result_df = pd.concat([df, tfidf_df], axis=1)
        
        logger.info(f"Extracted {len(feature_names)} TF-IDF features")
        return result_df, tfidf_matrix.toarray()
    
    def extract_topic_features(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Extract topic modeling features using LDA"""
        if not self.config.enable_topic_modeling:
            return df
        
        logger.info("Extracting topic modeling features...")
        
        try:
            # Prepare text data
            texts = df[text_column].fillna("").tolist()
            
            # Create count vectorizer for LDA
            count_vectorizer = CountVectorizer(
                max_features=1000,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )
            
            count_matrix = count_vectorizer.fit_transform(texts)
            
            # Fit LDA model
            self.topic_model = LatentDirichletAllocation(
                n_components=self.config.num_topics,
                random_state=42,
                max_iter=10
            )
            
            topic_distributions = self.topic_model.fit_transform(count_matrix)
            
            # Create topic feature columns
            result_df = df.copy()
            for i in range(self.config.num_topics):
                result_df[f'topic_{i}_weight'] = topic_distributions[:, i]
            
            # Add dominant topic
            result_df['dominant_topic'] = np.argmax(topic_distributions, axis=1)
            result_df['dominant_topic_weight'] = np.max(topic_distributions, axis=1)
            
            logger.info(f"Extracted {self.config.num_topics} topic features")
            return result_df
            
        except Exception as e:
            logger.warning(f"Topic modeling failed: {e}")
            return df
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text"""
        if not isinstance(text, str):
            return 0
        try:
            return len(sent_tokenize(text))
        except:
            return text.count('.') + text.count('!') + text.count('?')
    
    def _avg_word_length(self, text: str) -> float:
        """Calculate average word length"""
        if not isinstance(text, str):
            return 0.0
        words = text.split()
        if not words:
            return 0.0
        return sum(len(word) for word in words) / len(words)
    
    def _capital_ratio(self, text: str) -> float:
        """Calculate ratio of capital letters"""
        if not isinstance(text, str) or len(text) == 0:
            return 0.0
        return sum(1 for c in text if c.isupper()) / len(text)
    
    def _special_char_count(self, text: str) -> int:
        """Count special characters"""
        if not isinstance(text, str):
            return 0
        return len(re.findall(r'[^a-zA-Z0-9\s]', text))
    
    def _flesch_reading_ease(self, text: str) -> float:
        """Calculate Flesch Reading Ease score"""
        if not isinstance(text, str) or len(text) == 0:
            return 0.0
        
        try:
            sentences = self._count_sentences(text)
            words = len(text.split())
            syllables = sum(self._count_syllables(word) for word in text.split())
            
            if sentences == 0 or words == 0:
                return 0.0
            
            score = 206.835 - (1.015 * words / sentences) - (84.6 * syllables / words)
            return max(0, min(100, score))
        except:
            return 0.0
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (simplified)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def _unique_word_ratio(self, text: str) -> float:
        """Calculate ratio of unique words"""
        if not isinstance(text, str):
            return 0.0
        words = text.lower().split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)
    
    def _stopword_ratio(self, text: str) -> float:
        """Calculate ratio of stopwords"""
        if not isinstance(text, str):
            return 0.0
        words = text.lower().split()
        if not words:
            return 0.0
        stopword_count = sum(1 for word in words if word in self.stop_words)
        return stopword_count / len(words)


class TemporalFeatureExtractor:
    """Extract temporal features from review data"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def extract_temporal_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Extract temporal features"""
        if not self.config.enable_temporal_features:
            return df
        
        logger.info("Extracting temporal features...")
        
        result_df = df.copy()
        
        # Convert to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(result_df[date_column]):
            result_df[date_column] = pd.to_datetime(result_df[date_column])
        
        # Basic temporal features
        result_df['year'] = result_df[date_column].dt.year
        result_df['month'] = result_df[date_column].dt.month
        result_df['day'] = result_df[date_column].dt.day
        result_df['day_of_week'] = result_df[date_column].dt.dayofweek
        result_df['day_of_year'] = result_df[date_column].dt.dayofyear
        result_df['week_of_year'] = result_df[date_column].dt.isocalendar().week
        result_df['quarter'] = result_df[date_column].dt.quarter
        
        # Cyclical features (sine/cosine encoding)
        result_df['month_sin'] = np.sin(2 * np.pi * result_df['month'] / 12)
        result_df['month_cos'] = np.cos(2 * np.pi * result_df['month'] / 12)
        result_df['day_of_week_sin'] = np.sin(2 * np.pi * result_df['day_of_week'] / 7)
        result_df['day_of_week_cos'] = np.cos(2 * np.pi * result_df['day_of_week'] / 7)
        
        # Time since features (relative to latest date)
        latest_date = result_df[date_column].max()
        result_df['days_since_latest'] = (latest_date - result_df[date_column]).dt.days
        result_df['weeks_since_latest'] = result_df['days_since_latest'] / 7
        result_df['months_since_latest'] = result_df['days_since_latest'] / 30.44
        
        # Seasonal features
        result_df['is_weekend'] = result_df['day_of_week'].isin([5, 6]).astype(int)
        result_df['is_holiday_season'] = result_df['month'].isin([11, 12]).astype(int)
        result_df['is_summer'] = result_df['month'].isin([6, 7, 8]).astype(int)
        
        logger.info(f"Extracted {len([col for col in result_df.columns if col not in df.columns])} temporal features")
        return result_df


class UserFeatureExtractor:
    """Extract user behavior features"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def extract_user_features(self, df: pd.DataFrame, user_column: str, 
                             rating_column: str = None, date_column: str = None) -> pd.DataFrame:
        """Extract user behavior features"""
        if not self.config.enable_user_features:
            return df
        
        logger.info("Extracting user behavior features...")
        
        result_df = df.copy()
        
        # User review statistics
        user_stats = df.groupby(user_column).agg({
            user_column: 'count',  # review count
            rating_column: ['mean', 'std', 'min', 'max'] if rating_column else None
        }).round(3)
        
        if rating_column:
            user_stats.columns = ['user_review_count', 'user_avg_rating', 'user_rating_std', 
                                 'user_min_rating', 'user_max_rating']
            user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0)
        else:
            user_stats.columns = ['user_review_count']
        
        # User rating behavior
        if rating_column:
            user_stats['user_rating_range'] = user_stats['user_max_rating'] - user_stats['user_min_rating']
            user_stats['user_is_harsh_reviewer'] = (user_stats['user_avg_rating'] < 3).astype(int)
            user_stats['user_is_generous_reviewer'] = (user_stats['user_avg_rating'] > 4).astype(int)
        
        # Temporal user features
        if date_column:
            user_temporal = df.groupby(user_column)[date_column].agg(['min', 'max', 'count'])
            user_temporal.columns = ['user_first_review_date', 'user_last_review_date', 'user_review_count_temp']
            
            # Calculate user activity span
            user_temporal['user_activity_span_days'] = (
                user_temporal['user_last_review_date'] - user_temporal['user_first_review_date']
            ).dt.days
            
            # Review frequency
            user_temporal['user_review_frequency'] = (
                user_temporal['user_review_count_temp'] / 
                (user_temporal['user_activity_span_days'] + 1)
            ).fillna(0)
            
            # Merge temporal features
            user_stats = user_stats.join(user_temporal[['user_activity_span_days', 'user_review_frequency']])
        
        # User experience level
        user_stats['user_experience_level'] = pd.cut(
            user_stats['user_review_count'],
            bins=[0, 2, 5, 10, float('inf')],
            labels=['novice', 'casual', 'regular', 'expert']
        )
        
        # Merge user features back to main DataFrame
        result_df = result_df.merge(user_stats, left_on=user_column, right_index=True, how='left')
        
        logger.info(f"Extracted {len([col for col in result_df.columns if col not in df.columns])} user features")
        return result_df


class ProductFeatureExtractor:
    """Extract product-related features"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def extract_product_features(self, df: pd.DataFrame, product_column: str,
                                rating_column: str = None, date_column: str = None) -> pd.DataFrame:
        """Extract product features"""
        if not self.config.enable_product_features:
            return df
        
        logger.info("Extracting product features...")
        
        result_df = df.copy()
        
        # Product review statistics
        product_stats = df.groupby(product_column).agg({
            product_column: 'count',  # review count
            rating_column: ['mean', 'std', 'min', 'max'] if rating_column else None
        }).round(3)
        
        if rating_column:
            product_stats.columns = ['product_review_count', 'product_avg_rating', 'product_rating_std',
                                   'product_min_rating', 'product_max_rating']
            product_stats['product_rating_std'] = product_stats['product_rating_std'].fillna(0)
        else:
            product_stats.columns = ['product_review_count']
        
        # Product rating characteristics
        if rating_column:
            product_stats['product_rating_range'] = (
                product_stats['product_max_rating'] - product_stats['product_min_rating']
            )
            product_stats['product_is_controversial'] = (product_stats['product_rating_std'] > 1.5).astype(int)
            product_stats['product_is_highly_rated'] = (product_stats['product_avg_rating'] > 4).astype(int)
            product_stats['product_is_poorly_rated'] = (product_stats['product_avg_rating'] < 2.5).astype(int)
        
        # Product popularity
        product_stats['product_popularity_rank'] = product_stats['product_review_count'].rank(
            method='dense', ascending=False
        )
        
        # Product lifecycle features
        if date_column:
            product_temporal = df.groupby(product_column)[date_column].agg(['min', 'max'])
            product_temporal.columns = ['product_first_review_date', 'product_last_review_date']
            
            # Product age and activity
            latest_date = df[date_column].max()
            product_temporal['product_age_days'] = (
                latest_date - product_temporal['product_first_review_date']
            ).dt.days
            product_temporal['product_days_since_last_review'] = (
                latest_date - product_temporal['product_last_review_date']
            ).dt.days
            
            # Merge temporal features
            product_stats = product_stats.join(product_temporal[['product_age_days', 'product_days_since_last_review']])
        
        # Product category (based on review count)
        product_stats['product_category'] = pd.cut(
            product_stats['product_review_count'],
            bins=[0, 5, 20, 50, float('inf')],
            labels=['niche', 'moderate', 'popular', 'bestseller']
        )
        
        # Merge product features back to main DataFrame
        result_df = result_df.merge(product_stats, left_on=product_column, right_index=True, how='left')
        
        logger.info(f"Extracted {len([col for col in result_df.columns if col not in df.columns])} product features")
        return result_df


class InteractionFeatureExtractor:
    """Extract interaction and derived features"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def extract_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract interaction features between different variables"""
        if not self.config.enable_interaction_features:
            return df
        
        logger.info("Extracting interaction features...")
        
        result_df = df.copy()
        
        # Text-Rating interactions
        if 'word_count' in df.columns and 'rating' in df.columns:
            result_df['word_count_rating_ratio'] = result_df['word_count'] / (result_df['rating'] + 1)
            result_df['is_detailed_positive'] = (
                (result_df['word_count'] > result_df['word_count'].median()) & 
                (result_df['rating'] >= 4)
            ).astype(int)
            result_df['is_detailed_negative'] = (
                (result_df['word_count'] > result_df['word_count'].median()) & 
                (result_df['rating'] <= 2)
            ).astype(int)
        
        # User-Product interactions
        if 'user_avg_rating' in df.columns and 'product_avg_rating' in df.columns:
            result_df['user_product_rating_diff'] = (
                result_df['user_avg_rating'] - result_df['product_avg_rating']
            )
            result_df['user_harsher_than_average'] = (result_df['user_product_rating_diff'] < -0.5).astype(int)
            result_df['user_more_generous_than_average'] = (result_df['user_product_rating_diff'] > 0.5).astype(int)
        
        # Temporal interactions
        if 'days_since_latest' in df.columns and 'rating' in df.columns:
            result_df['recent_review_rating_interaction'] = (
                result_df['rating'] * np.exp(-result_df['days_since_latest'] / 30)
            )
        
        # Text complexity interactions
        if 'readability_score' in df.columns and 'word_count' in df.columns:
            result_df['complexity_score'] = result_df['word_count'] / (result_df['readability_score'] + 1)
        
        logger.info(f"Extracted {len([col for col in result_df.columns if col not in df.columns])} interaction features")
        return result_df


class StatisticalFeatureExtractor:
    """Extract statistical and aggregated features"""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features"""
        if not self.config.enable_statistical_features:
            return df
        
        logger.info("Extracting statistical features...")
        
        result_df = df.copy()
        
        # Numerical feature statistics
        numerical_cols = result_df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col.endswith('_id') or col in ['rating']:  # Skip ID columns and target
                continue
            
            # Z-score normalization
            mean_val = result_df[col].mean()
            std_val = result_df[col].std()
            if std_val > 0:
                result_df[f'{col}_zscore'] = (result_df[col] - mean_val) / std_val
            
            # Percentile ranks
            result_df[f'{col}_percentile'] = result_df[col].rank(pct=True)
            
            # Outlier detection
            q1 = result_df[col].quantile(0.25)
            q3 = result_df[col].quantile(0.75)
            iqr = q3 - q1
            result_df[f'{col}_is_outlier'] = (
                (result_df[col] < q1 - 1.5 * iqr) | 
                (result_df[col] > q3 + 1.5 * iqr)
            ).astype(int)
        
        # Rolling statistics (if temporal data is available)
        if 'created_at' in df.columns:
            result_df = result_df.sort_values('created_at')
            
            # Rolling averages for rating
            if 'rating' in df.columns:
                result_df['rating_rolling_mean_7d'] = result_df['rating'].rolling(
                    window='7D', on='created_at'
                ).mean()
                result_df['rating_rolling_std_7d'] = result_df['rating'].rolling(
                    window='7D', on='created_at'
                ).std()
        
        logger.info(f"Extracted statistical features")
        return result_df


class FeatureEngineer:
    """Main feature engineering class that orchestrates all feature extractors"""
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        
        # Initialize feature extractors
        self.text_extractor = TextFeatureExtractor(self.config)
        self.temporal_extractor = TemporalFeatureExtractor(self.config)
        self.user_extractor = UserFeatureExtractor(self.config)
        self.product_extractor = ProductFeatureExtractor(self.config)
        self.interaction_extractor = InteractionFeatureExtractor(self.config)
        self.statistical_extractor = StatisticalFeatureExtractor(self.config)
        
        logger.info("Feature Engineer initialized")
    
    def engineer_features(self, df: pd.DataFrame, 
                         text_column: str = 'review_text',
                         user_column: str = 'user_id',
                         product_column: str = 'product_id',
                         rating_column: str = 'rating',
                         date_column: str = 'created_at') -> pd.DataFrame:
        """Engineer all features for the dataset"""
        if df is None:
            logger.error("Input DataFrame is None")
            return pd.DataFrame()
        
        logger.info(f"Starting feature engineering for dataset with {len(df)} rows")
        
        result_df = df.copy()
        original_columns = set(result_df.columns)
        
        # Extract text features
        if self.config.enable_text_features and text_column in result_df.columns:
            result_df = self.text_extractor.extract_basic_text_features(result_df, text_column)
            
            if self.config.enable_tfidf_features:
                result_df, _ = self.text_extractor.extract_tfidf_features(result_df, text_column)
            
            if self.config.enable_topic_modeling:
                result_df = self.text_extractor.extract_topic_features(result_df, text_column)
        
        # Extract temporal features
        if date_column in result_df.columns:
            result_df = self.temporal_extractor.extract_temporal_features(result_df, date_column)
        
        # Extract user features
        if user_column in result_df.columns:
            result_df = self.user_extractor.extract_user_features(
                result_df, user_column, rating_column, date_column
            )
        
        # Extract product features
        if product_column in result_df.columns:
            result_df = self.product_extractor.extract_product_features(
                result_df, product_column, rating_column, date_column
            )
        
        # Extract interaction features
        result_df = self.interaction_extractor.extract_interaction_features(result_df)
        
        # Extract statistical features
        result_df = self.statistical_extractor.extract_statistical_features(result_df)
        
        # Summary
        new_columns = set(result_df.columns) - original_columns
        logger.info(f"Feature engineering completed. Added {len(new_columns)} new features")
        logger.info(f"Final dataset shape: {result_df.shape}")
        
        return result_df
    
    def get_feature_importance(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Calculate feature importance using correlation and mutual information"""
        logger.info("Calculating feature importance...")
        
        # Select numerical features only
        numerical_features = df.select_dtypes(include=[np.number]).columns
        numerical_features = [col for col in numerical_features if col != target_column]
        
        if not numerical_features:
            logger.warning("No numerical features found for importance calculation")
            return pd.DataFrame()
        
        feature_importance = []
        
        for feature in numerical_features:
            try:
                # Correlation with target
                correlation = abs(df[feature].corr(df[target_column]))
                
                # Basic statistics
                variance = df[feature].var()
                missing_ratio = df[feature].isnull().sum() / len(df)
                
                feature_importance.append({
                    'feature': feature,
                    'correlation': correlation,
                    'variance': variance,
                    'missing_ratio': missing_ratio,
                    'importance_score': correlation * (1 - missing_ratio) * min(1, variance)
                })
                
            except Exception as e:
                logger.warning(f"Could not calculate importance for {feature}: {e}")
        
        importance_df = pd.DataFrame(feature_importance)
        importance_df = importance_df.sort_values('importance_score', ascending=False)
        
        logger.info(f"Calculated importance for {len(importance_df)} features")
        return importance_df
    
    def select_top_features(self, df: pd.DataFrame, target_column: str, 
                           top_k: int = 50) -> List[str]:
        """Select top K most important features"""
        importance_df = self.get_feature_importance(df, target_column)
        
        if importance_df.empty:
            return []
        
        top_features = importance_df.head(top_k)['feature'].tolist()
        logger.info(f"Selected top {len(top_features)} features")
        
        return top_features


def main():
    """Demonstrate feature engineering functionality"""
    logger.info("🔧 Demonstrating Feature Engineering Module")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = []
    for i in range(n_samples):
        sample_data.append({
            'review_id': i + 1,
            'user_id': np.random.randint(1, 101),
            'product_id': np.random.randint(1, 51),
            'rating': np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3]),
            'review_text': f"This product is {'amazing' if np.random.random() > 0.5 else 'terrible'}! " +
                          f"{'I love it so much. ' * np.random.randint(1, 5)}" +
                          f"{'Quality is great. ' * np.random.randint(0, 3)}" +
                          f"{'Would recommend! ' if np.random.random() > 0.3 else 'Not worth it. '}",
            'created_at': datetime.now() - timedelta(days=np.random.randint(0, 365)),
            'helpful_votes': np.random.randint(0, 20)
        })
    
    df = pd.DataFrame(sample_data)
    
    print(f"\n📊 Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize feature engineer
    config = FeatureConfig(
        enable_text_features=True,
        enable_tfidf_features=False,  # Skip TF-IDF for demo (too many features)
        enable_topic_modeling=False,  # Skip topic modeling for demo
        enable_temporal_features=True,
        enable_user_features=True,
        enable_product_features=True,
        enable_interaction_features=True,
        enable_statistical_features=True
    )
    
    engineer = FeatureEngineer(config)
    
    # Engineer features
    print("\n🔧 Engineering features...")
    df_engineered = engineer.engineer_features(df)
    
    print(f"\n📈 Engineered dataset shape: {df_engineered.shape}")
    print(f"Added {df_engineered.shape[1] - df.shape[1]} new features")
    
    # Show sample of new features
    new_columns = [col for col in df_engineered.columns if col not in df.columns]
    print(f"\n🆕 Sample of new features:")
    for i, col in enumerate(new_columns[:10]):
        print(f"  • {col}")
    if len(new_columns) > 10:
        print(f"  ... and {len(new_columns) - 10} more")
    
    # Feature importance analysis
    print("\n📊 Feature importance analysis:")
    importance_df = engineer.get_feature_importance(df_engineered, 'rating')
    
    if not importance_df.empty:
        print("Top 10 most important features:")
        for i, row in importance_df.head(10).iterrows():
            print(f"  • {row['feature']}: {row['importance_score']:.3f}")
        
        # Select top features
        top_features = engineer.select_top_features(df_engineered, 'rating', top_k=20)
        print(f"\n🎯 Selected {len(top_features)} top features for modeling")
    
    # Show feature categories
    print("\n📂 Feature categories:")
    feature_categories = {
        'Text Features': [col for col in new_columns if any(x in col for x in ['text_', 'word_', 'sentence_', 'readability'])],
        'Temporal Features': [col for col in new_columns if any(x in col for x in ['year', 'month', 'day', 'since', 'temporal'])],
        'User Features': [col for col in new_columns if 'user_' in col],
        'Product Features': [col for col in new_columns if 'product_' in col],
        'Interaction Features': [col for col in new_columns if any(x in col for x in ['_diff', '_ratio', '_interaction'])],
        'Statistical Features': [col for col in new_columns if any(x in col for x in ['_zscore', '_percentile', '_outlier'])]
    }
    
    for category, features in feature_categories.items():
        if features:
            print(f"  • {category}: {len(features)} features")
    
    # Show sample statistics
    print("\n📈 Sample feature statistics:")
    sample_features = ['text_length', 'word_count', 'user_avg_rating', 'product_avg_rating']
    available_features = [f for f in sample_features if f in df_engineered.columns]
    
    if available_features:
        stats = df_engineered[available_features].describe()
        print(stats.round(2))
    
    print("\n✅ Feature engineering demonstration completed!")


if __name__ == "__main__":
    main()