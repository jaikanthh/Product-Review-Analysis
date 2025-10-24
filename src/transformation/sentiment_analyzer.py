"""
Sentiment Analysis Module
Implements sentiment analysis for product reviews using multiple approaches
Provides sentiment scoring, emotion detection, and text preprocessing
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Text processing libraries
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag

# Machine learning libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Download required NLTK data with SSL context fix
try:
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
except Exception as e:
    logger.warning(f"NLTK download failed: {e}. Some features may not work properly.")
    pass

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Result of sentiment analysis"""
    text: str
    sentiment_label: str  # positive, negative, neutral
    sentiment_score: float  # -1 to 1
    confidence: float  # 0 to 1
    emotions: Dict[str, float]  # emotion scores
    keywords: List[str]  # important keywords
    processed_text: str  # cleaned text
    word_count: int
    sentence_count: int


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis"""
    model_type: str = "vader"  # vader, ml, ensemble
    enable_preprocessing: bool = True
    enable_emotion_detection: bool = True
    enable_keyword_extraction: bool = True
    min_text_length: int = 5
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    model_save_path: str = "models/sentiment_model.pkl"


class TextPreprocessor:
    """Text preprocessing utilities for sentiment analysis"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        # Emotion keywords
        self.emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'pleased', 'delighted', 'thrilled', 'amazing', 'wonderful', 'fantastic', 'excellent'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated', 'terrible', 'awful', 'horrible', 'worst'],
            'sadness': ['sad', 'disappointed', 'upset', 'depressed', 'unhappy', 'miserable', 'poor', 'bad', 'unsatisfied'],
            'fear': ['scared', 'afraid', 'worried', 'anxious', 'concerned', 'nervous', 'uncertain', 'doubtful'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected', 'incredible', 'unbelievable'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'gross', 'nasty', 'disgusting', 'yuck']
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\!\?\,\;\:]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """Tokenize and lemmatize text"""
        try:
            tokens = word_tokenize(text)
            # Remove stopwords and lemmatize
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                     if token.lower() not in self.stop_words and len(token) > 2]
            return tokens
        except:
            return text.split()
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract important keywords from text"""
        try:
            # Simple keyword extraction based on word frequency and length
            tokens = self.tokenize_and_lemmatize(text)
            
            # Filter tokens by length and frequency
            word_freq = {}
            for token in tokens:
                if len(token) > 3:  # Only consider words longer than 3 characters
                    word_freq[token] = word_freq.get(token, 0) + 1
            
            # Sort by frequency and return top keywords
            keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, freq in keywords[:top_k]]
            
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []
    
    def detect_emotions(self, text: str) -> Dict[str, float]:
        """Detect emotions in text based on keyword matching"""
        emotions = {emotion: 0.0 for emotion in self.emotion_keywords.keys()}
        
        text_lower = text.lower()
        total_emotion_words = 0
        
        for emotion, keywords in self.emotion_keywords.items():
            emotion_count = 0
            for keyword in keywords:
                emotion_count += text_lower.count(keyword)
            
            emotions[emotion] = emotion_count
            total_emotion_words += emotion_count
        
        # Normalize emotions
        if total_emotion_words > 0:
            for emotion in emotions:
                emotions[emotion] = emotions[emotion] / total_emotion_words
        
        return emotions


class VaderSentimentAnalyzer:
    """VADER sentiment analysis implementation"""
    
    def __init__(self):
        try:
            self.analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            logger.error(f"Failed to initialize VADER analyzer: {e}")
            self.analyzer = None
    
    def analyze(self, text: str) -> Tuple[str, float, float]:
        """Analyze sentiment using VADER"""
        if not self.analyzer or not text:
            return "neutral", 0.0, 0.0
        
        try:
            scores = self.analyzer.polarity_scores(text)
            
            # Determine sentiment label
            compound_score = scores['compound']
            if compound_score >= 0.05:
                sentiment_label = "positive"
            elif compound_score <= -0.05:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            # Calculate confidence based on the absolute compound score
            confidence = abs(compound_score)
            
            return sentiment_label, compound_score, confidence
            
        except Exception as e:
            logger.warning(f"VADER analysis failed: {e}")
            return "neutral", 0.0, 0.0


class MLSentimentAnalyzer:
    """Machine Learning-based sentiment analyzer"""
    
    def __init__(self, config: SentimentConfig):
        self.config = config
        self.model = None
        self.vectorizer = None
        self.is_trained = False
        self.model_path = Path(config.model_save_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
    
    def train(self, texts: List[str], labels: List[str]) -> Dict[str, float]:
        """Train the ML sentiment model"""
        logger.info("Training ML sentiment model...")
        
        try:
            # Create pipeline with TF-IDF and classifier
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                stop_words='english'
            )
            
            # Use Logistic Regression as the classifier
            classifier = LogisticRegression(random_state=42, max_iter=1000)
            
            self.model = Pipeline([
                ('tfidf', self.vectorizer),
                ('classifier', classifier)
            ])
            
            # Split data for training and validation
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate the model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get detailed classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            self.is_trained = True
            
            # Save the model
            self.save_model()
            
            logger.info(f"Model trained successfully with accuracy: {accuracy:.3f}")
            
            return {
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {}
    
    def analyze(self, text: str) -> Tuple[str, float, float]:
        """Analyze sentiment using trained ML model"""
        if not self.is_trained or not self.model:
            return "neutral", 0.0, 0.0
        
        try:
            # Get prediction and probability
            prediction = self.model.predict([text])[0]
            probabilities = self.model.predict_proba([text])[0]
            
            # Get confidence (max probability)
            confidence = max(probabilities)
            
            # Convert to sentiment score (-1 to 1)
            if prediction == "positive":
                sentiment_score = confidence
            elif prediction == "negative":
                sentiment_score = -confidence
            else:
                sentiment_score = 0.0
            
            return prediction, sentiment_score, confidence
            
        except Exception as e:
            logger.warning(f"ML analysis failed: {e}")
            return "neutral", 0.0, 0.0
    
    def save_model(self):
        """Save the trained model"""
        if self.model:
            try:
                with open(self.model_path, 'wb') as f:
                    pickle.dump(self.model, f)
                logger.info(f"Model saved to {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
    
    def load_model(self) -> bool:
        """Load a pre-trained model"""
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                logger.info(f"Model loaded from {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        return False


class SentimentAnalyzer:
    """Main sentiment analyzer with multiple analysis methods"""
    
    def __init__(self, config: SentimentConfig = None):
        self.config = config or SentimentConfig()
        self.preprocessor = TextPreprocessor()
        
        # Initialize analyzers
        self.vader_analyzer = VaderSentimentAnalyzer()
        self.ml_analyzer = MLSentimentAnalyzer(self.config)
        
        # Try to load pre-trained ML model
        if self.config.model_type in ["ml", "ensemble"]:
            self.ml_analyzer.load_model()
    
        logger.info(f"Sentiment analyzer initialized with model type: {self.config.model_type}")
    
    def analyze_text(self, text: str) -> SentimentResult:
        """Analyze sentiment of a single text"""
        if not text or len(text) < self.config.min_text_length:
            return self._create_empty_result(text)
        
        # Preprocess text
        processed_text = text
        if self.config.enable_preprocessing:
            processed_text = self.preprocessor.clean_text(text)
        
        # Get sentiment analysis
        sentiment_label, sentiment_score, confidence = self._get_sentiment(processed_text)
        
        # Extract emotions
        emotions = {}
        if self.config.enable_emotion_detection:
            emotions = self.preprocessor.detect_emotions(processed_text)
        
        # Extract keywords
        keywords = []
        if self.config.enable_keyword_extraction:
            keywords = self.preprocessor.extract_keywords(processed_text)
        
        # Calculate text statistics
        word_count = len(processed_text.split())
        try:
            sentence_count = len(sent_tokenize(processed_text))
        except:
            sentence_count = processed_text.count('.') + processed_text.count('!') + processed_text.count('?')
        
        return SentimentResult(
            text=text,
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
            confidence=confidence,
            emotions=emotions,
            keywords=keywords,
            processed_text=processed_text,
            word_count=word_count,
            sentence_count=sentence_count
        )
    
    def _get_sentiment(self, text: str) -> Tuple[str, float, float]:
        """Get sentiment using configured method"""
        if self.config.model_type == "vader":
            return self.vader_analyzer.analyze(text)
        
        elif self.config.model_type == "ml" and self.ml_analyzer.is_trained:
            return self.ml_analyzer.analyze(text)
        
        elif self.config.model_type == "ensemble":
            # Combine VADER and ML results
            vader_label, vader_score, vader_conf = self.vader_analyzer.analyze(text)
            
            if self.ml_analyzer.is_trained:
                ml_label, ml_score, ml_conf = self.ml_analyzer.analyze(text)
                
                # Weighted average (VADER: 0.4, ML: 0.6)
                combined_score = 0.4 * vader_score + 0.6 * ml_score
                combined_conf = 0.4 * vader_conf + 0.6 * ml_conf
                
                # Determine final label
                if combined_score >= 0.05:
                    final_label = "positive"
                elif combined_score <= -0.05:
                    final_label = "negative"
                else:
                    final_label = "neutral"
                
                return final_label, combined_score, combined_conf
            else:
                return vader_label, vader_score, vader_conf
        
        else:
            # Fallback to VADER
            return self.vader_analyzer.analyze(text)
    
    def _create_empty_result(self, text: str) -> SentimentResult:
        """Create empty result for invalid text"""
        return SentimentResult(
            text=text,
            sentiment_label="neutral",
            sentiment_score=0.0,
            confidence=0.0,
            emotions={},
            keywords=[],
            processed_text="",
            word_count=0,
            sentence_count=0
        )
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """Analyze sentiment for multiple texts"""
        logger.info(f"Analyzing sentiment for {len(texts)} texts...")
        
        results = []
        for i, text in enumerate(texts):
            if i % 100 == 0:
                logger.debug(f"Processed {i}/{len(texts)} texts")
            
            result = self.analyze_text(text)
            results.append(result)
        
        logger.info(f"Sentiment analysis completed for {len(texts)} texts")
        return results
    
    def analyze_dataframe_chunked(self, df: pd.DataFrame, text_column: str, 
                                  batch_size: int = 1000, parallel: bool = True,
                                  progress_cb=None, max_workers: int = 8) -> pd.DataFrame:
        """Analyze sentiment for a DataFrame in chunks, with optional parallelism and progress callback."""
        total = len(df)
        if total == 0:
            return df.copy()
        
        # Prepare structures
        result_df = df.copy()
        sentiments = {
            'label': [None] * total,
            'score': [0.0] * total,
            'confidence': [0.0] * total,
            'emotions': [None] * total,
            'keywords': [None] * total,
            'word_count': [0] * total,
            'sentence_count': [0] * total,
        }
        
        # Build chunks of indices
        indices = list(range(total))
        chunks = [indices[i:i+batch_size] for i in range(0, total, batch_size)]
        texts = df[text_column].fillna("").tolist()
        
        # Optional Streamlit progress
        st_progress = None
        try:
            import streamlit as st
            st_progress = st.progress(0.0)
        except Exception:
            st_progress = None
        
        def update_progress(done_count):
            if progress_cb:
                try:
                    progress_cb(done_count / total)
                except Exception:
                    pass
            if st_progress:
                try:
                    st_progress.progress(min(done_count / total, 1.0))
                except Exception:
                    pass
        
        done = 0
        
        if parallel:
            # Parallel processing with thread pool
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_chunk = {}
                for chunk in chunks:
                    chunk_texts = [texts[i] for i in chunk]
                    future = executor.submit(self.analyze_batch, chunk_texts)
                    future_to_chunk[future] = chunk
                
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    try:
                        results = future.result()
                    except Exception:
                        # Fallback: empty results
                        results = []
                    # Write results into arrays
                    for local_idx, idx in enumerate(chunk):
                        if local_idx < len(results):
                            r = results[local_idx]
                            sentiments['label'][idx] = r.sentiment_label
                            sentiments['score'][idx] = r.sentiment_score
                            sentiments['confidence'][idx] = r.confidence
                            sentiments['emotions'][idx] = r.emotions
                            sentiments['keywords'][idx] = r.keywords
                            sentiments['word_count'][idx] = r.word_count
                            sentiments['sentence_count'][idx] = r.sentence_count
                    done += len(chunk)
                    update_progress(done)
        else:
            # Sequential processing
            for chunk in chunks:
                chunk_texts = [texts[i] for i in chunk]
                results = self.analyze_batch(chunk_texts)
                for local_idx, idx in enumerate(chunk):
                    r = results[local_idx]
                    sentiments['label'][idx] = r.sentiment_label
                    sentiments['score'][idx] = r.sentiment_score
                    sentiments['confidence'][idx] = r.confidence
                    sentiments['emotions'][idx] = r.emotions
                    sentiments['keywords'][idx] = r.keywords
                    sentiments['word_count'][idx] = r.word_count
                    sentiments['sentence_count'][idx] = r.sentence_count
                done += len(chunk)
                update_progress(done)
        
        # Assign results to dataframe
        result_df['sentiment_label'] = sentiments['label']
        result_df['sentiment_score'] = sentiments['score']
        result_df['sentiment_confidence'] = sentiments['confidence']
        result_df['emotions'] = sentiments['emotions']
        result_df['keywords'] = sentiments['keywords']
        result_df['word_count'] = sentiments['word_count']
        result_df['sentence_count'] = sentiments['sentence_count']
        
        # Add individual emotion columns if any
        first_emotions = next((e for e in sentiments['emotions'] if isinstance(e, dict) and e), None)
        if first_emotions:
            for emotion in first_emotions.keys():
                result_df[f'emotion_{emotion}'] = [ (e or {}).get(emotion, 0.0) for e in sentiments['emotions'] ]
        
        # Clear progress
        update_progress(total)
        
        return result_df
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Simple wrapper for analyze_dataframe_chunked with default parameters"""
        return self.analyze_dataframe_chunked(
            df=df, 
            text_column=text_column, 
            batch_size=1000, 
            parallel=True, 
            max_workers=8
        )
    
    def train_ml_model(self, df: pd.DataFrame, text_column: str, 
                      label_column: str) -> Dict[str, float]:
        """Train ML model on labeled data"""
        logger.info("Training ML sentiment model on provided data...")
        
        # Prepare data
        texts = df[text_column].fillna("").tolist()
        labels = df[label_column].tolist()
        
        # Train the model
        metrics = self.ml_analyzer.train(texts, labels)
        
        return metrics
    
    def get_sentiment_summary(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """Get summary statistics for sentiment analysis results"""
        if not results:
            return {}
        
        # Count sentiments
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_score = 0
        total_confidence = 0
        
        # Emotion aggregation
        all_emotions = {}
        
        for result in results:
            sentiment_counts[result.sentiment_label] += 1
            total_score += result.sentiment_score
            total_confidence += result.confidence
            
            for emotion, score in result.emotions.items():
                if emotion not in all_emotions:
                    all_emotions[emotion] = []
                all_emotions[emotion].append(score)
        
        # Calculate averages
        avg_score = total_score / len(results)
        avg_confidence = total_confidence / len(results)
        
        # Calculate emotion averages
        avg_emotions = {}
        for emotion, scores in all_emotions.items():
            avg_emotions[emotion] = sum(scores) / len(scores)
        
        return {
            'total_reviews': len(results),
            'sentiment_distribution': sentiment_counts,
            'sentiment_percentages': {
                k: (v / len(results)) * 100 for k, v in sentiment_counts.items()
            },
            'average_sentiment_score': avg_score,
            'average_confidence': avg_confidence,
            'emotion_scores': avg_emotions
        }


def main():
    """Demonstrate sentiment analysis functionality"""
    logger.info("🎭 Demonstrating Sentiment Analysis Module")
    
    # Sample review texts
    sample_reviews = [
        "This product is absolutely amazing! I love it so much and would definitely recommend it to everyone.",
        "Terrible quality, worst purchase ever. Very disappointed and angry about this waste of money.",
        "It's okay, nothing special but does the job. Average product for the price.",
        "I'm so excited about this purchase! The quality exceeded my expectations and shipping was fast.",
        "Poor customer service and the product broke after one day. Very frustrated with this experience.",
        "Good value for money. Works as expected and arrived on time. Satisfied with the purchase.",
        "Incredible product! Amazing quality and fantastic customer support. Highly recommended!",
        "Not worth the money. Quality is poor and doesn't match the description. Disappointed.",
        "Perfect! Exactly what I was looking for. Great quality and fast delivery.",
        "Mediocre product. It works but there are better alternatives available."
    ]
    
    # Initialize sentiment analyzer
    config = SentimentConfig(
        model_type="vader",
        enable_preprocessing=True,
        enable_emotion_detection=True,
        enable_keyword_extraction=True
    )
    
    analyzer = SentimentAnalyzer(config)
    
    # Analyze individual texts
    print("\n📝 Individual Sentiment Analysis:")
    for i, review in enumerate(sample_reviews[:3], 1):
        result = analyzer.analyze_text(review)
        print(f"\nReview {i}: {review[:50]}...")
        print(f"  • Sentiment: {result.sentiment_label} (score: {result.sentiment_score:.3f}, confidence: {result.confidence:.3f})")
        print(f"  • Keywords: {', '.join(result.keywords[:5])}")
        print(f"  • Top emotions: {sorted(result.emotions.items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    # Analyze batch
    print("\n📊 Batch Sentiment Analysis:")
    results = analyzer.analyze_batch(sample_reviews)
    summary = analyzer.get_sentiment_summary(results)
    
    print(f"  • Total reviews analyzed: {summary['total_reviews']}")
    print(f"  • Sentiment distribution: {summary['sentiment_distribution']}")
    print(f"  • Average sentiment score: {summary['average_sentiment_score']:.3f}")
    print(f"  • Average confidence: {summary['average_confidence']:.3f}")
    
    # Analyze DataFrame
    print("\n📋 DataFrame Sentiment Analysis:")
    df = pd.DataFrame({
        'review_id': range(1, len(sample_reviews) + 1),
        'review_text': sample_reviews,
        'rating': [5, 1, 3, 5, 1, 4, 5, 2, 5, 3]
    })
    
    df_with_sentiment = analyzer.analyze_dataframe(df, 'review_text')
    
    print(f"  • DataFrame shape: {df_with_sentiment.shape}")
    print(f"  • New columns added: {[col for col in df_with_sentiment.columns if col not in df.columns]}")
    
    # Show correlation between rating and sentiment
    correlation = df_with_sentiment['rating'].corr(df_with_sentiment['sentiment_score'])
    print(f"  • Correlation between rating and sentiment score: {correlation:.3f}")
    
    # Demonstrate different model types
    print("\n🔄 Comparing Different Analysis Methods:")
    test_text = "This product is fantastic! Amazing quality and great value for money."
    
    for model_type in ["vader"]:  # Only VADER for demo since ML needs training data
        config.model_type = model_type
        temp_analyzer = SentimentAnalyzer(config)
        result = temp_analyzer.analyze_text(test_text)
        print(f"  • {model_type.upper()}: {result.sentiment_label} (score: {result.sentiment_score:.3f})")
    
    print("\n✅ Sentiment analysis demonstration completed!")


if __name__ == "__main__":
    main()