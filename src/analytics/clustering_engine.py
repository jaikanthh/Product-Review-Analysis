"""
Advanced Clustering Engine for Customer Segmentation
Provides multiple clustering algorithms and segmentation strategies.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# ML imports with fallbacks
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.decomposition import PCA, TSNE
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import zscore
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ClusteringEngine:
    """Advanced clustering engine for customer segmentation."""
    
    def __init__(self):
        """Initialize the clustering engine."""
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.cluster_results = {}
        self.feature_importance = {}
        
        # Clustering parameters
        self.random_state = 42
        self.n_clusters_default = 5
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize clustering models."""
        try:
            if SKLEARN_AVAILABLE:
                # K-means variants
                self.models['kmeans'] = KMeans(
                    n_clusters=self.n_clusters_default,
                    random_state=self.random_state,
                    n_init=10
                )
                
                # Density-based clustering
                self.models['dbscan'] = DBSCAN(eps=0.5, min_samples=5)
                
                # Hierarchical clustering
                self.models['agglomerative'] = AgglomerativeClustering(
                    n_clusters=self.n_clusters_default
                )
                
                # Gaussian Mixture Models
                self.models['gmm'] = GaussianMixture(
                    n_components=self.n_clusters_default,
                    random_state=self.random_state
                )
                
                # Spectral clustering
                self.models['spectral'] = SpectralClustering(
                    n_clusters=self.n_clusters_default,
                    random_state=self.random_state
                )
                
                # Scalers
                self.scalers['standard'] = StandardScaler()
                self.scalers['minmax'] = MinMaxScaler()
                self.scalers['robust'] = RobustScaler()
                
                # Dimensionality reduction
                self.models['pca'] = PCA(n_components=2, random_state=self.random_state)
                self.models['tsne'] = TSNE(n_components=2, random_state=self.random_state)
            
            self.logger.info("Clustering models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing clustering models: {str(e)}")
    
    def prepare_clustering_features(self, reviews_df: pd.DataFrame, 
                                   users_df: pd.DataFrame = None,
                                   products_df: pd.DataFrame = None) -> pd.DataFrame:
        """Prepare features for customer clustering using vectorized operations."""
        try:
            self.logger.info(f"Preparing clustering features for {len(reviews_df)} reviews from {reviews_df['user_id'].nunique()} users")
            
            # Ensure timestamp column exists
            if 'timestamp' not in reviews_df.columns and 'created_at' in reviews_df.columns:
                reviews_df = reviews_df.copy()
                reviews_df['timestamp'] = pd.to_datetime(reviews_df['created_at'], errors='coerce')
            elif 'timestamp' in reviews_df.columns:
                reviews_df = reviews_df.copy()
                reviews_df['timestamp'] = pd.to_datetime(reviews_df['timestamp'], errors='coerce')
            
            # Use vectorized groupby operations for better performance
            user_groups = reviews_df.groupby('user_id')
            
            # Basic review statistics
            basic_stats = user_groups.agg({
                'rating': ['count', 'mean', 'std', 'min', 'max'],
                'product_id': 'nunique'
            }).round(4)
            
            # Flatten column names
            basic_stats.columns = ['total_reviews', 'avg_rating', 'rating_std', 'min_rating', 'max_rating', 'unique_products']
            basic_stats['rating_range'] = basic_stats['max_rating'] - basic_stats['min_rating']
            basic_stats['product_diversity'] = basic_stats['unique_products'] / basic_stats['total_reviews']
            basic_stats['rating_std'] = basic_stats['rating_std'].fillna(0)
            
            # Rating distribution (vectorized)
            rating_dist = reviews_df.groupby(['user_id', 'rating']).size().unstack(fill_value=0)
            for rating in range(1, 6):
                if rating not in rating_dist.columns:
                    rating_dist[rating] = 0
                basic_stats[f'rating_{rating}_count'] = rating_dist[rating]
                basic_stats[f'rating_{rating}_ratio'] = rating_dist[rating] / basic_stats['total_reviews']
            
            # Temporal features (if timestamp available)
            if 'timestamp' in reviews_df.columns and not reviews_df['timestamp'].isna().all():
                temporal_stats = user_groups['timestamp'].agg(['min', 'max', 'count'])
                temporal_stats['days_active'] = (temporal_stats['max'] - temporal_stats['min']).dt.days + 1
                temporal_stats['reviews_per_day'] = temporal_stats['count'] / temporal_stats['days_active']
                temporal_stats['last_review_days_ago'] = (pd.Timestamp.now() - temporal_stats['max']).dt.days
                
                # Add weekend ratio (vectorized)
                reviews_df['is_weekend'] = reviews_df['timestamp'].dt.dayofweek.isin([5, 6])
                weekend_stats = reviews_df.groupby('user_id')['is_weekend'].agg(['sum', 'count'])
                temporal_stats['weekend_review_ratio'] = weekend_stats['sum'] / weekend_stats['count']
                temporal_stats['review_frequency_consistency'] = 1.0  # Simplified for performance
                
                # Merge temporal features
                basic_stats = basic_stats.join(temporal_stats[['days_active', 'reviews_per_day', 'last_review_days_ago', 'weekend_review_ratio', 'review_frequency_consistency']])
            else:
                # Default temporal values
                basic_stats['days_active'] = 1
                basic_stats['reviews_per_day'] = basic_stats['total_reviews']
                basic_stats['last_review_days_ago'] = 0
                basic_stats['weekend_review_ratio'] = 0.5
                basic_stats['review_frequency_consistency'] = 1.0
            
            # Additional derived features
            basic_stats['repeat_purchase_ratio'] = 1 - basic_stats['product_diversity']
            
            # Text engagement features (vectorized)
            if 'review_text' in reviews_df.columns:
                reviews_df['text_length'] = reviews_df['review_text'].fillna('').str.len()
                text_stats = user_groups['text_length'].agg(['mean', 'std', 'min', 'max', 'sum'])
                text_stats.columns = ['avg_text_length', 'text_length_std', 'min_text_length', 'max_text_length', 'total_text_length']
                text_stats['text_length_std'] = text_stats['text_length_std'].fillna(0)
                
                # Text engagement levels (vectorized)
                reviews_df['is_short'] = reviews_df['text_length'] < 50
                reviews_df['is_medium'] = (reviews_df['text_length'] >= 50) & (reviews_df['text_length'] < 200)
                reviews_df['is_long'] = reviews_df['text_length'] >= 200
                
                engagement_stats = user_groups[['is_short', 'is_medium', 'is_long']].sum()
                engagement_stats.columns = ['short_reviews', 'medium_reviews', 'long_reviews']
                engagement_stats['short_review_ratio'] = engagement_stats['short_reviews'] / basic_stats['total_reviews']
                engagement_stats['long_review_ratio'] = engagement_stats['long_reviews'] / basic_stats['total_reviews']
                
                basic_stats = basic_stats.join(text_stats).join(engagement_stats)
            else:
                # Default text features
                for col in ['avg_text_length', 'text_length_std', 'max_text_length', 
                           'min_text_length', 'total_text_length', 'short_reviews',
                           'medium_reviews', 'long_reviews', 'short_review_ratio', 'long_review_ratio']:
                    basic_stats[col] = 0
            
            # Sentiment features (vectorized if available)
            if 'sentiment_score' in reviews_df.columns:
                sentiment_stats = user_groups['sentiment_score'].agg(['mean', 'std'])
                sentiment_stats.columns = ['avg_sentiment', 'sentiment_std']
                sentiment_stats['sentiment_std'] = sentiment_stats['sentiment_std'].fillna(0)
                
                reviews_df['is_positive'] = reviews_df['sentiment_score'] > 0
                reviews_df['is_negative'] = reviews_df['sentiment_score'] < 0
                sentiment_ratios = user_groups[['is_positive', 'is_negative']].mean()
                sentiment_ratios.columns = ['positive_sentiment_ratio', 'negative_sentiment_ratio']
                
                basic_stats = basic_stats.join(sentiment_stats).join(sentiment_ratios)
            else:
                basic_stats['avg_sentiment'] = 0
                basic_stats['sentiment_std'] = 0
                basic_stats['positive_sentiment_ratio'] = 0.5
                basic_stats['negative_sentiment_ratio'] = 0.5
            
            # Derived engagement metrics
            basic_stats['rating_consistency'] = 1 / (1 + basic_stats['rating_std'])
            basic_stats['engagement_score'] = (
                basic_stats['total_reviews'] * 0.3 +
                basic_stats['avg_text_length'] * 0.001 +
                basic_stats['product_diversity'] * 10 +
                basic_stats['reviews_per_day'] * 5
            )
            basic_stats['review_quality_score'] = (
                basic_stats['avg_text_length'] * 0.01 +
                basic_stats['rating_consistency'] * 2 +
                (5 - abs(basic_stats['avg_rating'] - 3)) * 0.5
            )
            
            # Reset index to make user_id a column
            customer_df = basic_stats.reset_index()
            
            # Add user demographic features if available
            if users_df is not None:
                customer_df = customer_df.merge(users_df, on='user_id', how='left')
            
            # Fill missing values
            customer_df = customer_df.fillna(0)
            
            # Replace infinite values
            customer_df = customer_df.replace([np.inf, -np.inf], 0)
            
            return customer_df
            
        except Exception as e:
            self.logger.error(f"Error preparing clustering features: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_frequency_consistency(self, user_reviews: pd.DataFrame) -> float:
        """Calculate review frequency consistency."""
        try:
            if len(user_reviews) < 3:
                return 1.0
            
            timestamps = user_reviews['timestamp'].sort_values()
            intervals = timestamps.diff().dt.days.dropna()
            
            if len(intervals) == 0:
                return 1.0
            
            # Coefficient of variation (lower = more consistent)
            cv = intervals.std() / intervals.mean() if intervals.mean() > 0 else 0
            consistency = 1 / (1 + cv)  # Convert to consistency score
            
            return consistency
            
        except Exception:
            return 1.0
    
    def _calculate_weekend_ratio(self, user_reviews: pd.DataFrame) -> float:
        """Calculate ratio of reviews posted on weekends."""
        try:
            if len(user_reviews) == 0:
                return 0.5
            
            weekend_reviews = user_reviews['timestamp'].dt.dayofweek.isin([5, 6]).sum()
            return weekend_reviews / len(user_reviews)
            
        except Exception:
            return 0.5
    
    def find_optimal_clusters(self, features_df: pd.DataFrame, 
                             max_clusters: int = 12, method: str = 'kmeans') -> Dict[str, Any]:
        """Find optimal number of clusters using various metrics."""
        try:
            if not SKLEARN_AVAILABLE:
                return {'error': 'Required dependencies not available'}
            
            # Prepare numeric feature matrix (exclude user_id)
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col != 'user_id']
            X = features_df[feature_cols].copy()
            
            # Drop columns with zero/near-zero variance to reduce noise
            variances = X.var(numeric_only=True).fillna(0)
            keep_cols = variances[variances > 1e-8].index.tolist()
            if len(keep_cols) >= 2:
                feature_cols = keep_cols
                X = X[feature_cols]
            
            # Scale features
            X_scaled = self.scalers['standard'].fit_transform(X)
            
            # Test different numbers of clusters (cap to 12)
            tested_max = min(max_clusters, 12)
            cluster_range = range(2, tested_max + 1)
            metrics = {
                'inertia': [],
                'silhouette': [],
                'calinski_harabasz': [],
                'davies_bouldin': []
            }
            
            for n_clusters in cluster_range:
                if method == 'kmeans':
                    model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
                    labels = model.fit_predict(X_scaled)
                    metrics['inertia'].append(model.inertia_)
                elif method == 'gmm':
                    model = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
                    labels = model.fit_predict(X_scaled)
                    metrics['inertia'].append(-model.score(X_scaled))
                else:
                    continue
                
                # Calculate clustering metrics only if we have >1 cluster
                if len(set(labels)) > 1:
                    try:
                        metrics['silhouette'].append(silhouette_score(X_scaled, labels))
                    except Exception:
                        metrics['silhouette'].append(np.nan)
                    try:
                        metrics['calinski_harabasz'].append(calinski_harabasz_score(X_scaled, labels))
                    except Exception:
                        metrics['calinski_harabasz'].append(np.nan)
                    try:
                        metrics['davies_bouldin'].append(davies_bouldin_score(X_scaled, labels))
                    except Exception:
                        metrics['davies_bouldin'].append(np.nan)
                else:
                    metrics['silhouette'].append(np.nan)
                    metrics['calinski_harabasz'].append(np.nan)
                    metrics['davies_bouldin'].append(np.nan)
            
            # Find optimal number of clusters using valid metrics
            optimal_clusters = self._determine_optimal_clusters(metrics, cluster_range)
            
            return {
                'optimal_clusters': optimal_clusters,
                'metrics': metrics,
                'cluster_range': list(cluster_range),
                'method': method
            }
        except Exception as e:
            self.logger.error(f"Error finding optimal clusters: {str(e)}")
            return {'error': str(e)}
    
    def _determine_optimal_clusters(self, metrics: Dict[str, List], 
                                   cluster_range: range) -> int:
        """Determine optimal number of clusters from metrics."""
        try:
            # Use silhouette as primary criterion when valid
            sil = np.array(metrics['silhouette'], dtype=float)
            valid_idx = np.where(~np.isnan(sil))[0]
            if len(valid_idx) > 0:
                best_idx = valid_idx[np.argmax(sil[valid_idx])]
                return cluster_range[best_idx]
            
            # Fall back to CH index if silhouette invalid
            ch = np.array(metrics['calinski_harabasz'], dtype=float)
            valid_idx = np.where(~np.isnan(ch))[0]
            if len(valid_idx) > 0:
                best_idx = valid_idx[np.argmax(ch[valid_idx])]
                return cluster_range[best_idx]
            
            # If all metrics invalid, default to 3
            return 3
        except Exception:
            return 3
    
    def perform_clustering(self, features_df: pd.DataFrame, 
                          method: str = 'kmeans', n_clusters: int = None,
                          scaling_method: str = 'standard') -> Dict[str, Any]:
        """Perform clustering using specified method."""
        try:
            if not SKLEARN_AVAILABLE:
                return {'error': 'Required dependencies not available'}
            
            # Prepare features
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col != 'user_id']
            X = features_df[feature_cols].copy()
            # Drop columns with zero/near-zero variance
            variances = X.var(numeric_only=True).fillna(0)
            keep_cols = variances[variances > 1e-8].index.tolist()
            if len(keep_cols) >= 2:
                feature_cols = keep_cols
                X = X[feature_cols]
            
            # Scale features
            scaler = self.scalers[scaling_method]
            X_scaled = scaler.fit_transform(X)
            
            # Set number of clusters
            if n_clusters is None:
                n_clusters = self.n_clusters_default
            
            # Fit model and get labels
            if method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init=10)
                labels = model.fit_predict(X_scaled)
                results = {'inertia': getattr(model, 'inertia_', None)}
            elif method == 'gmm':
                model = GaussianMixture(n_components=n_clusters, random_state=self.random_state)
                labels = model.fit_predict(X_scaled)
                results = {'log_likelihood': model.score(X_scaled)}
            elif method == 'agglomerative':
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(X_scaled)
                results = {}
            elif method == 'spectral':
                model = SpectralClustering(n_clusters=n_clusters, random_state=self.random_state)
                labels = model.fit_predict(X_scaled)
                results = {}
            else:
                return {'error': f'Unknown clustering method: {method}'}
            
            # Compute metrics when valid (>1 cluster)
            if len(set(labels)) > 1:
                try:
                    results['silhouette_score'] = silhouette_score(X_scaled, labels)
                except Exception:
                    results['silhouette_score'] = np.nan
                try:
                    results['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, labels)
                except Exception:
                    results['calinski_harabasz_score'] = np.nan
                try:
                    results['Davies-Bouldin Score'] = davies_bouldin_score(X_scaled, labels)
                except Exception:
                    results['Davies-Bouldin Score'] = np.nan
            else:
                results['silhouette_score'] = np.nan
                results['calinski_harabasz_score'] = np.nan
                results['Davies-Bouldin Score'] = np.nan
            
            # Build result dataframe and analysis
            result_df = features_df.copy()
            result_df['cluster'] = labels
            cluster_analysis = self._analyze_clusters(result_df, feature_cols)
            
            results.update({
                'method': method,
                'n_clusters': n_clusters,
                'n_clusters_found': len(set(labels)),
                'labels': labels.tolist(),
                'cluster_analysis': cluster_analysis,
                'feature_columns': feature_cols,
                'scaling_method': scaling_method
            })
            
            # Store results
            self.cluster_results[method] = results
            return results
        except Exception as e:
            self.logger.error(f"Error performing clustering: {str(e)}")
            return {'error': str(e)}
    
    def visualize_clusters(self, features_df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """Create PCA visualization data for clusters."""
        try:
            if not SKLEARN_AVAILABLE:
                return {'error': 'Required dependencies not available'}
            
            # Prepare features (same as in clustering)
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols if col != 'user_id']
            X = features_df[feature_cols].copy()
            
            # Drop low-variance columns
            variances = X.var(numeric_only=True).fillna(0)
            keep_cols = variances[variances > 1e-8].index.tolist()
            if len(keep_cols) >= 2:
                feature_cols = keep_cols
                X = X[feature_cols]
            
            # Scale features
            scaler = self.scalers['standard']
            X_scaled = scaler.fit_transform(X)
            
            # Sample for visualization if dataset is large
            if len(X_scaled) > 5000:
                sample_idx = np.random.choice(len(X_scaled), 5000, replace=False)
                X_viz = X_scaled[sample_idx]
                labels_viz = labels[sample_idx]
            else:
                X_viz = X_scaled
                labels_viz = labels
            
            # Apply PCA for 2D visualization
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_viz)
            
            return {
                'x': X_pca[:, 0].tolist(),
                'y': X_pca[:, 1].tolist(),
                'cluster': labels_viz.tolist(),
                'explained_variance': pca.explained_variance_ratio_.tolist(),
                'feature_columns': feature_cols
            }
            
        except Exception as e:
             self.logger.error(f"Error creating cluster visualization: {str(e)}")
             return {'error': str(e)}
    
    def _analyze_clusters(self, result_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """Analyze cluster characteristics and generate insights."""
        try:
            cluster_analysis = {}
            
            for cluster_id in result_df['cluster'].unique():
                cluster_data = result_df[result_df['cluster'] == cluster_id]
                cluster_size = len(cluster_data)
                cluster_percentage = (cluster_size / len(result_df)) * 100
                
                # Calculate feature means for this cluster
                cluster_features = cluster_data[feature_cols].mean()
                
                # Calculate global means for comparison
                global_features = result_df[feature_cols].mean()
                
                # Find distinguishing features (highest relative differences)
                feature_importance = {}
                for feature in feature_cols:
                    if global_features[feature] != 0:
                        relative_diff = abs(cluster_features[feature] - global_features[feature]) / abs(global_features[feature])
                        feature_importance[feature] = relative_diff
                    else:
                        feature_importance[feature] = 0
                
                # Sort features by importance
                top_features = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])
                
                # Generate characteristics based on feature values
                characteristics = []
                for feature, importance in list(top_features.items())[:3]:
                    cluster_val = cluster_features[feature]
                    global_val = global_features[feature]
                    
                    if cluster_val > global_val:
                        characteristics.append(f"High {feature.replace('_', ' ')}")
                    else:
                        characteristics.append(f"Low {feature.replace('_', ' ')}")
                
                cluster_analysis[f'cluster_{cluster_id}'] = {
                    'size': cluster_size,
                    'percentage': cluster_percentage,
                    'characteristics': characteristics,
                    'top_features': top_features,
                    'feature_means': cluster_features.to_dict()
                }
            
            return cluster_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing clusters: {str(e)}")
            return {}