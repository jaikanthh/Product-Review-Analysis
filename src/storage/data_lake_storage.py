"""
Data Lake Storage System
Implements scalable data lake architecture for multi-format data storage
Provides partitioning, compression, metadata management, and data cataloging
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import json
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
import shutil
import gzip
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataLakeConfig:
    """Configuration for data lake storage"""
    base_path: str
    compression: str = 'snappy'  # snappy, gzip, lz4, brotli
    partition_columns: List[str] = None
    max_file_size_mb: int = 128
    retention_days: int = 365
    enable_metadata_catalog: bool = True
    enable_data_versioning: bool = True
    
    def __post_init__(self):
        if self.partition_columns is None:
            self.partition_columns = []


@dataclass
class DatasetMetadata:
    """Metadata for datasets in the data lake"""
    dataset_name: str
    schema: Dict[str, str]
    partition_columns: List[str]
    file_format: str
    compression: str
    created_at: datetime
    updated_at: datetime
    total_files: int
    total_size_bytes: int
    row_count: int
    data_quality_score: float
    tags: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class DataLakeStorage:
    """Data Lake storage implementation with advanced features"""
    
    def __init__(self, config: DataLakeConfig):
        self.config = config
        self.base_path = Path(config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize data lake structure
        self.raw_path = self.base_path / "raw"
        self.processed_path = self.base_path / "processed"
        self.curated_path = self.base_path / "curated"
        self.metadata_path = self.base_path / "metadata"
        self.archive_path = self.base_path / "archive"
        
        for path in [self.raw_path, self.processed_path, self.curated_path, 
                     self.metadata_path, self.archive_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata catalog
        self.catalog_file = self.metadata_path / "catalog.json"
        self.catalog = self._load_catalog()
        
        logger.info(f"Data Lake initialized at {self.base_path}")
    
    def _load_catalog(self) -> Dict[str, DatasetMetadata]:
        """Load metadata catalog from file"""
        if self.catalog_file.exists():
            try:
                with open(self.catalog_file, 'r') as f:
                    catalog_data = json.load(f)
                
                catalog = {}
                for name, metadata_dict in catalog_data.items():
                    # Convert datetime strings back to datetime objects
                    metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
                    metadata_dict['updated_at'] = datetime.fromisoformat(metadata_dict['updated_at'])
                    catalog[name] = DatasetMetadata(**metadata_dict)
                
                logger.info(f"Loaded catalog with {len(catalog)} datasets")
                return catalog
                
            except Exception as e:
                logger.warning(f"Could not load catalog: {e}")
        
        return {}
    
    def _save_catalog(self):
        """Save metadata catalog to file"""
        if not self.config.enable_metadata_catalog:
            return
        
        try:
            catalog_data = {}
            for name, metadata in self.catalog.items():
                metadata_dict = asdict(metadata)
                # Convert datetime objects to strings for JSON serialization
                metadata_dict['created_at'] = metadata.created_at.isoformat()
                metadata_dict['updated_at'] = metadata.updated_at.isoformat()
                catalog_data[name] = metadata_dict
            
            with open(self.catalog_file, 'w') as f:
                json.dump(catalog_data, f, indent=2)
            
            logger.debug("Catalog saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save catalog: {e}")
    
    def write_dataset(self, data: Union[pd.DataFrame, List[Dict]], dataset_name: str, 
                     layer: str = "raw", partition_columns: List[str] = None,
                     file_format: str = "parquet", compression: str = None,
                     overwrite: bool = False) -> str:
        """Write dataset to data lake with partitioning and compression"""
        
        if layer not in ["raw", "processed", "curated"]:
            raise ValueError("Layer must be one of: raw, processed, curated")
        
        if compression is None:
            compression = self.config.compression
        
        if partition_columns is None:
            partition_columns = self.config.partition_columns
        
        # Convert to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Determine target path
        layer_path = getattr(self, f"{layer}_path")
        dataset_path = layer_path / dataset_name
        
        if overwrite and dataset_path.exists():
            shutil.rmtree(dataset_path)
        
        dataset_path.mkdir(parents=True, exist_ok=True)
        
        try:
            if file_format.lower() == "parquet":
                output_path = self._write_parquet_dataset(df, dataset_path, partition_columns, compression)
            elif file_format.lower() == "json":
                output_path = self._write_json_dataset(df, dataset_path, partition_columns, compression)
            elif file_format.lower() == "csv":
                output_path = self._write_csv_dataset(df, dataset_path, partition_columns, compression)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            # Update metadata catalog
            self._update_catalog(dataset_name, df, layer, partition_columns, file_format, compression, dataset_path)
            
            logger.info(f"Dataset '{dataset_name}' written to {layer} layer: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Failed to write dataset {dataset_name}: {e}")
            raise
    
    def _write_parquet_dataset(self, df: pd.DataFrame, dataset_path: Path, 
                              partition_columns: List[str], compression: str) -> Path:
        """Write DataFrame as partitioned Parquet dataset"""
        
        # Add timestamp columns for partitioning if not present
        if 'year' not in df.columns and 'created_at' in df.columns:
            df['year'] = pd.to_datetime(df['created_at']).dt.year
        if 'month' not in df.columns and 'created_at' in df.columns:
            df['month'] = pd.to_datetime(df['created_at']).dt.month
        if 'day' not in df.columns and 'created_at' in df.columns:
            df['day'] = pd.to_datetime(df['created_at']).dt.day
        
        # Convert to PyArrow table
        table = pa.Table.from_pandas(df)
        
        # Write partitioned dataset
        if partition_columns:
            pq.write_to_dataset(
                table,
                root_path=str(dataset_path),
                partition_cols=partition_columns,
                compression=compression,
                use_dictionary=True,
                row_group_size=50000,
                data_page_size=1024*1024  # 1MB pages
            )
        else:
            # Write as single file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = dataset_path / f"data_{timestamp}.parquet"
            pq.write_table(table, file_path, compression=compression)
        
        return dataset_path
    
    def _write_json_dataset(self, df: pd.DataFrame, dataset_path: Path, 
                           partition_columns: List[str], compression: str) -> Path:
        """Write DataFrame as JSON dataset with optional compression"""
        
        if partition_columns:
            # Group by partition columns and write separate files
            for partition_values, group_df in df.groupby(partition_columns):
                if not isinstance(partition_values, tuple):
                    partition_values = (partition_values,)
                
                # Create partition directory structure
                partition_path = dataset_path
                for col, val in zip(partition_columns, partition_values):
                    partition_path = partition_path / f"{col}={val}"
                partition_path.mkdir(parents=True, exist_ok=True)
                
                # Write JSON file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = partition_path / f"data_{timestamp}.json"
                
                if compression == 'gzip':
                    with gzip.open(f"{file_path}.gz", 'wt') as f:
                        group_df.to_json(f, orient='records', lines=True)
                else:
                    group_df.to_json(file_path, orient='records', lines=True)
        else:
            # Write as single file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = dataset_path / f"data_{timestamp}.json"
            
            if compression == 'gzip':
                with gzip.open(f"{file_path}.gz", 'wt') as f:
                    df.to_json(f, orient='records', lines=True)
            else:
                df.to_json(file_path, orient='records', lines=True)
        
        return dataset_path
    
    def _write_csv_dataset(self, df: pd.DataFrame, dataset_path: Path, 
                          partition_columns: List[str], compression: str) -> Path:
        """Write DataFrame as CSV dataset with optional compression"""
        
        if partition_columns:
            # Group by partition columns and write separate files
            for partition_values, group_df in df.groupby(partition_columns):
                if not isinstance(partition_values, tuple):
                    partition_values = (partition_values,)
                
                # Create partition directory structure
                partition_path = dataset_path
                for col, val in zip(partition_columns, partition_values):
                    partition_path = partition_path / f"{col}={val}"
                partition_path.mkdir(parents=True, exist_ok=True)
                
                # Write CSV file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = partition_path / f"data_{timestamp}.csv"
                
                if compression == 'gzip':
                    group_df.to_csv(f"{file_path}.gz", index=False, compression='gzip')
                else:
                    group_df.to_csv(file_path, index=False)
        else:
            # Write as single file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = dataset_path / f"data_{timestamp}.csv"
            
            if compression == 'gzip':
                df.to_csv(f"{file_path}.gz", index=False, compression='gzip')
            else:
                df.to_csv(file_path, index=False)
        
        return dataset_path
    
    def _update_catalog(self, dataset_name: str, df: pd.DataFrame, layer: str,
                       partition_columns: List[str], file_format: str, 
                       compression: str, dataset_path: Path):
        """Update metadata catalog with dataset information"""
        
        if not self.config.enable_metadata_catalog:
            return
        
        # Calculate dataset statistics
        total_size = sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file())
        total_files = len(list(dataset_path.rglob('*'))) - len(list(dataset_path.rglob('*/')))
        
        # Infer schema
        schema = {}
        for col, dtype in df.dtypes.items():
            schema[col] = str(dtype)
        
        # Calculate data quality score (simplified)
        null_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
        data_quality_score = max(0.0, 1.0 - null_percentage)
        
        # Create or update metadata
        now = datetime.now()
        
        if dataset_name in self.catalog:
            metadata = self.catalog[dataset_name]
            metadata.updated_at = now
            metadata.total_files = total_files
            metadata.total_size_bytes = total_size
            metadata.row_count = len(df)
            metadata.data_quality_score = data_quality_score
        else:
            metadata = DatasetMetadata(
                dataset_name=dataset_name,
                schema=schema,
                partition_columns=partition_columns,
                file_format=file_format,
                compression=compression,
                created_at=now,
                updated_at=now,
                total_files=total_files,
                total_size_bytes=total_size,
                row_count=len(df),
                data_quality_score=data_quality_score,
                tags=[layer],
                description=f"Dataset in {layer} layer"
            )
        
        self.catalog[dataset_name] = metadata
        self._save_catalog()
    
    def read_dataset(self, dataset_name: str, layer: str = "processed", 
                    filters: List[Tuple] = None, columns: List[str] = None,
                    limit: int = None) -> pd.DataFrame:
        """Read dataset from data lake with optional filtering"""
        
        if layer not in ["raw", "processed", "curated"]:
            raise ValueError("Layer must be one of: raw, processed, curated")
        
        layer_path = getattr(self, f"{layer}_path")
        dataset_path = layer_path / dataset_name
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found in {layer} layer")
        
        try:
            # Check if it's a Parquet dataset
            parquet_files = list(dataset_path.rglob("*.parquet"))
            if parquet_files:
                return self._read_parquet_dataset(dataset_path, filters, columns, limit)
            
            # Check for JSON files
            json_files = list(dataset_path.rglob("*.json*"))
            if json_files:
                return self._read_json_dataset(dataset_path, filters, columns, limit)
            
            # Check for CSV files
            csv_files = list(dataset_path.rglob("*.csv*"))
            if csv_files:
                return self._read_csv_dataset(dataset_path, filters, columns, limit)
            
            raise ValueError(f"No supported files found in dataset {dataset_name}")
            
        except Exception as e:
            logger.error(f"Failed to read dataset {dataset_name}: {e}")
            raise
    
    def _read_parquet_dataset(self, dataset_path: Path, filters: List[Tuple] = None,
                             columns: List[str] = None, limit: int = None) -> pd.DataFrame:
        """Read Parquet dataset with filtering and column selection"""
        
        try:
            # Use PyArrow dataset for efficient reading
            dataset = ds.dataset(str(dataset_path), format="parquet")
            
            # Apply filters
            filter_expression = None
            if filters:
                filter_expressions = []
                for filter_tuple in filters:
                    if len(filter_tuple) == 3:
                        col, op, value = filter_tuple
                        if op == "==":
                            filter_expressions.append(ds.field(col) == value)
                        elif op == "!=":
                            filter_expressions.append(ds.field(col) != value)
                        elif op == ">":
                            filter_expressions.append(ds.field(col) > value)
                        elif op == ">=":
                            filter_expressions.append(ds.field(col) >= value)
                        elif op == "<":
                            filter_expressions.append(ds.field(col) < value)
                        elif op == "<=":
                            filter_expressions.append(ds.field(col) <= value)
                        elif op == "in":
                            filter_expressions.append(ds.field(col).isin(value))
                
                if filter_expressions:
                    filter_expression = filter_expressions[0]
                    for expr in filter_expressions[1:]:
                        filter_expression = filter_expression & expr
            
            # Read data
            table = dataset.to_table(filter=filter_expression, columns=columns)
            df = table.to_pandas()
            
            # Apply limit
            if limit and len(df) > limit:
                df = df.head(limit)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to read Parquet dataset: {e}")
            raise
    
    def _read_json_dataset(self, dataset_path: Path, filters: List[Tuple] = None,
                          columns: List[str] = None, limit: int = None) -> pd.DataFrame:
        """Read JSON dataset files"""
        
        json_files = list(dataset_path.rglob("*.json*"))
        dataframes = []
        
        for file_path in json_files:
            try:
                if file_path.suffix == '.gz':
                    with gzip.open(file_path, 'rt') as f:
                        df = pd.read_json(f, lines=True)
                else:
                    df = pd.read_json(file_path, lines=True)
                
                dataframes.append(df)
                
            except Exception as e:
                logger.warning(f"Could not read JSON file {file_path}: {e}")
        
        if not dataframes:
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Apply filters (simplified)
        if filters:
            for col, op, value in filters:
                if col in combined_df.columns:
                    if op == "==":
                        combined_df = combined_df[combined_df[col] == value]
                    elif op == "!=":
                        combined_df = combined_df[combined_df[col] != value]
                    elif op == ">":
                        combined_df = combined_df[combined_df[col] > value]
                    elif op == ">=":
                        combined_df = combined_df[combined_df[col] >= value]
                    elif op == "<":
                        combined_df = combined_df[combined_df[col] < value]
                    elif op == "<=":
                        combined_df = combined_df[combined_df[col] <= value]
                    elif op == "in":
                        combined_df = combined_df[combined_df[col].isin(value)]
        
        # Select columns
        if columns:
            available_columns = [col for col in columns if col in combined_df.columns]
            combined_df = combined_df[available_columns]
        
        # Apply limit
        if limit and len(combined_df) > limit:
            combined_df = combined_df.head(limit)
        
        return combined_df
    
    def _read_csv_dataset(self, dataset_path: Path, filters: List[Tuple] = None,
                         columns: List[str] = None, limit: int = None) -> pd.DataFrame:
        """Read CSV dataset files"""
        
        csv_files = list(dataset_path.rglob("*.csv*"))
        dataframes = []
        
        for file_path in csv_files:
            try:
                if file_path.suffix == '.gz':
                    df = pd.read_csv(file_path, compression='gzip')
                else:
                    df = pd.read_csv(file_path)
                
                dataframes.append(df)
                
            except Exception as e:
                logger.warning(f"Could not read CSV file {file_path}: {e}")
        
        if not dataframes:
            return pd.DataFrame()
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        
        # Apply filters (simplified)
        if filters:
            for col, op, value in filters:
                if col in combined_df.columns:
                    if op == "==":
                        combined_df = combined_df[combined_df[col] == value]
                    elif op == "!=":
                        combined_df = combined_df[combined_df[col] != value]
                    elif op == ">":
                        combined_df = combined_df[combined_df[col] > value]
                    elif op == ">=":
                        combined_df = combined_df[combined_df[col] >= value]
                    elif op == "<":
                        combined_df = combined_df[combined_df[col] < value]
                    elif op == "<=":
                        combined_df = combined_df[combined_df[col] <= value]
                    elif op == "in":
                        combined_df = combined_df[combined_df[col].isin(value)]
        
        # Select columns
        if columns:
            available_columns = [col for col in columns if col in combined_df.columns]
            combined_df = combined_df[available_columns]
        
        # Apply limit
        if limit and len(combined_df) > limit:
            combined_df = combined_df.head(limit)
        
        return combined_df
    
    def list_datasets(self, layer: str = None) -> List[Dict[str, Any]]:
        """List all datasets in the data lake"""
        
        datasets = []
        
        if layer:
            layers = [layer]
        else:
            layers = ["raw", "processed", "curated"]
        
        for layer_name in layers:
            layer_path = getattr(self, f"{layer_name}_path")
            
            for dataset_path in layer_path.iterdir():
                if dataset_path.is_dir():
                    dataset_info = {
                        'name': dataset_path.name,
                        'layer': layer_name,
                        'path': str(dataset_path)
                    }
                    
                    # Add metadata if available
                    if dataset_path.name in self.catalog:
                        metadata = self.catalog[dataset_path.name]
                        dataset_info.update({
                            'schema': metadata.schema,
                            'row_count': metadata.row_count,
                            'size_bytes': metadata.total_size_bytes,
                            'created_at': metadata.created_at.isoformat(),
                            'updated_at': metadata.updated_at.isoformat(),
                            'data_quality_score': metadata.data_quality_score,
                            'tags': metadata.tags
                        })
                    
                    datasets.append(dataset_info)
        
        return datasets
    
    def get_dataset_metadata(self, dataset_name: str) -> Optional[DatasetMetadata]:
        """Get metadata for a specific dataset"""
        return self.catalog.get(dataset_name)
    
    def delete_dataset(self, dataset_name: str, layer: str = "raw", archive: bool = True):
        """Delete dataset with optional archiving"""
        
        layer_path = getattr(self, f"{layer}_path")
        dataset_path = layer_path / dataset_name
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found in {layer} layer")
        
        try:
            if archive:
                # Move to archive
                archive_dataset_path = self.archive_path / f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.move(str(dataset_path), str(archive_dataset_path))
                logger.info(f"Dataset {dataset_name} archived to {archive_dataset_path}")
            else:
                # Permanently delete
                shutil.rmtree(dataset_path)
                logger.info(f"Dataset {dataset_name} permanently deleted")
            
            # Remove from catalog
            if dataset_name in self.catalog:
                del self.catalog[dataset_name]
                self._save_catalog()
            
        except Exception as e:
            logger.error(f"Failed to delete dataset {dataset_name}: {e}")
            raise
    
    def optimize_dataset(self, dataset_name: str, layer: str = "processed"):
        """Optimize dataset by compacting files and updating statistics"""
        
        try:
            # Read the dataset
            df = self.read_dataset(dataset_name, layer)
            
            # Write it back with optimized settings
            self.write_dataset(
                df, 
                dataset_name, 
                layer, 
                partition_columns=self.config.partition_columns,
                file_format="parquet",
                compression=self.config.compression,
                overwrite=True
            )
            
            logger.info(f"Dataset {dataset_name} optimized successfully")
            
        except Exception as e:
            logger.error(f"Failed to optimize dataset {dataset_name}: {e}")
            raise
    
    def cleanup_old_data(self, retention_days: int = None):
        """Clean up old data based on retention policy"""
        
        if retention_days is None:
            retention_days = self.config.retention_days
        
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        cleaned_datasets = []
        
        for dataset_name, metadata in self.catalog.items():
            if metadata.created_at < cutoff_date:
                try:
                    # Archive old datasets
                    for layer in ["raw", "processed", "curated"]:
                        layer_path = getattr(self, f"{layer}_path")
                        dataset_path = layer_path / dataset_name
                        
                        if dataset_path.exists():
                            self.delete_dataset(dataset_name, layer, archive=True)
                            cleaned_datasets.append(f"{dataset_name} ({layer})")
                            break
                    
                except Exception as e:
                    logger.warning(f"Could not clean up dataset {dataset_name}: {e}")
        
        logger.info(f"Cleaned up {len(cleaned_datasets)} old datasets")
        return cleaned_datasets
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics"""
        
        stats = {
            'layers': {},
            'total_datasets': len(self.catalog),
            'total_size_bytes': 0,
            'total_files': 0
        }
        
        for layer in ["raw", "processed", "curated", "archive"]:
            layer_path = getattr(self, f"{layer}_path")
            
            layer_stats = {
                'datasets': 0,
                'size_bytes': 0,
                'files': 0
            }
            
            if layer_path.exists():
                for dataset_path in layer_path.iterdir():
                    if dataset_path.is_dir():
                        layer_stats['datasets'] += 1
                        
                        for file_path in dataset_path.rglob('*'):
                            if file_path.is_file():
                                layer_stats['files'] += 1
                                layer_stats['size_bytes'] += file_path.stat().st_size
            
            stats['layers'][layer] = layer_stats
            stats['total_size_bytes'] += layer_stats['size_bytes']
            stats['total_files'] += layer_stats['files']
        
        return stats


def main():
    """Demonstrate data lake storage system"""
    logger.info("🏞️  Demonstrating Data Lake Storage System")
    
    # Initialize data lake
    config = DataLakeConfig(
        base_path="data/lake",
        compression="snappy",
        partition_columns=["year", "month"],
        max_file_size_mb=64,
        retention_days=90
    )
    
    data_lake = DataLakeStorage(config)
    
    # Create sample data
    sample_data = []
    for i in range(1000):
        sample_data.append({
            'review_id': i + 1,
            'user_id': (i % 100) + 1,
            'product_id': (i % 50) + 1,
            'rating': (i % 5) + 1,
            'review_text': f'This is review number {i + 1}',
            'review_date': datetime.now() - timedelta(days=i % 365),
            'sentiment_score': 0.5 + (i % 10) * 0.05,
            'created_at': datetime.now() - timedelta(days=i % 365)
        })
    
    df = pd.DataFrame(sample_data)
    
    # Write to different layers
    print("\n📝 Writing datasets to data lake:")
    
    # Raw layer
    data_lake.write_dataset(df, "reviews_raw", "raw", file_format="json")
    print("  • Raw reviews data written")
    
    # Processed layer with partitioning
    data_lake.write_dataset(df, "reviews_processed", "processed", 
                           partition_columns=["year", "month"], file_format="parquet")
    print("  • Processed reviews data written with partitioning")
    
    # Curated layer
    curated_df = df.groupby(['product_id', 'year', 'month']).agg({
        'rating': 'mean',
        'sentiment_score': 'mean',
        'review_id': 'count'
    }).reset_index()
    curated_df.rename(columns={'review_id': 'review_count'}, inplace=True)
    
    data_lake.write_dataset(curated_df, "product_monthly_summary", "curated", 
                           partition_columns=["year"], file_format="parquet")
    print("  • Curated product summary written")
    
    # Read data with filtering
    print("\n🔍 Reading data with filters:")
    
    # Read recent reviews
    recent_reviews = data_lake.read_dataset(
        "reviews_processed", 
        "processed",
        filters=[("rating", ">=", 4)],
        columns=["review_id", "rating", "sentiment_score"],
        limit=10
    )
    print(f"  • Recent high-rated reviews: {len(recent_reviews)} records")
    
    # List all datasets
    print("\n📋 Dataset catalog:")
    datasets = data_lake.list_datasets()
    for dataset in datasets:
        print(f"  • {dataset['name']} ({dataset['layer']}): {dataset.get('row_count', 'N/A')} rows")
    
    # Get storage statistics
    print("\n📊 Storage statistics:")
    stats = data_lake.get_storage_stats()
    print(f"  • Total datasets: {stats['total_datasets']}")
    print(f"  • Total size: {stats['total_size_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  • Total files: {stats['total_files']}")
    
    for layer, layer_stats in stats['layers'].items():
        if layer_stats['datasets'] > 0:
            print(f"  • {layer}: {layer_stats['datasets']} datasets, "
                  f"{layer_stats['size_bytes'] / 1024 / 1024:.2f} MB")
    
    # Demonstrate metadata
    print("\n📄 Dataset metadata:")
    metadata = data_lake.get_dataset_metadata("reviews_processed")
    if metadata:
        print(f"  • Schema: {list(metadata.schema.keys())}")
        print(f"  • Data quality score: {metadata.data_quality_score:.2f}")
        print(f"  • Created: {metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n✅ Data lake storage demonstration completed!")


if __name__ == "__main__":
    main()