"""
Unified Storage Manager
Orchestrates multiple storage systems (relational, NoSQL, data lake)
Provides unified interface for data storage and retrieval across different abstractions
"""

import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import yaml
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from .relational_storage import SQLiteStorage, PostgreSQLStorage, TableSchema
from .nosql_storage import MongoDBStorage, DocumentSchema
from .data_lake_storage import DataLakeStorage, DataLakeConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Configuration for unified storage manager"""
    # Relational storage
    sqlite_path: str = "data/warehouse/analytics.db"
    postgres_config: Dict[str, str] = None
    
    # NoSQL storage
    mongodb_config: Dict[str, str] = None
    
    # Data lake storage
    data_lake_path: str = "data/lake"
    data_lake_compression: str = "snappy"
    data_lake_partition_columns: List[str] = None
    
    # General settings
    enable_sqlite: bool = True
    enable_postgres: bool = False
    enable_mongodb: bool = True
    enable_data_lake: bool = True
    
    def __post_init__(self):
        if self.postgres_config is None:
            self.postgres_config = {
                "host": "localhost",
                "port": "5432",
                "database": "reviews_db",
                "user": "postgres",
                "password": "password"
            }
        
        if self.mongodb_config is None:
            self.mongodb_config = {
                "host": "localhost",
                "port": 27017,
                "database": "reviews_db"
            }
        
        if self.data_lake_partition_columns is None:
            self.data_lake_partition_columns = ["year", "month"]


class StorageManager:
    """Unified storage manager for multiple storage systems"""
    
    def __init__(self, config: StorageConfig):
        self.config = config
        self.storage_systems = {}
        
        # Initialize storage systems based on configuration
        self._initialize_storage_systems()
        
        logger.info("Storage Manager initialized with available systems: " + 
                   ", ".join(self.storage_systems.keys()))
    
    def _initialize_storage_systems(self):
        """Initialize all enabled storage systems"""
        
        # Initialize SQLite storage
        if self.config.enable_sqlite:
            try:
                self.storage_systems['sqlite'] = SQLiteStorage(self.config.sqlite_path)
                logger.info("SQLite storage initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize SQLite storage: {e}")
        
        # Initialize PostgreSQL storage
        if self.config.enable_postgres:
            try:
                self.storage_systems['postgres'] = PostgreSQLStorage(self.config.postgres_config)
                logger.info("PostgreSQL storage initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PostgreSQL storage: {e}")
        
        # Initialize MongoDB storage
        if self.config.enable_mongodb:
            try:
                self.storage_systems['mongodb'] = MongoDBStorage(self.config.mongodb_config)
                logger.info("MongoDB storage initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize MongoDB storage: {e}")
        
        # Initialize Data Lake storage
        if self.config.enable_data_lake:
            try:
                data_lake_config = DataLakeConfig(
                    base_path=self.config.data_lake_path,
                    compression=self.config.data_lake_compression,
                    partition_columns=self.config.data_lake_partition_columns
                )
                self.storage_systems['data_lake'] = DataLakeStorage(data_lake_config)
                logger.info("Data Lake storage initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Data Lake storage: {e}")
    
    def store_data(self, data: Union[pd.DataFrame, List[Dict]], dataset_name: str,
                   storage_type: str = "auto", **kwargs) -> Dict[str, str]:
        """Store data in appropriate storage system(s)"""
        
        results = {}
        
        if storage_type == "auto":
            # Automatically determine best storage based on data characteristics
            storage_types = self._determine_optimal_storage(data, dataset_name)
        else:
            storage_types = [storage_type]
        
        for storage in storage_types:
            try:
                if storage == "relational" and 'sqlite' in self.storage_systems:
                    result = self._store_relational(data, dataset_name, **kwargs)
                    results['sqlite'] = result
                
                elif storage == "nosql" and 'mongodb' in self.storage_systems:
                    result = self._store_nosql(data, dataset_name, **kwargs)
                    results['mongodb'] = result
                
                elif storage == "data_lake" and 'data_lake' in self.storage_systems:
                    result = self._store_data_lake(data, dataset_name, **kwargs)
                    results['data_lake'] = result
                
                elif storage in self.storage_systems:
                    # Direct storage system specification
                    if storage == 'sqlite':
                        result = self._store_relational(data, dataset_name, **kwargs)
                        results[storage] = result
                    elif storage == 'mongodb':
                        result = self._store_nosql(data, dataset_name, **kwargs)
                        results[storage] = result
                    elif storage == 'data_lake':
                        result = self._store_data_lake(data, dataset_name, **kwargs)
                        results[storage] = result
                
            except Exception as e:
                logger.error(f"Failed to store data in {storage}: {e}")
                results[storage] = f"Error: {str(e)}"
        
        return results
    
    def _determine_optimal_storage(self, data: Union[pd.DataFrame, List[Dict]], 
                                  dataset_name: str) -> List[str]:
        """Determine optimal storage systems based on data characteristics"""
        
        if data is None:
            logger.error("Input data is None")
            return ["data_lake"]  # Default fallback
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        storage_types = []
        
        # Data size considerations
        data_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Schema complexity
        has_nested_data = any(df[col].dtype == 'object' and 
                             df[col].apply(lambda x: isinstance(x, (dict, list))).any() 
                             for col in df.columns if col in df.columns)
        
        # Determine storage strategy
        if data_size_mb > 100:  # Large datasets go to data lake
            storage_types.append("data_lake")
        
        if has_nested_data or 'text' in dataset_name.lower():  # Unstructured data to NoSQL
            storage_types.append("nosql")
        
        if data_size_mb < 50 and not has_nested_data:  # Structured data to relational
            storage_types.append("relational")
        
        # Default to data lake if no specific storage determined
        if not storage_types:
            storage_types.append("data_lake")
        
        return storage_types
    
    def _store_relational(self, data: Union[pd.DataFrame, List[Dict]], 
                         dataset_name: str, **kwargs) -> str:
        """Store data in relational storage (SQLite)"""
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data
        
        sqlite_storage = self.storage_systems['sqlite']
        
        # Create table schema
        schema = TableSchema(
            table_name=dataset_name,
            columns={col: str(dtype) for col, dtype in df.dtypes.items()},
            primary_key=kwargs.get('primary_key', []),
            indexes=kwargs.get('indexes', [])
        )
        
        # Create table
        sqlite_storage.create_table(schema)
        
        # Insert data
        records = df.to_dict('records')
        sqlite_storage.insert_data(dataset_name, records)
        
        return f"Stored {len(records)} records in SQLite table '{dataset_name}'"
    
    def _store_nosql(self, data: Union[pd.DataFrame, List[Dict]], 
                    dataset_name: str, **kwargs) -> str:
        """Store data in NoSQL storage (MongoDB)"""
        
        if isinstance(data, pd.DataFrame):
            documents = data.to_dict('records')
        else:
            documents = data
        
        mongodb_storage = self.storage_systems['mongodb']
        
        # Create collection schema
        schema = DocumentSchema(
            collection_name=dataset_name,
            indexes=kwargs.get('indexes', []),
            validation_rules=kwargs.get('validation_rules', {}),
            enable_sharding=kwargs.get('enable_sharding', False),
            ttl_field=kwargs.get('ttl_field'),
            ttl_seconds=kwargs.get('ttl_seconds')
        )
        
        # Initialize collection
        mongodb_storage.initialize_collection(schema)
        
        # Insert documents
        result = mongodb_storage.insert_documents(dataset_name, documents)
        
        return f"Stored {len(documents)} documents in MongoDB collection '{dataset_name}'"
    
    def _store_data_lake(self, data: Union[pd.DataFrame, List[Dict]], 
                        dataset_name: str, **kwargs) -> str:
        """Store data in data lake storage"""
        
        data_lake_storage = self.storage_systems['data_lake']
        
        layer = kwargs.get('layer', 'raw')
        file_format = kwargs.get('file_format', 'parquet')
        partition_columns = kwargs.get('partition_columns', self.config.data_lake_partition_columns)
        
        result_path = data_lake_storage.write_dataset(
            data=data,
            dataset_name=dataset_name,
            layer=layer,
            partition_columns=partition_columns,
            file_format=file_format,
            overwrite=kwargs.get('overwrite', False)
        )
        
        return f"Stored dataset in data lake at '{result_path}'"
    
    def retrieve_data(self, dataset_name: str, storage_type: str = "auto",
                     filters: List[Tuple] = None, columns: List[str] = None,
                     limit: int = None, **kwargs) -> pd.DataFrame:
        """Retrieve data from storage systems"""
        
        if storage_type == "auto":
            # Try to find data in available storage systems
            for storage in ['data_lake', 'mongodb', 'sqlite']:
                if storage in self.storage_systems:
                    try:
                        return self._retrieve_from_storage(storage, dataset_name, 
                                                         filters, columns, limit, **kwargs)
                    except Exception as e:
                        logger.debug(f"Could not retrieve from {storage}: {e}")
                        continue
            
            raise ValueError(f"Dataset '{dataset_name}' not found in any storage system")
        
        else:
            return self._retrieve_from_storage(storage_type, dataset_name, 
                                             filters, columns, limit, **kwargs)
    
    def _retrieve_from_storage(self, storage_type: str, dataset_name: str,
                              filters: List[Tuple] = None, columns: List[str] = None,
                              limit: int = None, **kwargs) -> pd.DataFrame:
        """Retrieve data from specific storage system"""
        
        if storage_type == 'sqlite' and 'sqlite' in self.storage_systems:
            sqlite_storage = self.storage_systems['sqlite']
            
            # Build query
            query = f"SELECT "
            if columns:
                query += ", ".join(columns)
            else:
                query += "*"
            query += f" FROM {dataset_name}"
            
            # Add filters
            if filters:
                where_clauses = []
                for col, op, value in filters:
                    if op == "==":
                        where_clauses.append(f"{col} = '{value}'")
                    elif op == "!=":
                        where_clauses.append(f"{col} != '{value}'")
                    elif op == ">":
                        where_clauses.append(f"{col} > {value}")
                    elif op == ">=":
                        where_clauses.append(f"{col} >= {value}")
                    elif op == "<":
                        where_clauses.append(f"{col} < {value}")
                    elif op == "<=":
                        where_clauses.append(f"{col} <= {value}")
                    elif op == "in":
                        value_str = "', '".join(map(str, value))
                        where_clauses.append(f"{col} IN ('{value_str}')")
                
                if where_clauses:
                    query += " WHERE " + " AND ".join(where_clauses)
            
            # Add limit
            if limit:
                query += f" LIMIT {limit}"
            
            return sqlite_storage.execute_query(query)
        
        elif storage_type == 'mongodb' and 'mongodb' in self.storage_systems:
            mongodb_storage = self.storage_systems['mongodb']
            
            # Build MongoDB query
            mongo_filters = {}
            if filters:
                for col, op, value in filters:
                    if op == "==":
                        mongo_filters[col] = value
                    elif op == "!=":
                        mongo_filters[col] = {"$ne": value}
                    elif op == ">":
                        mongo_filters[col] = {"$gt": value}
                    elif op == ">=":
                        mongo_filters[col] = {"$gte": value}
                    elif op == "<":
                        mongo_filters[col] = {"$lt": value}
                    elif op == "<=":
                        mongo_filters[col] = {"$lte": value}
                    elif op == "in":
                        mongo_filters[col] = {"$in": value}
            
            documents = mongodb_storage.find_documents(dataset_name, mongo_filters, limit)
            return pd.DataFrame(list(documents))
        
        elif storage_type == 'data_lake' and 'data_lake' in self.storage_systems:
            data_lake_storage = self.storage_systems['data_lake']
            
            layer = kwargs.get('layer', 'processed')
            return data_lake_storage.read_dataset(dataset_name, layer, filters, columns, limit)
        
        else:
            raise ValueError(f"Storage type '{storage_type}' not available")
    
    def list_datasets(self, storage_type: str = None) -> Dict[str, List[Dict]]:
        """List datasets across all or specific storage systems"""
        
        datasets = {}
        
        if storage_type is None:
            storage_types = list(self.storage_systems.keys())
        else:
            storage_types = [storage_type] if storage_type in self.storage_systems else []
        
        for storage in storage_types:
            try:
                if storage == 'sqlite':
                    sqlite_storage = self.storage_systems['sqlite']
                    tables = sqlite_storage.list_tables()
                    datasets[storage] = [{'name': table, 'type': 'table'} for table in tables]
                
                elif storage == 'mongodb':
                    mongodb_storage = self.storage_systems['mongodb']
                    collections = mongodb_storage.list_collections()
                    datasets[storage] = [{'name': coll, 'type': 'collection'} for coll in collections]
                
                elif storage == 'data_lake':
                    data_lake_storage = self.storage_systems['data_lake']
                    lake_datasets = data_lake_storage.list_datasets()
                    datasets[storage] = lake_datasets
                
            except Exception as e:
                logger.warning(f"Could not list datasets from {storage}: {e}")
                datasets[storage] = []
        
        return datasets
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive storage statistics across all systems"""
        
        stats = {
            'systems': {},
            'total_datasets': 0,
            'total_size_bytes': 0
        }
        
        for storage_name, storage_system in self.storage_systems.items():
            try:
                if storage_name == 'sqlite':
                    sqlite_stats = storage_system.get_database_stats()
                    stats['systems'][storage_name] = {
                        'tables': len(sqlite_stats.get('tables', [])),
                        'total_rows': sum(table.get('row_count', 0) 
                                        for table in sqlite_stats.get('tables', [])),
                        'size_bytes': sqlite_stats.get('size_bytes', 0)
                    }
                
                elif storage_name == 'mongodb':
                    collections = storage_system.list_collections()
                    total_docs = 0
                    total_size = 0
                    
                    for collection in collections:
                        coll_stats = storage_system.get_collection_stats(collection)
                        total_docs += coll_stats.get('document_count', 0)
                        total_size += coll_stats.get('size_bytes', 0)
                    
                    stats['systems'][storage_name] = {
                        'collections': len(collections),
                        'total_documents': total_docs,
                        'size_bytes': total_size
                    }
                
                elif storage_name == 'data_lake':
                    lake_stats = storage_system.get_storage_stats()
                    stats['systems'][storage_name] = lake_stats
                
                # Add to totals
                stats['total_datasets'] += stats['systems'][storage_name].get('tables', 0) + \
                                         stats['systems'][storage_name].get('collections', 0) + \
                                         stats['systems'][storage_name].get('total_datasets', 0)
                
                stats['total_size_bytes'] += stats['systems'][storage_name].get('size_bytes', 0)
                
            except Exception as e:
                logger.warning(f"Could not get statistics from {storage_name}: {e}")
                stats['systems'][storage_name] = {'error': str(e)}
        
        return stats
    
    def migrate_data(self, dataset_name: str, source_storage: str, 
                    target_storage: str, **kwargs) -> str:
        """Migrate data between storage systems"""
        
        try:
            # Retrieve data from source
            data = self.retrieve_data(dataset_name, source_storage, **kwargs)
            
            # Store in target
            result = self.store_data(data, dataset_name, target_storage, **kwargs)
            
            logger.info(f"Migrated dataset '{dataset_name}' from {source_storage} to {target_storage}")
            return f"Migration successful: {result.get(target_storage, 'Unknown result')}"
            
        except Exception as e:
            logger.error(f"Failed to migrate dataset '{dataset_name}': {e}")
            raise
    
    def backup_data(self, dataset_name: str, storage_type: str = "auto") -> str:
        """Backup data to data lake archive"""
        
        try:
            # Retrieve data
            data = self.retrieve_data(dataset_name, storage_type)
            
            # Store in data lake archive
            if 'data_lake' in self.storage_systems:
                data_lake_storage = self.storage_systems['data_lake']
                
                # Create backup with timestamp
                backup_name = f"{dataset_name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                result_path = data_lake_storage.write_dataset(
                    data=data,
                    dataset_name=backup_name,
                    layer="archive",
                    file_format="parquet",
                    compression="gzip"
                )
                
                logger.info(f"Backup created for dataset '{dataset_name}' at {result_path}")
                return f"Backup successful: {result_path}"
            
            else:
                raise ValueError("Data lake storage not available for backup")
                
        except Exception as e:
            logger.error(f"Failed to backup dataset '{dataset_name}': {e}")
            raise
    
    def cleanup_storage(self, retention_days: int = 30):
        """Cleanup old data across all storage systems"""
        
        cleanup_results = {}
        
        # Cleanup data lake
        if 'data_lake' in self.storage_systems:
            try:
                data_lake_storage = self.storage_systems['data_lake']
                cleaned_datasets = data_lake_storage.cleanup_old_data(retention_days)
                cleanup_results['data_lake'] = f"Cleaned {len(cleaned_datasets)} datasets"
            except Exception as e:
                cleanup_results['data_lake'] = f"Error: {str(e)}"
        
        # Additional cleanup logic for other storage systems can be added here
        
        logger.info(f"Storage cleanup completed: {cleanup_results}")
        return cleanup_results
    
    def health_check(self) -> Dict[str, str]:
        """Check health of all storage systems"""
        
        health_status = {}
        
        for storage_name, storage_system in self.storage_systems.items():
            try:
                if storage_name == 'sqlite':
                    # Test SQLite connection
                    storage_system.execute_query("SELECT 1")
                    health_status[storage_name] = "Healthy"
                
                elif storage_name == 'mongodb':
                    # Test MongoDB connection
                    storage_system.list_collections()
                    health_status[storage_name] = "Healthy"
                
                elif storage_name == 'data_lake':
                    # Test data lake access
                    storage_system.list_datasets()
                    health_status[storage_name] = "Healthy"
                
            except Exception as e:
                health_status[storage_name] = f"Unhealthy: {str(e)}"
        
        return health_status


def main():
    """Demonstrate unified storage manager"""
    logger.info("🗄️  Demonstrating Unified Storage Manager")
    
    # Initialize storage manager
    config = StorageConfig(
        enable_sqlite=True,
        enable_mongodb=True,
        enable_data_lake=True
    )
    
    storage_manager = StorageManager(config)
    
    # Create sample data
    sample_reviews = []
    for i in range(100):
        sample_reviews.append({
            'review_id': i + 1,
            'user_id': (i % 20) + 1,
            'product_id': (i % 10) + 1,
            'rating': (i % 5) + 1,
            'review_text': f'This is review number {i + 1}',
            'sentiment_score': 0.5 + (i % 10) * 0.05,
            'created_at': datetime.now() - timedelta(days=i % 30),
            'metadata': {'helpful_votes': i % 10, 'verified_purchase': i % 2 == 0}
        })
    
    df = pd.DataFrame(sample_reviews)
    
    # Store data using automatic storage selection
    print("\n💾 Storing data with automatic storage selection:")
    results = storage_manager.store_data(df, "product_reviews", storage_type="auto")
    for storage, result in results.items():
        print(f"  • {storage}: {result}")
    
    # Store data in specific storage systems
    print("\n🎯 Storing data in specific storage systems:")
    
    # Store in data lake with partitioning
    lake_result = storage_manager.store_data(
        df, "reviews_partitioned", "data_lake",
        layer="processed", partition_columns=["year", "month"]
    )
    print(f"  • Data Lake: {lake_result.get('data_lake', 'Failed')}")
    
    # Store aggregated data in relational storage
    agg_data = df.groupby('product_id').agg({
        'rating': 'mean',
        'sentiment_score': 'mean',
        'review_id': 'count'
    }).reset_index()
    agg_data.rename(columns={'review_id': 'review_count'}, inplace=True)
    
    relational_result = storage_manager.store_data(
        agg_data, "product_summary", "relational",
        primary_key=['product_id']
    )
    print(f"  • SQLite: {relational_result.get('sqlite', 'Failed')}")
    
    # Retrieve data
    print("\n🔍 Retrieving data:")
    
    # Retrieve with filters
    high_rated = storage_manager.retrieve_data(
        "product_reviews", "auto",
        filters=[("rating", ">=", 4)],
        limit=5
    )
    print(f"  • High-rated reviews: {len(high_rated)} records")
    
    # List all datasets
    print("\n📋 Dataset inventory:")
    datasets = storage_manager.list_datasets()
    for storage, dataset_list in datasets.items():
        print(f"  • {storage}: {len(dataset_list)} datasets")
        for dataset in dataset_list[:3]:  # Show first 3
            print(f"    - {dataset['name']}")
    
    # Get storage statistics
    print("\n📊 Storage statistics:")
    stats = storage_manager.get_storage_statistics()
    print(f"  • Total datasets: {stats['total_datasets']}")
    print(f"  • Total size: {stats['total_size_bytes'] / 1024 / 1024:.2f} MB")
    
    for storage, storage_stats in stats['systems'].items():
        if 'error' not in storage_stats:
            print(f"  • {storage}: {storage_stats}")
    
    # Health check
    print("\n🏥 Storage health check:")
    health = storage_manager.health_check()
    for storage, status in health.items():
        print(f"  • {storage}: {status}")
    
    # Demonstrate data migration
    print("\n🚚 Data migration example:")
    try:
        migration_result = storage_manager.migrate_data(
            "product_summary", "sqlite", "data_lake",
            layer="curated"
        )
        print(f"  • Migration: {migration_result}")
    except Exception as e:
        print(f"  • Migration failed: {e}")
    
    # Backup data
    print("\n💾 Data backup example:")
    try:
        backup_result = storage_manager.backup_data("product_reviews")
        print(f"  • Backup: {backup_result}")
    except Exception as e:
        print(f"  • Backup failed: {e}")
    
    print("\n✅ Unified storage manager demonstration completed!")


if __name__ == "__main__":
    main()