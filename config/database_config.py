"""
Database Configuration for Product Review Analysis Platform
Manages connections to different storage systems following data engineering best practices
"""

import os
from dataclasses import dataclass
from typing import Dict, Any
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@dataclass
class DatabaseConfig:
    """Configuration class for database connections"""
    
    # PostgreSQL Configuration (Data Warehouse)
    POSTGRES_CONFIG = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DB', 'review_analytics'),
        'username': os.getenv('POSTGRES_USER', 'data_engineer'),
        'password': os.getenv('POSTGRES_PASSWORD', 'secure_password'),
    }
    
    # MongoDB Configuration (Semi-structured data)
    MONGODB_CONFIG = {
        'host': os.getenv('MONGODB_HOST', 'localhost'),
        'port': os.getenv('MONGODB_PORT', '27017'),
        'database': os.getenv('MONGODB_DB', 'review_data'),
        'username': os.getenv('MONGODB_USER', ''),
        'password': os.getenv('MONGODB_PASSWORD', ''),
    }
    
    # SQLite Configuration (Local development)
    SQLITE_CONFIG = {
        'database_path': os.getenv('SQLITE_PATH', 'data/warehouse/reviews.db'),
    }
    
    # Data Lake Configuration (File-based storage)
    DATA_LAKE_CONFIG = {
        'base_path': os.getenv('DATA_LAKE_PATH', 'data/'),
        'raw_data_path': 'raw/',
        'processed_data_path': 'processed/',
        'warehouse_path': 'warehouse/',
    }


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self):
        self.config = DatabaseConfig()
        self._postgres_engine = None
        self._sqlite_engine = None
        
    def get_postgres_engine(self):
        """Get PostgreSQL database engine"""
        if self._postgres_engine is None:
            config = self.config.POSTGRES_CONFIG
            connection_string = (
                f"postgresql://{config['username']}:{config['password']}"
                f"@{config['host']}:{config['port']}/{config['database']}"
            )
            self._postgres_engine = create_engine(connection_string)
        return self._postgres_engine
    
    def get_sqlite_engine(self):
        """Get SQLite database engine for local development"""
        if self._sqlite_engine is None:
            db_path = self.config.SQLITE_CONFIG['database_path']
            # Ensure directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            connection_string = f"sqlite:///{db_path}"
            self._sqlite_engine = create_engine(connection_string)
        return self._sqlite_engine
    
    def get_mongodb_connection_string(self):
        """Get MongoDB connection string"""
        config = self.config.MONGODB_CONFIG
        if config['username'] and config['password']:
            return (
                f"mongodb://{config['username']}:{config['password']}"
                f"@{config['host']}:{config['port']}/{config['database']}"
            )
        else:
            return f"mongodb://{config['host']}:{config['port']}/{config['database']}"
    
    def get_data_lake_paths(self):
        """Get data lake directory paths"""
        base_path = self.config.DATA_LAKE_CONFIG['base_path']
        return {
            'raw': os.path.join(base_path, self.config.DATA_LAKE_CONFIG['raw_data_path']),
            'processed': os.path.join(base_path, self.config.DATA_LAKE_CONFIG['processed_data_path']),
            'warehouse': os.path.join(base_path, self.config.DATA_LAKE_CONFIG['warehouse_path']),
        }
    
    def create_session(self, engine_type='sqlite'):
        """Create database session"""
        if engine_type == 'postgres':
            engine = self.get_postgres_engine()
        else:
            engine = self.get_sqlite_engine()
        
        Session = sessionmaker(bind=engine)
        return Session()
    
    def test_connections(self):
        """Test all database connections"""
        results = {}
        
        # Test SQLite connection
        try:
            engine = self.get_sqlite_engine()
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            results['sqlite'] = 'Connected'
        except Exception as e:
            results['sqlite'] = f'Error: {str(e)}'
        
        # Test PostgreSQL connection (if available)
        try:
            engine = self.get_postgres_engine()
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            results['postgres'] = 'Connected'
        except Exception as e:
            results['postgres'] = f'Error: {str(e)}'
        
        # Test MongoDB connection (if available)
        try:
            from pymongo import MongoClient
            client = MongoClient(self.get_mongodb_connection_string())
            client.server_info()
            results['mongodb'] = 'Connected'
        except Exception as e:
            results['mongodb'] = f'Error: {str(e)}'
        
        return results


# Global database manager instance
db_manager = DatabaseManager()


def get_database_manager():
    """Get the global database manager instance"""
    return db_manager


if __name__ == "__main__":
    # Test database connections
    manager = DatabaseManager()
    results = manager.test_connections()
    
    print("Database Connection Test Results:")
    for db_type, status in results.items():
        print(f"  {db_type.upper()}: {status}")