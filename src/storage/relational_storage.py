"""
Relational Database Storage System
Implements structured data storage using PostgreSQL and SQLite
Provides ACID properties, schema enforcement, and SQL querying capabilities
"""

import sqlite3
import psycopg2
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TableSchema:
    """Defines table schema for relational databases"""
    name: str
    columns: Dict[str, str]  # column_name: data_type
    primary_key: List[str]
    foreign_keys: Dict[str, str] = None  # column: referenced_table.column
    indexes: List[str] = None
    constraints: List[str] = None
    
    def __post_init__(self):
        if self.foreign_keys is None:
            self.foreign_keys = {}
        if self.indexes is None:
            self.indexes = []
        if self.constraints is None:
            self.constraints = []


class SQLiteStorage:
    """SQLite storage implementation for local development and testing"""
    
    def __init__(self, db_path: str = "data/warehouse/analytics.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        self.schemas = self._define_schemas()
        
        # Initialize database
        self._initialize_database()
    
    def _define_schemas(self) -> Dict[str, TableSchema]:
        """Define table schemas for the analytics database"""
        return {
            'users': TableSchema(
                name='users',
                columns={
                    'user_id': 'INTEGER PRIMARY KEY',
                    'username': 'TEXT NOT NULL',
                    'email': 'TEXT UNIQUE NOT NULL',
                    'registration_date': 'TIMESTAMP',
                    'location': 'TEXT',
                    'age_group': 'TEXT',
                    'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    'updated_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
                },
                primary_key=['user_id'],
                indexes=['email', 'username', 'registration_date']
            ),
            
            'products': TableSchema(
                name='products',
                columns={
                    'product_id': 'INTEGER PRIMARY KEY',
                    'name': 'TEXT NOT NULL',
                    'category': 'TEXT NOT NULL',
                    'subcategory': 'TEXT',
                    'brand': 'TEXT',
                    'price': 'DECIMAL(10,2)',
                    'description': 'TEXT',
                    'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    'updated_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
                },
                primary_key=['product_id'],
                indexes=['category', 'brand', 'price']
            ),
            
            'reviews': TableSchema(
                name='reviews',
                columns={
                    'review_id': 'INTEGER PRIMARY KEY',
                    'user_id': 'INTEGER NOT NULL',
                    'product_id': 'INTEGER NOT NULL',
                    'rating': 'INTEGER CHECK (rating >= 1 AND rating <= 5)',
                    'review_text': 'TEXT',
                    'review_date': 'TIMESTAMP NOT NULL',
                    'verified_purchase': 'BOOLEAN DEFAULT FALSE',
                    'helpful_votes': 'INTEGER DEFAULT 0',
                    'sentiment_score': 'DECIMAL(3,2)',
                    'sentiment_label': 'TEXT',
                    'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
                },
                primary_key=['review_id'],
                foreign_keys={
                    'user_id': 'users.user_id',
                    'product_id': 'products.product_id'
                },
                indexes=['user_id', 'product_id', 'rating', 'review_date', 'sentiment_label']
            ),
            
            'review_analytics': TableSchema(
                name='review_analytics',
                columns={
                    'analytics_id': 'INTEGER PRIMARY KEY',
                    'product_id': 'INTEGER NOT NULL',
                    'total_reviews': 'INTEGER DEFAULT 0',
                    'average_rating': 'DECIMAL(3,2)',
                    'sentiment_positive': 'INTEGER DEFAULT 0',
                    'sentiment_negative': 'INTEGER DEFAULT 0',
                    'sentiment_neutral': 'INTEGER DEFAULT 0',
                    'last_updated': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
                },
                primary_key=['analytics_id'],
                foreign_keys={'product_id': 'products.product_id'},
                indexes=['product_id', 'average_rating', 'last_updated']
            ),
            
            'user_activity': TableSchema(
                name='user_activity',
                columns={
                    'activity_id': 'INTEGER PRIMARY KEY',
                    'user_id': 'INTEGER NOT NULL',
                    'activity_type': 'TEXT NOT NULL',
                    'product_id': 'INTEGER',
                    'activity_timestamp': 'TIMESTAMP NOT NULL',
                    'session_id': 'TEXT',
                    'metadata': 'TEXT'  # JSON string
                },
                primary_key=['activity_id'],
                foreign_keys={
                    'user_id': 'users.user_id',
                    'product_id': 'products.product_id'
                },
                indexes=['user_id', 'activity_type', 'activity_timestamp', 'session_id']
            )
        }
    
    def _initialize_database(self):
        """Initialize database with tables and indexes"""
        try:
            self.connection = sqlite3.connect(str(self.db_path))
            self.connection.row_factory = sqlite3.Row  # Enable column access by name
            
            # Create tables
            for schema in self.schemas.values():
                self._create_table(schema)
            
            # Create indexes
            for schema in self.schemas.values():
                self._create_indexes(schema)
            
            logger.info(f"SQLite database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SQLite database: {e}")
            raise
    
    def _create_table(self, schema: TableSchema):
        """Create table from schema"""
        columns_sql = []
        
        for col_name, col_type in schema.columns.items():
            columns_sql.append(f"{col_name} {col_type}")
        
        # Add foreign key constraints
        for fk_col, fk_ref in schema.foreign_keys.items():
            ref_table, ref_col = fk_ref.split('.')
            columns_sql.append(f"FOREIGN KEY ({fk_col}) REFERENCES {ref_table}({ref_col})")
        
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {schema.name} (
            {', '.join(columns_sql)}
        )
        """
        
        self.connection.execute(create_sql)
        self.connection.commit()
        logger.info(f"Table '{schema.name}' created/verified")
    
    def _create_indexes(self, schema: TableSchema):
        """Create indexes for table"""
        for index_col in schema.indexes:
            index_name = f"idx_{schema.name}_{index_col}"
            index_sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {schema.name}({index_col})"
            
            try:
                self.connection.execute(index_sql)
                self.connection.commit()
            except Exception as e:
                logger.warning(f"Could not create index {index_name}: {e}")
    
    def insert_data(self, table_name: str, data: List[Dict], batch_size: int = 1000) -> int:
        """Insert data into table with batch processing"""
        if not data:
            return 0
        
        if table_name not in self.schemas:
            raise ValueError(f"Unknown table: {table_name}")
        
        schema = self.schemas[table_name]
        columns = list(data[0].keys())
        
        # Validate columns exist in schema
        schema_columns = set(schema.columns.keys())
        for col in columns:
            if col not in schema_columns:
                logger.warning(f"Column '{col}' not in schema for table '{table_name}'")
        
        # Prepare insert statement
        placeholders = ', '.join(['?' for _ in columns])
        insert_sql = f"INSERT OR REPLACE INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
        
        inserted_count = 0
        
        try:
            # Process in batches
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                batch_values = []
                
                for row in batch:
                    values = []
                    for col in columns:
                        value = row.get(col)
                        # Handle JSON serialization for metadata columns
                        if col == 'metadata' and isinstance(value, (dict, list)):
                            value = json.dumps(value)
                        values.append(value)
                    batch_values.append(values)
                
                self.connection.executemany(insert_sql, batch_values)
                inserted_count += len(batch)
            
            self.connection.commit()
            logger.info(f"Inserted {inserted_count} records into {table_name}")
            return inserted_count
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to insert data into {table_name}: {e}")
            raise
    
    def query_data(self, sql: str, params: Tuple = None) -> List[Dict]:
        """Execute SQL query and return results"""
        try:
            cursor = self.connection.cursor()
            
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            
            # Convert rows to dictionaries
            columns = [description[0] for description in cursor.description]
            results = []
            
            for row in cursor.fetchall():
                row_dict = {}
                for i, value in enumerate(row):
                    col_name = columns[i]
                    # Parse JSON for metadata columns
                    if col_name == 'metadata' and isinstance(value, str):
                        try:
                            value = json.loads(value)
                        except:
                            pass
                    row_dict[col_name] = value
                results.append(row_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
    
    def get_table_stats(self, table_name: str) -> Dict:
        """Get statistics for a table"""
        if table_name not in self.schemas:
            raise ValueError(f"Unknown table: {table_name}")
        
        stats = {}
        
        # Row count
        count_result = self.query_data(f"SELECT COUNT(*) as count FROM {table_name}")
        stats['row_count'] = count_result[0]['count']
        
        # Table size (approximate)
        size_result = self.query_data(f"SELECT page_count * page_size as size FROM pragma_page_count('{table_name}'), pragma_page_size")
        if size_result:
            stats['size_bytes'] = size_result[0]['size']
        
        # Column statistics for numeric columns
        schema = self.schemas[table_name]
        numeric_columns = []
        
        for col_name, col_type in schema.columns.items():
            if any(t in col_type.upper() for t in ['INTEGER', 'DECIMAL', 'REAL', 'NUMERIC']):
                numeric_columns.append(col_name)
        
        if numeric_columns:
            stats['column_stats'] = {}
            for col in numeric_columns:
                try:
                    col_stats = self.query_data(f"""
                        SELECT 
                            MIN({col}) as min_val,
                            MAX({col}) as max_val,
                            AVG({col}) as avg_val,
                            COUNT(DISTINCT {col}) as distinct_count
                        FROM {table_name}
                        WHERE {col} IS NOT NULL
                    """)
                    if col_stats and col_stats[0]['min_val'] is not None:
                        stats['column_stats'][col] = col_stats[0]
                except Exception as e:
                    logger.warning(f"Could not get stats for column {col}: {e}")
        
        return stats
    
    def create_analytics_views(self):
        """Create analytical views for common queries"""
        views = {
            'product_review_summary': """
                CREATE VIEW IF NOT EXISTS product_review_summary AS
                SELECT 
                    p.product_id,
                    p.name as product_name,
                    p.category,
                    p.brand,
                    COUNT(r.review_id) as total_reviews,
                    AVG(r.rating) as average_rating,
                    SUM(CASE WHEN r.sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_reviews,
                    SUM(CASE WHEN r.sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_reviews,
                    SUM(CASE WHEN r.sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_reviews,
                    MAX(r.review_date) as latest_review_date
                FROM products p
                LEFT JOIN reviews r ON p.product_id = r.product_id
                GROUP BY p.product_id, p.name, p.category, p.brand
            """,
            
            'user_engagement_summary': """
                CREATE VIEW IF NOT EXISTS user_engagement_summary AS
                SELECT 
                    u.user_id,
                    u.username,
                    u.location,
                    COUNT(DISTINCT r.review_id) as total_reviews,
                    AVG(r.rating) as average_rating_given,
                    COUNT(DISTINCT ua.activity_id) as total_activities,
                    MAX(ua.activity_timestamp) as last_activity_date,
                    COUNT(DISTINCT r.product_id) as products_reviewed
                FROM users u
                LEFT JOIN reviews r ON u.user_id = r.user_id
                LEFT JOIN user_activity ua ON u.user_id = ua.user_id
                GROUP BY u.user_id, u.username, u.location
            """,
            
            'daily_review_trends': """
                CREATE VIEW IF NOT EXISTS daily_review_trends AS
                SELECT 
                    DATE(review_date) as review_date,
                    COUNT(*) as total_reviews,
                    AVG(rating) as average_rating,
                    SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
                    SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
                    SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count
                FROM reviews
                GROUP BY DATE(review_date)
                ORDER BY review_date DESC
            """
        }
        
        for view_name, view_sql in views.items():
            try:
                self.connection.execute(view_sql)
                self.connection.commit()
                logger.info(f"Created view: {view_name}")
            except Exception as e:
                logger.warning(f"Could not create view {view_name}: {e}")
    
    def export_to_dataframe(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Export table data to pandas DataFrame"""
        sql = f"SELECT * FROM {table_name}"
        if limit:
            sql += f" LIMIT {limit}"
        
        return pd.read_sql_query(sql, self.connection)
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("SQLite connection closed")


class PostgreSQLStorage:
    """PostgreSQL storage implementation for production environments"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.connection = None
        self.schemas = self._define_schemas()
        
        # Initialize connection
        self._connect()
        self._initialize_database()
    
    def _define_schemas(self) -> Dict[str, TableSchema]:
        """Define PostgreSQL table schemas with advanced features"""
        return {
            'users': TableSchema(
                name='users',
                columns={
                    'user_id': 'SERIAL PRIMARY KEY',
                    'username': 'VARCHAR(100) NOT NULL',
                    'email': 'VARCHAR(255) UNIQUE NOT NULL',
                    'registration_date': 'TIMESTAMP',
                    'location': 'VARCHAR(100)',
                    'age_group': 'VARCHAR(20)',
                    'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    'updated_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
                },
                primary_key=['user_id'],
                indexes=['email', 'username', 'registration_date', 'location']
            ),
            
            'products': TableSchema(
                name='products',
                columns={
                    'product_id': 'SERIAL PRIMARY KEY',
                    'name': 'VARCHAR(500) NOT NULL',
                    'category': 'VARCHAR(100) NOT NULL',
                    'subcategory': 'VARCHAR(100)',
                    'brand': 'VARCHAR(100)',
                    'price': 'DECIMAL(10,2)',
                    'description': 'TEXT',
                    'metadata': 'JSONB',  # PostgreSQL JSON support
                    'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
                    'updated_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
                },
                primary_key=['product_id'],
                indexes=['category', 'brand', 'price', 'metadata']
            ),
            
            'reviews': TableSchema(
                name='reviews',
                columns={
                    'review_id': 'SERIAL PRIMARY KEY',
                    'user_id': 'INTEGER NOT NULL',
                    'product_id': 'INTEGER NOT NULL',
                    'rating': 'INTEGER CHECK (rating >= 1 AND rating <= 5)',
                    'review_text': 'TEXT',
                    'review_date': 'TIMESTAMP NOT NULL',
                    'verified_purchase': 'BOOLEAN DEFAULT FALSE',
                    'helpful_votes': 'INTEGER DEFAULT 0',
                    'sentiment_score': 'DECIMAL(5,4)',
                    'sentiment_label': 'VARCHAR(20)',
                    'language': 'VARCHAR(10) DEFAULT \'en\'',
                    'metadata': 'JSONB',
                    'created_at': 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'
                },
                primary_key=['review_id'],
                foreign_keys={
                    'user_id': 'users.user_id',
                    'product_id': 'products.product_id'
                },
                indexes=['user_id', 'product_id', 'rating', 'review_date', 'sentiment_label', 'metadata']
            )
        }
    
    def _connect(self):
        """Establish PostgreSQL connection"""
        try:
            self.connection = psycopg2.connect(
                host=self.config.get('host', 'localhost'),
                port=self.config.get('port', 5432),
                database=self.config.get('database', 'reviews_analytics'),
                user=self.config.get('user', 'postgres'),
                password=self.config.get('password', '')
            )
            self.connection.autocommit = False
            logger.info("PostgreSQL connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def _initialize_database(self):
        """Initialize PostgreSQL database with tables and indexes"""
        try:
            cursor = self.connection.cursor()
            
            # Create tables
            for schema in self.schemas.values():
                self._create_table_postgresql(cursor, schema)
            
            # Create indexes
            for schema in self.schemas.values():
                self._create_indexes_postgresql(cursor, schema)
            
            self.connection.commit()
            logger.info("PostgreSQL database initialized")
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Failed to initialize PostgreSQL database: {e}")
            raise
    
    def _create_table_postgresql(self, cursor, schema: TableSchema):
        """Create PostgreSQL table from schema"""
        columns_sql = []
        
        for col_name, col_type in schema.columns.items():
            columns_sql.append(f"{col_name} {col_type}")
        
        # Add foreign key constraints
        for fk_col, fk_ref in schema.foreign_keys.items():
            ref_table, ref_col = fk_ref.split('.')
            columns_sql.append(f"FOREIGN KEY ({fk_col}) REFERENCES {ref_table}({ref_col}) ON DELETE CASCADE")
        
        create_sql = f"""
        CREATE TABLE IF NOT EXISTS {schema.name} (
            {', '.join(columns_sql)}
        )
        """
        
        cursor.execute(create_sql)
        logger.info(f"PostgreSQL table '{schema.name}' created/verified")
    
    def _create_indexes_postgresql(self, cursor, schema: TableSchema):
        """Create PostgreSQL indexes"""
        for index_col in schema.indexes:
            index_name = f"idx_{schema.name}_{index_col}"
            
            # Special handling for JSONB columns
            if index_col == 'metadata':
                index_sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {schema.name} USING GIN ({index_col})"
            else:
                index_sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {schema.name}({index_col})"
            
            try:
                cursor.execute(index_sql)
            except Exception as e:
                logger.warning(f"Could not create PostgreSQL index {index_name}: {e}")
    
    def close(self):
        """Close PostgreSQL connection"""
        if self.connection:
            self.connection.close()
            logger.info("PostgreSQL connection closed")


def main():
    """Demonstrate relational storage systems"""
    logger.info("🗄️  Demonstrating Relational Storage Systems")
    
    # Initialize SQLite storage
    sqlite_storage = SQLiteStorage()
    
    # Create sample data
    sample_users = [
        {
            'user_id': 1,
            'username': 'john_doe',
            'email': 'john@example.com',
            'registration_date': '2024-01-15',
            'location': 'New York',
            'age_group': '25-34'
        },
        {
            'user_id': 2,
            'username': 'jane_smith',
            'email': 'jane@example.com',
            'registration_date': '2024-02-20',
            'location': 'California',
            'age_group': '35-44'
        }
    ]
    
    sample_products = [
        {
            'product_id': 1,
            'name': 'Wireless Headphones',
            'category': 'Electronics',
            'subcategory': 'Audio',
            'brand': 'TechBrand',
            'price': 99.99,
            'description': 'High-quality wireless headphones with noise cancellation'
        },
        {
            'product_id': 2,
            'name': 'Running Shoes',
            'category': 'Sports',
            'subcategory': 'Footwear',
            'brand': 'SportsBrand',
            'price': 129.99,
            'description': 'Comfortable running shoes for daily training'
        }
    ]
    
    sample_reviews = [
        {
            'review_id': 1,
            'user_id': 1,
            'product_id': 1,
            'rating': 5,
            'review_text': 'Excellent headphones! Great sound quality and comfortable to wear.',
            'review_date': '2024-03-01',
            'verified_purchase': True,
            'helpful_votes': 15,
            'sentiment_score': 0.85,
            'sentiment_label': 'positive'
        },
        {
            'review_id': 2,
            'user_id': 2,
            'product_id': 2,
            'rating': 4,
            'review_text': 'Good shoes, but could be more durable.',
            'review_date': '2024-03-02',
            'verified_purchase': True,
            'helpful_votes': 8,
            'sentiment_score': 0.65,
            'sentiment_label': 'positive'
        }
    ]
    
    # Insert data
    sqlite_storage.insert_data('users', sample_users)
    sqlite_storage.insert_data('products', sample_products)
    sqlite_storage.insert_data('reviews', sample_reviews)
    
    # Create analytical views
    sqlite_storage.create_analytics_views()
    
    # Query data
    print("\n📊 Product Review Summary:")
    summary = sqlite_storage.query_data("SELECT * FROM product_review_summary")
    for row in summary:
        print(f"  • {row['product_name']}: {row['total_reviews']} reviews, avg rating: {row['average_rating']:.2f}")
    
    # Get table statistics
    print("\n📈 Table Statistics:")
    for table_name in ['users', 'products', 'reviews']:
        stats = sqlite_storage.get_table_stats(table_name)
        print(f"  • {table_name}: {stats['row_count']} rows")
    
    # Export to DataFrame
    reviews_df = sqlite_storage.export_to_dataframe('reviews')
    print(f"\n📋 Reviews DataFrame shape: {reviews_df.shape}")
    
    sqlite_storage.close()
    
    print("\n✅ Relational storage demonstration completed!")


if __name__ == "__main__":
    main()