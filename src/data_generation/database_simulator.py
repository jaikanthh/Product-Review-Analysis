"""
Database Simulator for Product Review Analysis
Simulates traditional database systems as data sources (OLTP)
Demonstrates database-based data generation and extraction patterns
"""

import sqlite3
import psycopg2
import pymongo
import pandas as pd
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
from contextlib import contextmanager
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SQLiteSimulator:
    """Simulates SQLite database as a source system (OLTP)"""
    
    def __init__(self, db_path: str = "data/raw/databases/ecommerce.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._setup_database()
    
    def _setup_database(self):
        """Setup SQLite database with tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    first_name TEXT,
                    last_name TEXT,
                    registration_date TEXT,
                    last_login TEXT,
                    is_verified BOOLEAN DEFAULT 0,
                    total_orders INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Products table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS products (
                    product_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    subcategory TEXT,
                    brand TEXT,
                    price REAL NOT NULL,
                    description TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    stock_quantity INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Orders table (transactional data)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    order_date TEXT NOT NULL,
                    total_amount REAL NOT NULL,
                    status TEXT DEFAULT 'pending',
                    shipping_address TEXT,
                    payment_method TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            # Order items table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS order_items (
                    order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_id TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    unit_price REAL NOT NULL,
                    total_price REAL NOT NULL,
                    FOREIGN KEY (order_id) REFERENCES orders (order_id),
                    FOREIGN KEY (product_id) REFERENCES products (product_id)
                )
            """)
            
            # Reviews table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    review_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    order_id TEXT,
                    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
                    title TEXT,
                    review_text TEXT,
                    verified_purchase BOOLEAN DEFAULT 0,
                    helpful_votes INTEGER DEFAULT 0,
                    total_votes INTEGER DEFAULT 0,
                    review_date TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    FOREIGN KEY (product_id) REFERENCES products (product_id),
                    FOREIGN KEY (order_id) REFERENCES orders (order_id)
                )
            """)
            
            conn.commit()
            logger.info("SQLite database setup completed")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def populate_sample_data(self, num_users: int = 1000, num_products: int = 500, num_orders: int = 2000):
        """Populate database with sample data"""
        logger.info("Populating SQLite database with sample data...")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Generate users
            users_data = []
            for i in range(num_users):
                user_id = f"USER_{i+1:06d}"
                username = f"user_{i+1}"
                email = f"user{i+1}@example.com"
                first_name = random.choice(['John', 'Jane', 'Mike', 'Sarah', 'David', 'Lisa'])
                last_name = random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'])
                reg_date = (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat()
                last_login = (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
                is_verified = random.choice([0, 1])
                total_orders = random.randint(0, 50)
                
                users_data.append((
                    user_id, username, email, first_name, last_name,
                    reg_date, last_login, is_verified, total_orders,
                    datetime.now().isoformat(), datetime.now().isoformat()
                ))
            
            cursor.executemany("""
                INSERT OR REPLACE INTO users 
                (user_id, username, email, first_name, last_name, registration_date, 
                 last_login, is_verified, total_orders, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, users_data)
            
            # Generate products
            categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Beauty']
            brands = ['BrandA', 'BrandB', 'BrandC', 'BrandD', 'BrandE']
            
            products_data = []
            for i in range(num_products):
                product_id = f"PROD_{i+1:06d}"
                name = f"Product {i+1}"
                category = random.choice(categories)
                subcategory = f"{category} Sub"
                brand = random.choice(brands)
                price = round(random.uniform(10.0, 500.0), 2)
                description = f"Description for {name}"
                is_active = 1
                stock_quantity = random.randint(0, 100)
                
                products_data.append((
                    product_id, name, category, subcategory, brand, price,
                    description, is_active, stock_quantity,
                    datetime.now().isoformat(), datetime.now().isoformat()
                ))
            
            cursor.executemany("""
                INSERT OR REPLACE INTO products 
                (product_id, name, category, subcategory, brand, price, description,
                 is_active, stock_quantity, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, products_data)
            
            # Generate orders
            orders_data = []
            order_items_data = []
            
            for i in range(num_orders):
                order_id = f"ORD_{i+1:08d}"
                user_id = f"USER_{random.randint(1, num_users):06d}"
                order_date = (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat()
                status = random.choice(['pending', 'processing', 'shipped', 'delivered', 'cancelled'])
                payment_method = random.choice(['credit_card', 'debit_card', 'paypal', 'bank_transfer'])
                
                # Generate order items
                num_items = random.randint(1, 5)
                total_amount = 0
                
                for j in range(num_items):
                    product_id = f"PROD_{random.randint(1, num_products):06d}"
                    quantity = random.randint(1, 3)
                    unit_price = round(random.uniform(10.0, 500.0), 2)
                    item_total = quantity * unit_price
                    total_amount += item_total
                    
                    order_items_data.append((
                        order_id, product_id, quantity, unit_price, item_total
                    ))
                
                orders_data.append((
                    order_id, user_id, order_date, round(total_amount, 2),
                    status, f"Address {i+1}", payment_method,
                    datetime.now().isoformat()
                ))
            
            cursor.executemany("""
                INSERT OR REPLACE INTO orders 
                (order_id, user_id, order_date, total_amount, status, 
                 shipping_address, payment_method, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, orders_data)
            
            cursor.executemany("""
                INSERT OR REPLACE INTO order_items 
                (order_id, product_id, quantity, unit_price, total_price)
                VALUES (?, ?, ?, ?, ?)
            """, order_items_data)
            
            conn.commit()
            logger.info(f"Populated SQLite database with {num_users} users, {num_products} products, {num_orders} orders")
    
    def extract_data(self, table_name: str, limit: int = None) -> List[Dict]:
        """Extract data from specific table"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = f"SELECT * FROM {table_name}"
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            return [dict(row) for row in rows]
    
    def get_table_info(self) -> Dict:
        """Get information about all tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            table_info = {}
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                table_info[table] = count
            
            return table_info


class PostgreSQLSimulator:
    """Simulates PostgreSQL database as a source system"""
    
    def __init__(self, connection_params: Dict):
        self.connection_params = connection_params
        self._setup_database()
    
    def _setup_database(self):
        """Setup PostgreSQL database with tables"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create schema for analytics
                cursor.execute("CREATE SCHEMA IF NOT EXISTS analytics")
                
                # User activity logs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analytics.user_activity_logs (
                        log_id SERIAL PRIMARY KEY,
                        user_id VARCHAR(50) NOT NULL,
                        activity_type VARCHAR(50) NOT NULL,
                        page_url VARCHAR(500),
                        session_id VARCHAR(100),
                        ip_address INET,
                        user_agent TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        additional_data JSONB
                    )
                """)
                
                # Product views table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analytics.product_views (
                        view_id SERIAL PRIMARY KEY,
                        user_id VARCHAR(50),
                        product_id VARCHAR(50) NOT NULL,
                        view_duration INTEGER,
                        referrer_url VARCHAR(500),
                        device_type VARCHAR(50),
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Search queries table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS analytics.search_queries (
                        query_id SERIAL PRIMARY KEY,
                        user_id VARCHAR(50),
                        search_term VARCHAR(500) NOT NULL,
                        results_count INTEGER,
                        clicked_product_id VARCHAR(50),
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                logger.info("PostgreSQL database setup completed")
                
        except Exception as e:
            logger.error(f"Error setting up PostgreSQL database: {e}")
    
    @contextmanager
    def get_connection(self):
        """Get PostgreSQL connection with context manager"""
        conn = psycopg2.connect(**self.connection_params)
        try:
            yield conn
        finally:
            conn.close()
    
    def populate_analytics_data(self, num_records: int = 10000):
        """Populate analytics tables with sample data"""
        logger.info("Populating PostgreSQL analytics data...")
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Generate user activity logs
                activity_types = ['login', 'logout', 'page_view', 'search', 'add_to_cart', 'purchase']
                
                for i in range(num_records):
                    user_id = f"USER_{random.randint(1, 1000):06d}"
                    activity_type = random.choice(activity_types)
                    page_url = f"/page/{random.randint(1, 100)}"
                    session_id = f"sess_{random.randint(100000, 999999)}"
                    ip_address = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
                    user_agent = "Mozilla/5.0 (compatible; DataEngineering/1.0)"
                    timestamp = datetime.now() - timedelta(minutes=random.randint(0, 10080))  # Last week
                    additional_data = json.dumps({"browser": "Chrome", "os": "Windows"})
                    
                    cursor.execute("""
                        INSERT INTO analytics.user_activity_logs 
                        (user_id, activity_type, page_url, session_id, ip_address, 
                         user_agent, timestamp, additional_data)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (user_id, activity_type, page_url, session_id, ip_address, 
                          user_agent, timestamp, additional_data))
                
                conn.commit()
                logger.info(f"Populated PostgreSQL with {num_records} analytics records")
                
        except Exception as e:
            logger.error(f"Error populating PostgreSQL data: {e}")


class MongoDBSimulator:
    """Simulates MongoDB as a document-based source system"""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/"):
        self.connection_string = connection_string
        self.client = None
        self.db = None
        self._setup_database()
    
    def _setup_database(self):
        """Setup MongoDB database and collections"""
        try:
            self.client = pymongo.MongoClient(self.connection_string)
            self.db = self.client.ecommerce_reviews
            
            # Create collections with indexes
            self.db.product_catalog.create_index("product_id")
            self.db.user_profiles.create_index("user_id")
            self.db.review_metadata.create_index([("product_id", 1), ("timestamp", -1)])
            
            logger.info("MongoDB database setup completed")
            
        except Exception as e:
            logger.error(f"Error setting up MongoDB: {e}")
    
    def populate_document_data(self, num_products: int = 500):
        """Populate MongoDB with document-based data"""
        logger.info("Populating MongoDB with document data...")
        
        try:
            # Product catalog with rich metadata
            products = []
            for i in range(num_products):
                product = {
                    "product_id": f"PROD_{i+1:06d}",
                    "name": f"Product {i+1}",
                    "description": f"Detailed description for product {i+1}",
                    "specifications": {
                        "weight": f"{random.uniform(0.1, 10.0):.2f} kg",
                        "dimensions": {
                            "length": random.randint(10, 100),
                            "width": random.randint(10, 100),
                            "height": random.randint(5, 50)
                        },
                        "color": random.choice(["Red", "Blue", "Green", "Black", "White"]),
                        "material": random.choice(["Plastic", "Metal", "Wood", "Glass", "Fabric"])
                    },
                    "categories": {
                        "primary": random.choice(["Electronics", "Clothing", "Home"]),
                        "secondary": random.choice(["Gadgets", "Accessories", "Furniture"]),
                        "tags": random.sample(["popular", "new", "sale", "premium", "eco-friendly"], 2)
                    },
                    "pricing": {
                        "base_price": round(random.uniform(10.0, 500.0), 2),
                        "discount_percentage": random.randint(0, 50),
                        "currency": "USD"
                    },
                    "availability": {
                        "in_stock": random.choice([True, False]),
                        "quantity": random.randint(0, 100),
                        "warehouse_locations": random.sample(["NY", "CA", "TX", "FL"], 2)
                    },
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "last_updated": datetime.now().isoformat(),
                        "version": "1.0"
                    }
                }
                products.append(product)
            
            self.db.product_catalog.insert_many(products)
            
            # User profiles with preferences
            users = []
            for i in range(1000):
                user = {
                    "user_id": f"USER_{i+1:06d}",
                    "profile": {
                        "preferences": {
                            "categories": random.sample(["Electronics", "Clothing", "Home", "Sports"], 2),
                            "price_range": {
                                "min": random.randint(10, 50),
                                "max": random.randint(100, 500)
                            },
                            "brands": random.sample(["BrandA", "BrandB", "BrandC"], 2)
                        },
                        "demographics": {
                            "age_group": random.choice(["18-25", "26-35", "36-45", "46-55", "55+"]),
                            "location": {
                                "country": "USA",
                                "state": random.choice(["NY", "CA", "TX", "FL", "IL"]),
                                "city": f"City{random.randint(1, 100)}"
                            }
                        },
                        "behavior": {
                            "avg_session_duration": random.randint(300, 3600),
                            "purchase_frequency": random.choice(["weekly", "monthly", "quarterly"]),
                            "device_preference": random.choice(["mobile", "desktop", "tablet"])
                        }
                    },
                    "metadata": {
                        "created_at": datetime.now().isoformat(),
                        "last_active": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
                    }
                }
                users.append(user)
            
            self.db.user_profiles.insert_many(users)
            
            logger.info(f"Populated MongoDB with {num_products} products and 1000 user profiles")
            
        except Exception as e:
            logger.error(f"Error populating MongoDB data: {e}")
    
    def extract_documents(self, collection_name: str, query: Dict = None, limit: int = None) -> List[Dict]:
        """Extract documents from MongoDB collection"""
        try:
            collection = self.db[collection_name]
            cursor = collection.find(query or {})
            
            if limit:
                cursor = cursor.limit(limit)
            
            return list(cursor)
            
        except Exception as e:
            logger.error(f"Error extracting MongoDB documents: {e}")
            return []


class DatabaseExtractor:
    """Unified interface for extracting data from different database sources"""
    
    def __init__(self):
        self.sqlite_sim = SQLiteSimulator()
        # PostgreSQL and MongoDB simulators would be initialized with actual connection params
    
    def extract_all_data(self) -> Dict:
        """Extract data from all database sources"""
        logger.info("Extracting data from all database sources...")
        
        extracted_data = {
            "sqlite": {
                "users": self.sqlite_sim.extract_data("users", limit=100),
                "products": self.sqlite_sim.extract_data("products", limit=100),
                "orders": self.sqlite_sim.extract_data("orders", limit=100),
                "reviews": self.sqlite_sim.extract_data("reviews", limit=100)
            },
            "extraction_timestamp": datetime.now().isoformat(),
            "source_info": self.sqlite_sim.get_table_info()
        }
        
        return extracted_data
    
    def save_extracted_data(self, data: Dict, output_dir: str = "data/raw/extracted"):
        """Save extracted data to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save SQLite data
        for table_name, table_data in data["sqlite"].items():
            if table_data:  # Only save if data exists
                df = pd.DataFrame(table_data)
                
                # Save as CSV
                csv_path = f"{output_dir}/{table_name}_sqlite.csv"
                df.to_csv(csv_path, index=False)
                
                # Save as JSON
                json_path = f"{output_dir}/{table_name}_sqlite.json"
                with open(json_path, 'w') as f:
                    json.dump(table_data, f, indent=2)
                
                logger.info(f"Saved {table_name} data to {csv_path} and {json_path}")


if __name__ == "__main__":
    # Initialize and populate SQLite database
    sqlite_sim = SQLiteSimulator()
    sqlite_sim.populate_sample_data()
    
    # Extract and save data
    extractor = DatabaseExtractor()
    extracted_data = extractor.extract_all_data()
    extractor.save_extracted_data(extracted_data)
    
    logger.info("Database simulation and extraction completed!")