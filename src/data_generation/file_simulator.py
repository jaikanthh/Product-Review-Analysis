"""
File-based Data Simulator for Product Review Analysis
Simulates various file formats as data sources (CSV, JSON, Parquet, XML, etc.)
Demonstrates file-based data generation and batch processing patterns
"""

import pandas as pd
import json
import xml.etree.ElementTree as ET
import csv
import random
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import logging
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileDataSimulator:
    """Simulates various file-based data sources"""
    
    def __init__(self, base_path: str = "data/raw/files"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different file types
        self.csv_path = self.base_path / "csv"
        self.json_path = self.base_path / "json"
        self.parquet_path = self.base_path / "parquet"
        self.xml_path = self.base_path / "xml"
        self.logs_path = self.base_path / "logs"
        
        for path in [self.csv_path, self.json_path, self.parquet_path, self.xml_path, self.logs_path]:
            path.mkdir(exist_ok=True)
    
    def generate_csv_files(self, num_files: int = 5, records_per_file: int = 1000):
        """Generate CSV files simulating daily batch exports"""
        logger.info(f"Generating {num_files} CSV files with {records_per_file} records each...")
        
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Beauty']
        
        for file_idx in range(num_files):
            # Simulate daily exports
            file_date = datetime.now() - timedelta(days=file_idx)
            filename = f"product_reviews_{file_date.strftime('%Y%m%d')}.csv"
            filepath = self.csv_path / filename
            
            # Generate review data
            reviews = []
            for i in range(records_per_file):
                review = {
                    'review_id': f"REV_{file_date.strftime('%Y%m%d')}_{i+1:06d}",
                    'user_id': f"USER_{random.randint(1, 10000):06d}",
                    'product_id': f"PROD_{random.randint(1, 5000):06d}",
                    'product_name': f"Product {random.randint(1, 5000)}",
                    'category': random.choice(categories),
                    'rating': random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.10, 0.20, 0.35, 0.30])[0],
                    'title': f"Review title {i+1}",
                    'review_text': self._generate_review_text(),
                    'verified_purchase': random.choice([True, False]),
                    'helpful_votes': random.randint(0, 100),
                    'total_votes': random.randint(0, 150),
                    'review_date': file_date.strftime('%Y-%m-%d'),
                    'review_time': f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}",
                    'price_paid': round(random.uniform(10.0, 500.0), 2),
                    'discount_applied': round(random.uniform(0, 50), 2),
                    'shipping_cost': round(random.uniform(0, 25.0), 2),
                    'delivery_days': random.randint(1, 14)
                }
                reviews.append(review)
            
            # Save to CSV
            df = pd.DataFrame(reviews)
            df.to_csv(filepath, index=False)
            logger.info(f"Generated CSV file: {filepath}")
    
    def generate_json_files(self, num_files: int = 3, records_per_file: int = 500):
        """Generate JSON files simulating API exports and nested data"""
        logger.info(f"Generating {num_files} JSON files with {records_per_file} records each...")
        
        for file_idx in range(num_files):
            file_date = datetime.now() - timedelta(days=file_idx)
            filename = f"user_activity_{file_date.strftime('%Y%m%d')}.json"
            filepath = self.json_path / filename
            
            # Generate user activity data with nested structure
            activities = []
            for i in range(records_per_file):
                activity = {
                    'activity_id': f"ACT_{file_date.strftime('%Y%m%d')}_{i+1:06d}",
                    'user_id': f"USER_{random.randint(1, 10000):06d}",
                    'session_info': {
                        'session_id': f"sess_{random.randint(100000, 999999)}",
                        'start_time': (file_date + timedelta(minutes=random.randint(0, 1440))).isoformat(),
                        'duration_minutes': random.randint(1, 120),
                        'device_info': {
                            'type': random.choice(['mobile', 'desktop', 'tablet']),
                            'os': random.choice(['iOS', 'Android', 'Windows', 'macOS']),
                            'browser': random.choice(['Chrome', 'Safari', 'Firefox', 'Edge'])
                        }
                    },
                    'actions': [
                        {
                            'action_type': random.choice(['page_view', 'product_view', 'search', 'add_to_cart', 'purchase']),
                            'timestamp': (file_date + timedelta(minutes=random.randint(0, 1440))).isoformat(),
                            'details': {
                                'page_url': f"/page/{random.randint(1, 1000)}",
                                'product_id': f"PROD_{random.randint(1, 5000):06d}" if random.random() > 0.5 else None,
                                'search_term': f"search term {random.randint(1, 100)}" if random.random() > 0.7 else None
                            }
                        } for _ in range(random.randint(1, 10))
                    ],
                    'user_profile': {
                        'age_group': random.choice(['18-25', '26-35', '36-45', '46-55', '55+']),
                        'location': {
                            'country': 'USA',
                            'state': random.choice(['NY', 'CA', 'TX', 'FL', 'IL']),
                            'city': f"City{random.randint(1, 100)}"
                        },
                        'preferences': {
                            'categories': random.sample(['Electronics', 'Clothing', 'Home', 'Sports'], 2),
                            'price_range': {
                                'min': random.randint(10, 50),
                                'max': random.randint(100, 500)
                            }
                        }
                    },
                    'metadata': {
                        'created_at': file_date.isoformat(),
                        'data_version': '1.0',
                        'source_system': 'web_analytics'
                    }
                }
                activities.append(activity)
            
            # Save to JSON
            with open(filepath, 'w') as f:
                json.dump({
                    'export_date': file_date.strftime('%Y-%m-%d'),
                    'total_records': len(activities),
                    'activities': activities
                }, f, indent=2)
            
            logger.info(f"Generated JSON file: {filepath}")
    
    def generate_parquet_files(self, num_files: int = 4, records_per_file: int = 2000):
        """Generate Parquet files simulating big data exports"""
        logger.info(f"Generating {num_files} Parquet files with {records_per_file} records each...")
        
        for file_idx in range(num_files):
            file_date = datetime.now() - timedelta(days=file_idx)
            filename = f"sales_data_{file_date.strftime('%Y%m%d')}.parquet"
            filepath = self.parquet_path / filename
            
            # Generate sales transaction data
            transactions = []
            for i in range(records_per_file):
                transaction = {
                    'transaction_id': f"TXN_{file_date.strftime('%Y%m%d')}_{i+1:08d}",
                    'order_id': f"ORD_{file_date.strftime('%Y%m%d')}_{random.randint(1, records_per_file//2):06d}",
                    'user_id': f"USER_{random.randint(1, 10000):06d}",
                    'product_id': f"PROD_{random.randint(1, 5000):06d}",
                    'quantity': random.randint(1, 5),
                    'unit_price': round(random.uniform(5.0, 200.0), 2),
                    'total_amount': 0,  # Will calculate below
                    'discount_amount': round(random.uniform(0, 20.0), 2),
                    'tax_amount': 0,  # Will calculate below
                    'shipping_cost': round(random.uniform(0, 15.0), 2),
                    'payment_method': random.choice(['credit_card', 'debit_card', 'paypal', 'bank_transfer']),
                    'transaction_status': random.choice(['completed', 'pending', 'failed', 'refunded']),
                    'transaction_timestamp': (file_date + timedelta(
                        hours=random.randint(0, 23),
                        minutes=random.randint(0, 59),
                        seconds=random.randint(0, 59)
                    )).isoformat(),
                    'customer_segment': random.choice(['premium', 'regular', 'new', 'vip']),
                    'sales_channel': random.choice(['web', 'mobile_app', 'phone', 'store']),
                    'promotion_code': f"PROMO{random.randint(1, 100)}" if random.random() > 0.7 else None,
                    'referrer_source': random.choice(['google', 'facebook', 'email', 'direct', 'affiliate']),
                    'device_category': random.choice(['mobile', 'desktop', 'tablet']),
                    'geographic_region': random.choice(['North', 'South', 'East', 'West', 'Central'])
                }
                
                # Calculate derived fields
                subtotal = transaction['quantity'] * transaction['unit_price']
                transaction['total_amount'] = round(subtotal - transaction['discount_amount'] + transaction['shipping_cost'], 2)
                transaction['tax_amount'] = round(transaction['total_amount'] * 0.08, 2)  # 8% tax
                transaction['total_amount'] += transaction['tax_amount']
                
                transactions.append(transaction)
            
            # Convert to DataFrame and save as Parquet
            df = pd.DataFrame(transactions)
            
            # Convert timestamp to datetime
            df['transaction_timestamp'] = pd.to_datetime(df['transaction_timestamp'])
            
            # Save with compression
            df.to_parquet(filepath, compression='snappy', index=False)
            logger.info(f"Generated Parquet file: {filepath}")
    
    def generate_xml_files(self, num_files: int = 2, records_per_file: int = 300):
        """Generate XML files simulating legacy system exports"""
        logger.info(f"Generating {num_files} XML files with {records_per_file} records each...")
        
        for file_idx in range(num_files):
            file_date = datetime.now() - timedelta(days=file_idx)
            filename = f"product_catalog_{file_date.strftime('%Y%m%d')}.xml"
            filepath = self.xml_path / filename
            
            # Create XML structure
            root = ET.Element("ProductCatalog")
            root.set("export_date", file_date.strftime('%Y-%m-%d'))
            root.set("total_products", str(records_per_file))
            root.set("version", "1.0")
            
            for i in range(records_per_file):
                product = ET.SubElement(root, "Product")
                product.set("id", f"PROD_{random.randint(1, 5000):06d}")
                
                # Basic product info
                ET.SubElement(product, "Name").text = f"Product {i+1}"
                ET.SubElement(product, "Description").text = f"Description for product {i+1}"
                ET.SubElement(product, "Category").text = random.choice(['Electronics', 'Clothing', 'Home'])
                ET.SubElement(product, "Brand").text = random.choice(['BrandA', 'BrandB', 'BrandC'])
                ET.SubElement(product, "Price").text = str(round(random.uniform(10.0, 500.0), 2))
                ET.SubElement(product, "Currency").text = "USD"
                
                # Inventory info
                inventory = ET.SubElement(product, "Inventory")
                ET.SubElement(inventory, "InStock").text = str(random.choice([True, False])).lower()
                ET.SubElement(inventory, "Quantity").text = str(random.randint(0, 100))
                ET.SubElement(inventory, "WarehouseLocation").text = random.choice(['NY', 'CA', 'TX'])
                
                # Specifications
                specs = ET.SubElement(product, "Specifications")
                ET.SubElement(specs, "Weight").text = f"{random.uniform(0.1, 10.0):.2f} kg"
                ET.SubElement(specs, "Color").text = random.choice(['Red', 'Blue', 'Green', 'Black'])
                ET.SubElement(specs, "Material").text = random.choice(['Plastic', 'Metal', 'Wood'])
                
                # Reviews summary
                reviews_summary = ET.SubElement(product, "ReviewsSummary")
                ET.SubElement(reviews_summary, "AverageRating").text = str(round(random.uniform(1.0, 5.0), 1))
                ET.SubElement(reviews_summary, "TotalReviews").text = str(random.randint(0, 1000))
                ET.SubElement(reviews_summary, "LastReviewDate").text = (file_date - timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d')
            
            # Save XML file
            tree = ET.ElementTree(root)
            tree.write(filepath, encoding='utf-8', xml_declaration=True)
            logger.info(f"Generated XML file: {filepath}")
    
    def generate_log_files(self, num_files: int = 7, lines_per_file: int = 5000):
        """Generate log files simulating application and server logs"""
        logger.info(f"Generating {num_files} log files with {lines_per_file} lines each...")
        
        log_levels = ['INFO', 'WARN', 'ERROR', 'DEBUG']
        components = ['WebServer', 'Database', 'PaymentService', 'ReviewService', 'UserService']
        
        for file_idx in range(num_files):
            file_date = datetime.now() - timedelta(days=file_idx)
            filename = f"application_{file_date.strftime('%Y%m%d')}.log"
            filepath = self.logs_path / filename
            
            with open(filepath, 'w') as f:
                for i in range(lines_per_file):
                    timestamp = file_date + timedelta(
                        hours=random.randint(0, 23),
                        minutes=random.randint(0, 59),
                        seconds=random.randint(0, 59),
                        microseconds=random.randint(0, 999999)
                    )
                    
                    level = random.choice(log_levels)
                    component = random.choice(components)
                    
                    # Generate different types of log messages
                    if level == 'ERROR':
                        messages = [
                            "Database connection timeout",
                            "Payment processing failed",
                            "User authentication error",
                            "Review submission validation failed",
                            "External API call failed"
                        ]
                    elif level == 'WARN':
                        messages = [
                            "High memory usage detected",
                            "Slow query performance",
                            "Rate limit approaching",
                            "Cache miss ratio high",
                            "Disk space low"
                        ]
                    else:
                        messages = [
                            "User login successful",
                            "Review submitted successfully",
                            "Product view recorded",
                            "Search query processed",
                            "Order completed"
                        ]
                    
                    message = random.choice(messages)
                    user_id = f"USER_{random.randint(1, 10000):06d}" if random.random() > 0.3 else "SYSTEM"
                    request_id = f"req_{random.randint(100000, 999999)}"
                    
                    log_line = f"{timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} [{level}] {component} - {message} | user_id={user_id} request_id={request_id}\n"
                    f.write(log_line)
            
            logger.info(f"Generated log file: {filepath}")
    
    def generate_delimited_files(self, num_files: int = 3):
        """Generate files with different delimiters (TSV, pipe-separated, etc.)"""
        logger.info(f"Generating {num_files} delimited files...")
        
        delimiters = [('\t', 'tsv'), ('|', 'pipe'), (';', 'semicolon')]
        
        for file_idx, (delimiter, suffix) in enumerate(delimiters[:num_files]):
            file_date = datetime.now() - timedelta(days=file_idx)
            filename = f"customer_data_{file_date.strftime('%Y%m%d')}.{suffix}"
            filepath = self.base_path / filename
            
            # Generate customer data
            customers = []
            for i in range(1000):
                customer = {
                    'customer_id': f"CUST_{i+1:06d}",
                    'first_name': random.choice(['John', 'Jane', 'Mike', 'Sarah', 'David']),
                    'last_name': random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones']),
                    'email': f"customer{i+1}@example.com",
                    'phone': f"+1-{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
                    'address': f"{random.randint(1, 9999)} Main St",
                    'city': f"City{random.randint(1, 100)}",
                    'state': random.choice(['NY', 'CA', 'TX', 'FL', 'IL']),
                    'zip_code': f"{random.randint(10000, 99999)}",
                    'registration_date': (file_date - timedelta(days=random.randint(0, 365))).strftime('%Y-%m-%d'),
                    'last_purchase_date': (file_date - timedelta(days=random.randint(0, 90))).strftime('%Y-%m-%d'),
                    'total_spent': round(random.uniform(50.0, 5000.0), 2),
                    'loyalty_tier': random.choice(['Bronze', 'Silver', 'Gold', 'Platinum'])
                }
                customers.append(customer)
            
            # Save with specific delimiter
            df = pd.DataFrame(customers)
            df.to_csv(filepath, sep=delimiter, index=False)
            logger.info(f"Generated {suffix.upper()} file: {filepath}")
    
    def _generate_review_text(self) -> str:
        """Generate realistic review text"""
        positive_phrases = [
            "Great product!", "Excellent quality", "Highly recommend",
            "Perfect for my needs", "Amazing value", "Fast shipping"
        ]
        
        negative_phrases = [
            "Poor quality", "Not as described", "Waste of money",
            "Broke after a week", "Terrible customer service", "Overpriced"
        ]
        
        neutral_phrases = [
            "Average product", "Okay for the price", "Does what it says",
            "Nothing special", "Standard quality", "As expected"
        ]
        
        all_phrases = positive_phrases + negative_phrases + neutral_phrases
        return " ".join(random.sample(all_phrases, random.randint(2, 4)))
    
    def generate_all_files(self):
        """Generate all types of files"""
        logger.info("Starting comprehensive file generation...")
        
        self.generate_csv_files()
        self.generate_json_files()
        self.generate_parquet_files()
        self.generate_xml_files()
        self.generate_log_files()
        self.generate_delimited_files()
        
        logger.info("All file generation completed!")
    
    def get_file_inventory(self) -> Dict:
        """Get inventory of all generated files"""
        inventory = {}
        
        for subdir in [self.csv_path, self.json_path, self.parquet_path, self.xml_path, self.logs_path]:
            subdir_name = subdir.name
            files = list(subdir.glob('*'))
            inventory[subdir_name] = {
                'count': len(files),
                'files': [f.name for f in files],
                'total_size_mb': sum(f.stat().st_size for f in files) / (1024 * 1024)
            }
        
        # Add root level files
        root_files = [f for f in self.base_path.glob('*') if f.is_file()]
        if root_files:
            inventory['root'] = {
                'count': len(root_files),
                'files': [f.name for f in root_files],
                'total_size_mb': sum(f.stat().st_size for f in root_files) / (1024 * 1024)
            }
        
        return inventory


class FileDataProcessor:
    """Processes and validates generated files"""
    
    def __init__(self, base_path: str = "data/raw/files"):
        self.base_path = Path(base_path)
    
    def validate_csv_files(self) -> Dict:
        """Validate CSV files and return summary"""
        csv_path = self.base_path / "csv"
        results = {}
        
        for csv_file in csv_path.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                results[csv_file.name] = {
                    'status': 'valid',
                    'rows': len(df),
                    'columns': len(df.columns),
                    'size_mb': csv_file.stat().st_size / (1024 * 1024),
                    'sample_data': df.head(2).to_dict('records')
                }
            except Exception as e:
                results[csv_file.name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    def validate_json_files(self) -> Dict:
        """Validate JSON files and return summary"""
        json_path = self.base_path / "json"
        results = {}
        
        for json_file in json_path.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                results[json_file.name] = {
                    'status': 'valid',
                    'records': len(data.get('activities', [])),
                    'size_mb': json_file.stat().st_size / (1024 * 1024),
                    'structure': list(data.keys())
                }
            except Exception as e:
                results[json_file.name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return results
    
    def validate_all_files(self) -> Dict:
        """Validate all generated files"""
        return {
            'csv_validation': self.validate_csv_files(),
            'json_validation': self.validate_json_files(),
            'validation_timestamp': datetime.now().isoformat()
        }


if __name__ == "__main__":
    # Generate all file types
    simulator = FileDataSimulator()
    simulator.generate_all_files()
    
    # Get file inventory
    inventory = simulator.get_file_inventory()
    logger.info(f"File generation completed. Inventory: {inventory}")
    
    # Validate generated files
    processor = FileDataProcessor()
    validation_results = processor.validate_all_files()
    logger.info("File validation completed!")
    
    # Save inventory and validation results
    with open("data/raw/files/file_inventory.json", 'w') as f:
        json.dump(inventory, f, indent=2)
    
    with open("data/raw/files/validation_results.json", 'w') as f:
        json.dump(validation_results, f, indent=2)