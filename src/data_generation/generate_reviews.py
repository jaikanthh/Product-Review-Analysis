"""
Data Generation Module for Product Review Analysis
Simulates multiple source systems generating e-commerce review data
Demonstrates the "Generation" phase of the Data Engineering Lifecycle
"""

import pandas as pd
import numpy as np
import json
import csv
from datetime import datetime, timedelta
from faker import Faker
import random
import os
import yaml
from typing import Dict, List, Tuple
import uuid


class ReviewDataGenerator:
    """Generates realistic e-commerce review data from multiple source systems"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        self.fake = Faker()
        self.config = self._load_config(config_path)
        self.categories = self.config['data_generation']['categories']
        self.rating_distribution = self.config['data_generation']['review_distribution']
        
        # Ensure output directories exist
        self._create_directories()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            # Default configuration if file not found
            return {
                'data_generation': {
                    'sample_size': {'users': 1000, 'products': 500, 'reviews': 5000},
                    'categories': ['Electronics', 'Clothing', 'Books'],
                    'review_distribution': {
                        'rating_1': 0.05, 'rating_2': 0.10, 'rating_3': 0.20,
                        'rating_4': 0.35, 'rating_5': 0.30
                    }
                }
            }
    
    def _create_directories(self):
        """Create necessary directories for data storage"""
        directories = [
            'data/raw/users',
            'data/raw/products', 
            'data/raw/reviews',
            'data/raw/api_logs',
            'data/processed',
            'data/warehouse'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def generate_users(self, num_users: int = None) -> pd.DataFrame:
        """Generate user data simulating customer database"""
        if num_users is None:
            num_users = self.config['data_generation']['sample_size']['users']
        
        users = []
        for i in range(num_users):
            user = {
                'user_id': f"USER_{i+1:06d}",
                'username': self.fake.user_name(),
                'email': self.fake.email(),
                'first_name': self.fake.first_name(),
                'last_name': self.fake.last_name(),
                'date_of_birth': self.fake.date_of_birth(minimum_age=18, maximum_age=80),
                'gender': random.choice(['M', 'F', 'Other']),
                'country': self.fake.country(),
                'city': self.fake.city(),
                'registration_date': self.fake.date_between(start_date='-2y', end_date='today'),
                'is_verified': random.choice([True, False]),
                'total_orders': random.randint(0, 50),
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            users.append(user)
        
        df_users = pd.DataFrame(users)
        
        # Save to multiple formats (simulating different source systems)
        df_users.to_csv('data/raw/users/users.csv', index=False)
        df_users.to_json('data/raw/users/users.json', orient='records', date_format='iso')
        
        print(f"Generated {len(df_users)} users")
        return df_users
    
    def generate_products(self, num_products: int = None) -> pd.DataFrame:
        """Generate product catalog data"""
        if num_products is None:
            num_products = self.config['data_generation']['sample_size']['products']
        
        products = []
        for i in range(num_products):
            category = random.choice(self.categories)
            product = {
                'product_id': f"PROD_{i+1:06d}",
                'name': self._generate_product_name(category),
                'category': category,
                'subcategory': self._generate_subcategory(category),
                'brand': self.fake.company(),
                'price': round(random.uniform(10.0, 1000.0), 2),
                'description': self.fake.text(max_nb_chars=500),
                'weight_kg': round(random.uniform(0.1, 10.0), 2),
                'dimensions': f"{random.randint(10,50)}x{random.randint(10,50)}x{random.randint(5,30)}",
                'color': self.fake.color_name(),
                'material': random.choice(['Plastic', 'Metal', 'Wood', 'Fabric', 'Glass']),
                'manufacturer': self.fake.company(),
                'model_number': f"MODEL_{random.randint(1000, 9999)}",
                'release_date': self.fake.date_between(start_date='-3y', end_date='today'),
                'is_active': random.choice([True, True, True, False]),  # 75% active
                'stock_quantity': random.randint(0, 1000),
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            products.append(product)
        
        df_products = pd.DataFrame(products)
        
        # Save to multiple formats
        df_products.to_csv('data/raw/products/products.csv', index=False)
        df_products.to_json('data/raw/products/products.json', orient='records', date_format='iso')
        
        # Save as parquet (simulating data lake format)
        df_products.to_parquet('data/raw/products/products.parquet')
        
        print(f"Generated {len(df_products)} products")
        return df_products
    
    def generate_reviews(self, users_df: pd.DataFrame, products_df: pd.DataFrame, 
                        num_reviews: int = None) -> pd.DataFrame:
        """Generate product reviews with realistic patterns"""
        if num_reviews is None:
            num_reviews = self.config['data_generation']['sample_size']['reviews']
        
        reviews = []
        
        # Create rating distribution
        ratings = []
        for rating, probability in self.rating_distribution.items():
            rating_value = int(rating.split('_')[1])
            count = int(num_reviews * probability)
            ratings.extend([rating_value] * count)
        
        # Fill remaining slots with random ratings
        while len(ratings) < num_reviews:
            ratings.append(random.randint(1, 5))
        
        random.shuffle(ratings)
        
        for i in range(num_reviews):
            user = users_df.sample(1).iloc[0]
            product = products_df.sample(1).iloc[0]
            rating = ratings[i]
            
            review = {
                'review_id': f"REV_{i+1:08d}",
                'user_id': user['user_id'],
                'product_id': product['product_id'],
                'rating': rating,
                'title': self._generate_review_title(rating),
                'review_text': self._generate_review_text(rating, product['category']),
                'helpful_votes': random.randint(0, 100),
                'total_votes': random.randint(0, 150),
                'verified_purchase': random.choice([True, True, False]),  # 67% verified
                'review_date': self.fake.date_between(start_date='-1y', end_date='today'),
                'language': 'en',
                'sentiment_score': self._calculate_sentiment_score(rating),
                'word_count': 0,  # Will be calculated
                'contains_images': random.choice([True, False]),
                'contains_video': random.choice([True, False, False, False]),  # 25% chance
                'is_spam': random.choice([True, False, False, False, False]),  # 20% spam
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            
            # Calculate word count
            review['word_count'] = len(review['review_text'].split())
            
            reviews.append(review)
        
        df_reviews = pd.DataFrame(reviews)
        
        # Save to multiple formats (simulating different source systems)
        df_reviews.to_csv('data/raw/reviews/reviews.csv', index=False)
        df_reviews.to_json('data/raw/reviews/reviews.json', orient='records', date_format='iso')
        df_reviews.to_parquet('data/raw/reviews/reviews.parquet')
        
        # Save daily partitioned data (simulating streaming data)
        self._save_partitioned_reviews(df_reviews)
        
        print(f"Generated {len(df_reviews)} reviews")
        return df_reviews
    
    def _generate_product_name(self, category: str) -> str:
        """Generate realistic product names based on category"""
        category_names = {
            'Electronics': ['Smartphone', 'Laptop', 'Tablet', 'Headphones', 'Camera', 'TV'],
            'Clothing': ['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Shoes', 'Hat'],
            'Books': ['Novel', 'Textbook', 'Biography', 'Cookbook', 'Guide', 'Manual'],
            'Home & Garden': ['Chair', 'Table', 'Lamp', 'Plant', 'Tool', 'Decoration'],
            'Sports & Outdoors': ['Ball', 'Equipment', 'Gear', 'Clothing', 'Shoes', 'Accessory'],
            'Beauty & Personal Care': ['Cream', 'Shampoo', 'Makeup', 'Perfume', 'Soap', 'Lotion'],
            'Toys & Games': ['Toy', 'Game', 'Puzzle', 'Doll', 'Car', 'Building Set'],
            'Automotive': ['Part', 'Accessory', 'Tool', 'Oil', 'Filter', 'Cover']
        }
        
        base_names = category_names.get(category, ['Product'])
        base_name = random.choice(base_names)
        brand = self.fake.company().split()[0]
        model = random.choice(['Pro', 'Max', 'Elite', 'Premium', 'Standard', 'Basic'])
        
        return f"{brand} {base_name} {model}"
    
    def _generate_subcategory(self, category: str) -> str:
        """Generate subcategories based on main category"""
        subcategories = {
            'Electronics': ['Mobile Phones', 'Computers', 'Audio', 'Photography', 'TV & Video'],
            'Clothing': ['Men', 'Women', 'Kids', 'Accessories', 'Footwear'],
            'Books': ['Fiction', 'Non-Fiction', 'Educational', 'Children', 'Reference'],
            'Home & Garden': ['Furniture', 'Decor', 'Kitchen', 'Garden', 'Tools'],
            'Sports & Outdoors': ['Fitness', 'Outdoor Recreation', 'Team Sports', 'Water Sports'],
            'Beauty & Personal Care': ['Skincare', 'Haircare', 'Makeup', 'Fragrance', 'Personal Care'],
            'Toys & Games': ['Action Figures', 'Board Games', 'Educational', 'Electronic', 'Outdoor'],
            'Automotive': ['Parts', 'Accessories', 'Tools', 'Maintenance', 'Electronics']
        }
        
        return random.choice(subcategories.get(category, ['General']))
    
    def _generate_review_title(self, rating: int) -> str:
        """Generate review titles based on rating"""
        positive_titles = [
            "Excellent product!", "Highly recommended", "Great value for money",
            "Perfect!", "Love it!", "Amazing quality", "Best purchase ever"
        ]
        
        negative_titles = [
            "Disappointed", "Not worth it", "Poor quality", "Waste of money",
            "Terrible experience", "Don't buy this", "Regret buying"
        ]
        
        neutral_titles = [
            "It's okay", "Average product", "Mixed feelings", "Could be better",
            "Decent but not great", "Fair quality"
        ]
        
        if rating >= 4:
            return random.choice(positive_titles)
        elif rating <= 2:
            return random.choice(negative_titles)
        else:
            return random.choice(neutral_titles)
    
    def _generate_review_text(self, rating: int, category: str) -> str:
        """Generate realistic review text based on rating and category"""
        if rating >= 4:
            templates = [
                f"I'm really happy with this {category.lower()} product. The quality is excellent and it works exactly as described. Would definitely recommend to others.",
                f"Great {category.lower()} item! Fast shipping, good packaging, and the product exceeded my expectations. Very satisfied with this purchase.",
                f"This {category.lower()} product is fantastic. Good build quality, works perfectly, and great value for the price. Will buy again!"
            ]
        elif rating <= 2:
            templates = [
                f"Very disappointed with this {category.lower()} product. Poor quality and doesn't work as advertised. Would not recommend.",
                f"Terrible {category.lower()} item. Broke after just a few uses. Customer service was unhelpful. Waste of money.",
                f"This {category.lower()} product is awful. Cheap materials, poor construction, and doesn't match the description at all."
            ]
        else:
            templates = [
                f"This {category.lower()} product is okay. It works but nothing special. Average quality for the price.",
                f"Mixed feelings about this {category.lower()} item. Some good points but also some issues. Could be better.",
                f"Decent {category.lower()} product but not amazing. Does what it's supposed to do but room for improvement."
            ]
        
        base_text = random.choice(templates)
        
        # Add some random additional details
        additional_details = [
            " The packaging was good.",
            " Shipping was fast.",
            " Easy to use.",
            " Good customer service.",
            " Would buy again.",
            " Recommended by a friend.",
            " Exactly as pictured.",
            " Good value for money."
        ]
        
        if random.random() < 0.7:  # 70% chance to add details
            base_text += random.choice(additional_details)
        
        return base_text
    
    def _calculate_sentiment_score(self, rating: int) -> float:
        """Calculate sentiment score based on rating with some noise"""
        base_score = (rating - 1) / 4  # Convert 1-5 to 0-1 scale
        noise = random.uniform(-0.1, 0.1)  # Add some noise
        return max(0, min(1, base_score + noise))
    
    def _save_partitioned_reviews(self, df_reviews: pd.DataFrame):
        """Save reviews in partitioned format by date (simulating streaming data)"""
        df_reviews['review_date'] = pd.to_datetime(df_reviews['review_date'])
        
        for date, group in df_reviews.groupby(df_reviews['review_date'].dt.date):
            year = date.year
            month = date.month
            day = date.day
            
            partition_path = f"data/raw/reviews/partitioned/year={year}/month={month:02d}/day={day:02d}"
            os.makedirs(partition_path, exist_ok=True)
            
            group.to_parquet(f"{partition_path}/reviews.parquet")
    
    def generate_api_logs(self, num_requests: int = 1000):
        """Generate API access logs (simulating API gateway logs)"""
        api_logs = []
        
        endpoints = [
            '/api/v1/products',
            '/api/v1/reviews',
            '/api/v1/users',
            '/api/v1/categories',
            '/api/v1/search'
        ]
        
        methods = ['GET', 'POST', 'PUT', 'DELETE']
        status_codes = [200, 201, 400, 401, 404, 500]
        status_weights = [0.7, 0.1, 0.1, 0.05, 0.03, 0.02]
        
        for i in range(num_requests):
            log_entry = {
                'timestamp': self.fake.date_time_between(start_date='-7d', end_date='now'),
                'request_id': str(uuid.uuid4()),
                'endpoint': random.choice(endpoints),
                'method': random.choice(methods),
                'status_code': random.choices(status_codes, weights=status_weights)[0],
                'response_time_ms': random.randint(50, 2000),
                'user_agent': self.fake.user_agent(),
                'ip_address': self.fake.ipv4(),
                'user_id': f"USER_{random.randint(1, 1000):06d}" if random.random() > 0.3 else None,
                'request_size_bytes': random.randint(100, 5000),
                'response_size_bytes': random.randint(500, 50000)
            }
            api_logs.append(log_entry)
        
        df_logs = pd.DataFrame(api_logs)
        df_logs.to_json('data/raw/api_logs/api_access_logs.json', orient='records', date_format='iso')
        
        print(f"Generated {len(df_logs)} API log entries")
        return df_logs
    
    def generate_all_data(self):
        """Generate complete dataset from all source systems"""
        print("Starting data generation for Product Review Analysis...")
        print("=" * 60)
        
        # Generate users (simulating customer database)
        print("1. Generating user data...")
        users_df = self.generate_users()
        
        # Generate products (simulating product catalog)
        print("2. Generating product catalog...")
        products_df = self.generate_products()
        
        # Generate reviews (simulating review system)
        print("3. Generating product reviews...")
        reviews_df = self.generate_reviews(users_df, products_df)
        
        # Generate API logs (simulating API gateway)
        print("4. Generating API access logs...")
        api_logs_df = self.generate_api_logs()
        
        print("=" * 60)
        print("Data generation completed successfully!")
        print(f"Generated:")
        print(f"  - {len(users_df)} users")
        print(f"  - {len(products_df)} products") 
        print(f"  - {len(reviews_df)} reviews")
        print(f"  - {len(api_logs_df)} API log entries")
        print(f"\nData saved to 'data/raw/' directory in multiple formats")
        
        return {
            'users': users_df,
            'products': products_df,
            'reviews': reviews_df,
            'api_logs': api_logs_df
        }


if __name__ == "__main__":
    # Generate sample data
    generator = ReviewDataGenerator()
    data = generator.generate_all_data()