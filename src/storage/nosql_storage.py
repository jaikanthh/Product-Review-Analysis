"""
NoSQL Document Storage System
Implements flexible document-based storage using MongoDB
Provides schema-less design, horizontal scaling, and complex data structures
"""

import pymongo
from pymongo import MongoClient, ASCENDING, DESCENDING
import pandas as pd
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
import json
import yaml
from bson import ObjectId
import gridfs

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DocumentSchema:
    """Defines document schema for MongoDB collections"""
    collection_name: str
    indexes: List[Dict[str, Any]]
    validation_schema: Optional[Dict] = None
    sharding_key: Optional[str] = None
    ttl_field: Optional[str] = None
    ttl_seconds: Optional[int] = None


class MongoDBStorage:
    """MongoDB storage implementation for document-based data"""
    
    def __init__(self, connection_string: str = "mongodb://localhost:27017/", database_name: str = "reviews_analytics"):
        self.connection_string = connection_string
        self.database_name = database_name
        self.client = None
        self.database = None
        self.gridfs = None
        
        # Document schemas
        self.schemas = self._define_schemas()
        
        # Initialize connection
        self._connect()
        self._initialize_collections()
    
    def _define_schemas(self) -> Dict[str, DocumentSchema]:
        """Define MongoDB collection schemas"""
        return {
            'users': DocumentSchema(
                collection_name='users',
                indexes=[
                    {'keys': [('user_id', ASCENDING)], 'unique': True},
                    {'keys': [('email', ASCENDING)], 'unique': True},
                    {'keys': [('username', ASCENDING)]},
                    {'keys': [('location', ASCENDING)]},
                    {'keys': [('registration_date', DESCENDING)]},
                    {'keys': [('preferences.categories', ASCENDING)]},  # Nested field index
                ],
                validation_schema={
                    '$jsonSchema': {
                        'bsonType': 'object',
                        'required': ['user_id', 'username', 'email'],
                        'properties': {
                            'user_id': {'bsonType': 'int'},
                            'username': {'bsonType': 'string'},
                            'email': {'bsonType': 'string'},
                            'registration_date': {'bsonType': 'date'},
                            'location': {'bsonType': 'string'},
                            'preferences': {
                                'bsonType': 'object',
                                'properties': {
                                    'categories': {'bsonType': 'array'},
                                    'price_range': {'bsonType': 'object'},
                                    'notifications': {'bsonType': 'bool'}
                                }
                            }
                        }
                    }
                }
            ),
            
            'products': DocumentSchema(
                collection_name='products',
                indexes=[
                    {'keys': [('product_id', ASCENDING)], 'unique': True},
                    {'keys': [('category', ASCENDING)]},
                    {'keys': [('brand', ASCENDING)]},
                    {'keys': [('price', ASCENDING)]},
                    {'keys': [('tags', ASCENDING)]},
                    {'keys': [('specifications.weight', ASCENDING)]},
                    {'keys': [('name', 'text'), ('description', 'text')]},  # Text index for search
                ],
                validation_schema={
                    '$jsonSchema': {
                        'bsonType': 'object',
                        'required': ['product_id', 'name', 'category'],
                        'properties': {
                            'product_id': {'bsonType': 'int'},
                            'name': {'bsonType': 'string'},
                            'category': {'bsonType': 'string'},
                            'price': {'bsonType': 'number'},
                            'specifications': {'bsonType': 'object'},
                            'tags': {'bsonType': 'array'},
                            'images': {'bsonType': 'array'}
                        }
                    }
                }
            ),
            
            'reviews': DocumentSchema(
                collection_name='reviews',
                indexes=[
                    {'keys': [('review_id', ASCENDING)], 'unique': True},
                    {'keys': [('user_id', ASCENDING)]},
                    {'keys': [('product_id', ASCENDING)]},
                    {'keys': [('rating', DESCENDING)]},
                    {'keys': [('review_date', DESCENDING)]},
                    {'keys': [('sentiment.label', ASCENDING)]},
                    {'keys': [('verified_purchase', ASCENDING)]},
                    {'keys': [('review_text', 'text')]},  # Text index for search
                    {'keys': [('user_id', ASCENDING), ('product_id', ASCENDING)]},  # Compound index
                ],
                validation_schema={
                    '$jsonSchema': {
                        'bsonType': 'object',
                        'required': ['review_id', 'user_id', 'product_id', 'rating', 'review_date'],
                        'properties': {
                            'review_id': {'bsonType': 'int'},
                            'user_id': {'bsonType': 'int'},
                            'product_id': {'bsonType': 'int'},
                            'rating': {'bsonType': 'int', 'minimum': 1, 'maximum': 5},
                            'review_date': {'bsonType': 'date'},
                            'sentiment': {
                                'bsonType': 'object',
                                'properties': {
                                    'score': {'bsonType': 'number'},
                                    'label': {'bsonType': 'string'},
                                    'confidence': {'bsonType': 'number'}
                                }
                            }
                        }
                    }
                }
            ),
            
            'user_sessions': DocumentSchema(
                collection_name='user_sessions',
                indexes=[
                    {'keys': [('session_id', ASCENDING)], 'unique': True},
                    {'keys': [('user_id', ASCENDING)]},
                    {'keys': [('start_time', DESCENDING)]},
                    {'keys': [('activities.timestamp', DESCENDING)]},
                ],
                ttl_field='end_time',
                ttl_seconds=30 * 24 * 3600  # 30 days TTL
            ),
            
            'product_analytics': DocumentSchema(
                collection_name='product_analytics',
                indexes=[
                    {'keys': [('product_id', ASCENDING), ('date', DESCENDING)], 'unique': True},
                    {'keys': [('date', DESCENDING)]},
                    {'keys': [('metrics.total_views', DESCENDING)]},
                    {'keys': [('metrics.conversion_rate', DESCENDING)]},
                ]
            ),
            
            'real_time_events': DocumentSchema(
                collection_name='real_time_events',
                indexes=[
                    {'keys': [('timestamp', DESCENDING)]},
                    {'keys': [('event_type', ASCENDING)]},
                    {'keys': [('user_id', ASCENDING)]},
                    {'keys': [('product_id', ASCENDING)]},
                ],
                ttl_field='timestamp',
                ttl_seconds=7 * 24 * 3600  # 7 days TTL for real-time events
            )
        }
    
    def _connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(self.connection_string)
            self.database = self.client[self.database_name]
            self.gridfs = gridfs.GridFS(self.database)
            
            # Test connection
            self.client.admin.command('ping')
            logger.info(f"MongoDB connection established to database: {self.database_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def _initialize_collections(self):
        """Initialize MongoDB collections with schemas and indexes"""
        for schema in self.schemas.values():
            try:
                # Create collection with validation schema
                if schema.validation_schema:
                    self.database.create_collection(
                        schema.collection_name,
                        validator=schema.validation_schema
                    )
                
                collection = self.database[schema.collection_name]
                
                # Create indexes
                for index_spec in schema.indexes:
                    keys = index_spec['keys']
                    options = {k: v for k, v in index_spec.items() if k != 'keys'}
                    collection.create_index(keys, **options)
                
                # Create TTL index if specified
                if schema.ttl_field and schema.ttl_seconds:
                    collection.create_index(
                        [(schema.ttl_field, ASCENDING)],
                        expireAfterSeconds=schema.ttl_seconds
                    )
                
                logger.info(f"Collection '{schema.collection_name}' initialized with {len(schema.indexes)} indexes")
                
            except Exception as e:
                logger.warning(f"Could not fully initialize collection {schema.collection_name}: {e}")
    
    def insert_documents(self, collection_name: str, documents: List[Dict], batch_size: int = 1000) -> List[ObjectId]:
        """Insert documents into collection with batch processing"""
        if not documents:
            return []
        
        collection = self.database[collection_name]
        inserted_ids = []
        
        try:
            # Process in batches
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                # Convert datetime strings to datetime objects
                for doc in batch:
                    self._convert_datetime_fields(doc)
                
                result = collection.insert_many(batch, ordered=False)
                inserted_ids.extend(result.inserted_ids)
            
            logger.info(f"Inserted {len(inserted_ids)} documents into {collection_name}")
            return inserted_ids
            
        except Exception as e:
            logger.error(f"Failed to insert documents into {collection_name}: {e}")
            raise
    
    def _convert_datetime_fields(self, document: Dict):
        """Convert datetime string fields to datetime objects"""
        datetime_fields = ['review_date', 'registration_date', 'timestamp', 'start_time', 'end_time', 'created_at', 'updated_at']
        
        for field in datetime_fields:
            if field in document and isinstance(document[field], str):
                try:
                    document[field] = datetime.fromisoformat(document[field].replace('Z', '+00:00'))
                except:
                    try:
                        document[field] = datetime.strptime(document[field], '%Y-%m-%d')
                    except:
                        pass
    
    def find_documents(self, collection_name: str, query: Dict = None, projection: Dict = None, 
                      sort: List[Tuple] = None, limit: int = None, skip: int = 0) -> List[Dict]:
        """Find documents in collection with advanced querying"""
        collection = self.database[collection_name]
        
        if query is None:
            query = {}
        
        cursor = collection.find(query, projection)
        
        if sort:
            cursor = cursor.sort(sort)
        
        if skip > 0:
            cursor = cursor.skip(skip)
        
        if limit:
            cursor = cursor.limit(limit)
        
        # Convert ObjectId to string for JSON serialization
        results = []
        for doc in cursor:
            if '_id' in doc:
                doc['_id'] = str(doc['_id'])
            results.append(doc)
        
        return results
    
    def aggregate_documents(self, collection_name: str, pipeline: List[Dict]) -> List[Dict]:
        """Execute aggregation pipeline"""
        collection = self.database[collection_name]
        
        try:
            results = list(collection.aggregate(pipeline))
            
            # Convert ObjectId to string
            for doc in results:
                if '_id' in doc and isinstance(doc['_id'], ObjectId):
                    doc['_id'] = str(doc['_id'])
            
            return results
            
        except Exception as e:
            logger.error(f"Aggregation failed on {collection_name}: {e}")
            raise
    
    def update_documents(self, collection_name: str, query: Dict, update: Dict, upsert: bool = False) -> int:
        """Update documents in collection"""
        collection = self.database[collection_name]
        
        try:
            result = collection.update_many(query, update, upsert=upsert)
            logger.info(f"Updated {result.modified_count} documents in {collection_name}")
            return result.modified_count
            
        except Exception as e:
            logger.error(f"Failed to update documents in {collection_name}: {e}")
            raise
    
    def delete_documents(self, collection_name: str, query: Dict) -> int:
        """Delete documents from collection"""
        collection = self.database[collection_name]
        
        try:
            result = collection.delete_many(query)
            logger.info(f"Deleted {result.deleted_count} documents from {collection_name}")
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete documents from {collection_name}: {e}")
            raise
    
    def create_text_search_index(self, collection_name: str, fields: List[str]):
        """Create text search index for full-text search"""
        collection = self.database[collection_name]
        
        index_spec = [(field, 'text') for field in fields]
        
        try:
            collection.create_index(index_spec)
            logger.info(f"Created text search index on {collection_name} for fields: {fields}")
        except Exception as e:
            logger.warning(f"Could not create text search index: {e}")
    
    def text_search(self, collection_name: str, search_text: str, limit: int = 10) -> List[Dict]:
        """Perform full-text search"""
        collection = self.database[collection_name]
        
        try:
            results = list(collection.find(
                {'$text': {'$search': search_text}},
                {'score': {'$meta': 'textScore'}}
            ).sort([('score', {'$meta': 'textScore'})]).limit(limit))
            
            # Convert ObjectId to string
            for doc in results:
                if '_id' in doc:
                    doc['_id'] = str(doc['_id'])
            
            return results
            
        except Exception as e:
            logger.error(f"Text search failed on {collection_name}: {e}")
            return []
    
    def get_collection_stats(self, collection_name: str) -> Dict:
        """Get statistics for a collection"""
        try:
            stats = self.database.command('collStats', collection_name)
            
            return {
                'document_count': stats.get('count', 0),
                'size_bytes': stats.get('size', 0),
                'storage_size_bytes': stats.get('storageSize', 0),
                'index_count': stats.get('nindexes', 0),
                'index_size_bytes': stats.get('totalIndexSize', 0),
                'average_document_size': stats.get('avgObjSize', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats for {collection_name}: {e}")
            return {}
    
    def create_analytical_views(self):
        """Create analytical aggregation pipelines as views"""
        # Product review analytics
        product_analytics_pipeline = [
            {
                '$group': {
                    '_id': '$product_id',
                    'total_reviews': {'$sum': 1},
                    'average_rating': {'$avg': '$rating'},
                    'sentiment_distribution': {
                        '$push': '$sentiment.label'
                    },
                    'latest_review': {'$max': '$review_date'},
                    'verified_reviews': {
                        '$sum': {'$cond': ['$verified_purchase', 1, 0]}
                    }
                }
            },
            {
                '$addFields': {
                    'sentiment_counts': {
                        'positive': {
                            '$size': {
                                '$filter': {
                                    'input': '$sentiment_distribution',
                                    'cond': {'$eq': ['$$this', 'positive']}
                                }
                            }
                        },
                        'negative': {
                            '$size': {
                                '$filter': {
                                    'input': '$sentiment_distribution',
                                    'cond': {'$eq': ['$$this', 'negative']}
                                }
                            }
                        },
                        'neutral': {
                            '$size': {
                                '$filter': {
                                    'input': '$sentiment_distribution',
                                    'cond': {'$eq': ['$$this', 'neutral']}
                                }
                            }
                        }
                    }
                }
            },
            {
                '$project': {
                    'sentiment_distribution': 0  # Remove the array field
                }
            }
        ]
        
        # User engagement analytics
        user_analytics_pipeline = [
            {
                '$group': {
                    '_id': '$user_id',
                    'total_reviews': {'$sum': 1},
                    'average_rating_given': {'$avg': '$rating'},
                    'review_dates': {'$push': '$review_date'},
                    'products_reviewed': {'$addToSet': '$product_id'},
                    'sentiment_given': {'$push': '$sentiment.label'}
                }
            },
            {
                '$addFields': {
                    'products_reviewed_count': {'$size': '$products_reviewed'},
                    'review_frequency_days': {
                        '$divide': [
                            {'$subtract': [{'$max': '$review_dates'}, {'$min': '$review_dates'}]},
                            86400000  # Convert milliseconds to days
                        ]
                    }
                }
            }
        ]
        
        # Store pipelines for later use
        self.analytical_pipelines = {
            'product_analytics': product_analytics_pipeline,
            'user_analytics': user_analytics_pipeline
        }
        
        logger.info("Analytical aggregation pipelines created")
    
    def run_analytics(self, analytics_type: str) -> List[Dict]:
        """Run predefined analytical aggregations"""
        if not hasattr(self, 'analytical_pipelines'):
            self.create_analytical_views()
        
        if analytics_type == 'product_analytics':
            return self.aggregate_documents('reviews', self.analytical_pipelines['product_analytics'])
        elif analytics_type == 'user_analytics':
            return self.aggregate_documents('reviews', self.analytical_pipelines['user_analytics'])
        else:
            raise ValueError(f"Unknown analytics type: {analytics_type}")
    
    def store_file(self, file_path: str, filename: str = None, metadata: Dict = None) -> ObjectId:
        """Store large files using GridFS"""
        if filename is None:
            filename = Path(file_path).name
        
        try:
            with open(file_path, 'rb') as f:
                file_id = self.gridfs.put(f, filename=filename, metadata=metadata)
            
            logger.info(f"File {filename} stored in GridFS with ID: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"Failed to store file {file_path}: {e}")
            raise
    
    def retrieve_file(self, file_id: ObjectId, output_path: str):
        """Retrieve file from GridFS"""
        try:
            file_data = self.gridfs.get(file_id)
            
            with open(output_path, 'wb') as f:
                f.write(file_data.read())
            
            logger.info(f"File retrieved from GridFS to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to retrieve file {file_id}: {e}")
            raise
    
    def export_to_dataframe(self, collection_name: str, query: Dict = None, limit: int = None) -> pd.DataFrame:
        """Export collection data to pandas DataFrame"""
        documents = self.find_documents(collection_name, query, limit=limit)
        
        if documents:
            return pd.DataFrame(documents)
        else:
            return pd.DataFrame()
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")


def main():
    """Demonstrate NoSQL storage systems"""
    logger.info("📄 Demonstrating NoSQL Document Storage Systems")
    
    try:
        # Initialize MongoDB storage
        mongo_storage = MongoDBStorage()
        
        # Create sample data with complex nested structures
        sample_users = [
            {
                'user_id': 1,
                'username': 'john_doe',
                'email': 'john@example.com',
                'registration_date': '2024-01-15',
                'location': 'New York',
                'preferences': {
                    'categories': ['Electronics', 'Books'],
                    'price_range': {'min': 10, 'max': 500},
                    'notifications': True
                },
                'profile': {
                    'age_group': '25-34',
                    'interests': ['technology', 'reading'],
                    'social_links': {
                        'twitter': '@johndoe',
                        'linkedin': 'john-doe-123'
                    }
                }
            },
            {
                'user_id': 2,
                'username': 'jane_smith',
                'email': 'jane@example.com',
                'registration_date': '2024-02-20',
                'location': 'California',
                'preferences': {
                    'categories': ['Fashion', 'Sports'],
                    'price_range': {'min': 20, 'max': 300},
                    'notifications': False
                },
                'profile': {
                    'age_group': '35-44',
                    'interests': ['fitness', 'fashion'],
                    'social_links': {
                        'instagram': '@janesmith'
                    }
                }
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
                'description': 'High-quality wireless headphones with noise cancellation',
                'specifications': {
                    'weight': '250g',
                    'battery_life': '30 hours',
                    'connectivity': ['Bluetooth 5.0', 'USB-C'],
                    'features': ['Noise Cancellation', 'Voice Assistant', 'Touch Controls']
                },
                'tags': ['wireless', 'bluetooth', 'noise-cancelling', 'premium'],
                'images': [
                    {'url': 'https://example.com/img1.jpg', 'type': 'main'},
                    {'url': 'https://example.com/img2.jpg', 'type': 'detail'}
                ],
                'inventory': {
                    'stock': 150,
                    'warehouse_locations': ['NY', 'CA', 'TX']
                }
            },
            {
                'product_id': 2,
                'name': 'Running Shoes',
                'category': 'Sports',
                'subcategory': 'Footwear',
                'brand': 'SportsBrand',
                'price': 129.99,
                'description': 'Comfortable running shoes for daily training',
                'specifications': {
                    'weight': '280g',
                    'materials': ['Mesh', 'Rubber', 'EVA Foam'],
                    'sizes': ['7', '8', '9', '10', '11', '12'],
                    'colors': ['Black', 'White', 'Blue', 'Red']
                },
                'tags': ['running', 'comfortable', 'breathable', 'durable'],
                'images': [
                    {'url': 'https://example.com/shoe1.jpg', 'type': 'main'},
                    {'url': 'https://example.com/shoe2.jpg', 'type': 'side'}
                ],
                'inventory': {
                    'stock': 200,
                    'warehouse_locations': ['CA', 'FL']
                }
            }
        ]
        
        sample_reviews = [
            {
                'review_id': 1,
                'user_id': 1,
                'product_id': 1,
                'rating': 5,
                'review_text': 'Excellent headphones! Great sound quality and comfortable to wear for long periods.',
                'review_date': '2024-03-01',
                'verified_purchase': True,
                'helpful_votes': 15,
                'sentiment': {
                    'score': 0.85,
                    'label': 'positive',
                    'confidence': 0.92
                },
                'review_details': {
                    'pros': ['Great sound quality', 'Comfortable', 'Long battery life'],
                    'cons': ['Slightly expensive'],
                    'usage_context': 'Daily commute and work from home'
                },
                'metadata': {
                    'device_used': 'mobile',
                    'review_length': 'medium',
                    'language': 'en'
                }
            },
            {
                'review_id': 2,
                'user_id': 2,
                'product_id': 2,
                'rating': 4,
                'review_text': 'Good shoes for running, but could be more durable. Comfortable fit though.',
                'review_date': '2024-03-02',
                'verified_purchase': True,
                'helpful_votes': 8,
                'sentiment': {
                    'score': 0.65,
                    'label': 'positive',
                    'confidence': 0.78
                },
                'review_details': {
                    'pros': ['Comfortable fit', 'Good for running'],
                    'cons': ['Durability concerns'],
                    'usage_context': 'Daily running, 5km average'
                },
                'metadata': {
                    'device_used': 'desktop',
                    'review_length': 'short',
                    'language': 'en'
                }
            }
        ]
        
        # Insert documents
        mongo_storage.insert_documents('users', sample_users)
        mongo_storage.insert_documents('products', sample_products)
        mongo_storage.insert_documents('reviews', sample_reviews)
        
        # Create analytical views
        mongo_storage.create_analytical_views()
        
        # Demonstrate complex queries
        print("\n🔍 Complex MongoDB Queries:")
        
        # 1. Find products with specific features
        tech_products = mongo_storage.find_documents(
            'products',
            {'specifications.features': {'$in': ['Noise Cancellation']}},
            {'name': 1, 'price': 1, 'specifications.features': 1}
        )
        print(f"  • Products with Noise Cancellation: {len(tech_products)}")
        
        # 2. Find users with specific preferences
        electronics_users = mongo_storage.find_documents(
            'users',
            {'preferences.categories': 'Electronics'},
            {'username': 1, 'preferences.categories': 1}
        )
        print(f"  • Users interested in Electronics: {len(electronics_users)}")
        
        # 3. Find high-rated reviews with sentiment analysis
        positive_reviews = mongo_storage.find_documents(
            'reviews',
            {
                'rating': {'$gte': 4},
                'sentiment.label': 'positive',
                'sentiment.confidence': {'$gte': 0.8}
            },
            sort=[('sentiment.score', -1)]
        )
        print(f"  • High-confidence positive reviews: {len(positive_reviews)}")
        
        # Demonstrate aggregation analytics
        print("\n📊 Aggregation Analytics:")
        
        product_analytics = mongo_storage.run_analytics('product_analytics')
        print(f"  • Product analytics computed for {len(product_analytics)} products")
        
        for product in product_analytics:
            print(f"    - Product {product['_id']}: {product['total_reviews']} reviews, "
                  f"avg rating: {product['average_rating']:.2f}")
        
        user_analytics = mongo_storage.run_analytics('user_analytics')
        print(f"  • User analytics computed for {len(user_analytics)} users")
        
        # Demonstrate text search
        print("\n🔎 Text Search:")
        search_results = mongo_storage.text_search('reviews', 'comfortable great', limit=5)
        print(f"  • Text search results: {len(search_results)} reviews found")
        
        # Get collection statistics
        print("\n📈 Collection Statistics:")
        for collection_name in ['users', 'products', 'reviews']:
            stats = mongo_storage.get_collection_stats(collection_name)
            print(f"  • {collection_name}: {stats.get('document_count', 0)} documents, "
                  f"{stats.get('size_bytes', 0)} bytes")
        
        # Export to DataFrame
        reviews_df = mongo_storage.export_to_dataframe('reviews')
        print(f"\n📋 Reviews DataFrame shape: {reviews_df.shape}")
        
        mongo_storage.close()
        
        print("\n✅ NoSQL storage demonstration completed!")
        
    except Exception as e:
        logger.error(f"MongoDB demonstration failed: {e}")
        print(f"\n❌ MongoDB demonstration failed: {e}")
        print("Note: Make sure MongoDB is running locally or update connection string")


if __name__ == "__main__":
    main()