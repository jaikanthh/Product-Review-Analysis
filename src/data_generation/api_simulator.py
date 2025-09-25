"""
API Simulator for Product Review Analysis
Simulates real-time e-commerce APIs as data sources
Demonstrates API-based data generation and ingestion patterns
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
import pandas as pd
import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import uvicorn
from pydantic import BaseModel
import os


class ReviewRequest(BaseModel):
    """Model for incoming review requests"""
    user_id: str
    product_id: str
    rating: int
    title: str
    review_text: str
    verified_purchase: bool = True


class ProductRequest(BaseModel):
    """Model for product creation requests"""
    name: str
    category: str
    price: float
    description: str
    brand: str


class EcommerceAPISimulator:
    """Simulates e-commerce platform APIs for data generation"""
    
    def __init__(self):
        self.app = FastAPI(
            title="E-commerce API Simulator",
            description="Simulates real-time e-commerce data sources for data engineering pipeline",
            version="1.0.0"
        )
        
        # Load existing data if available
        self.users_data = self._load_data('data/raw/users/users.json', [])
        self.products_data = self._load_data('data/raw/products/products.json', [])
        self.reviews_data = self._load_data('data/raw/reviews/reviews.json', [])
        
        # Setup API routes
        self._setup_routes()
    
    def _load_data(self, file_path: str, default: List) -> List[Dict]:
        """Load data from JSON file or return default"""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
        return default
    
    def _setup_routes(self):
        """Setup all API routes"""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "E-commerce API Simulator",
                "version": "1.0.0",
                "endpoints": {
                    "products": "/api/v1/products",
                    "reviews": "/api/v1/reviews", 
                    "users": "/api/v1/users",
                    "real-time": "/api/v1/stream"
                }
            }
        
        @self.app.get("/api/v1/products")
        async def get_products(
            category: Optional[str] = None,
            limit: int = Query(default=100, le=1000),
            offset: int = Query(default=0, ge=0)
        ):
            """Get products with optional filtering"""
            products = self.products_data.copy()
            
            if category:
                products = [p for p in products if p.get('category', '').lower() == category.lower()]
            
            total = len(products)
            products = products[offset:offset + limit]
            
            return {
                "products": products,
                "total": total,
                "limit": limit,
                "offset": offset,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/api/v1/products/{product_id}")
        async def get_product(product_id: str):
            """Get specific product by ID"""
            product = next((p for p in self.products_data if p['product_id'] == product_id), None)
            if not product:
                raise HTTPException(status_code=404, detail="Product not found")
            return product
        
        @self.app.post("/api/v1/products")
        async def create_product(product: ProductRequest):
            """Create new product (simulates product catalog updates)"""
            new_product = {
                "product_id": f"PROD_{len(self.products_data) + 1:06d}",
                "name": product.name,
                "category": product.category,
                "price": product.price,
                "description": product.description,
                "brand": product.brand,
                "created_at": datetime.now().isoformat(),
                "is_active": True
            }
            
            self.products_data.append(new_product)
            return {"message": "Product created", "product": new_product}
        
        @self.app.get("/api/v1/reviews")
        async def get_reviews(
            product_id: Optional[str] = None,
            user_id: Optional[str] = None,
            rating: Optional[int] = None,
            limit: int = Query(default=100, le=1000),
            offset: int = Query(default=0, ge=0)
        ):
            """Get reviews with optional filtering"""
            reviews = self.reviews_data.copy()
            
            if product_id:
                reviews = [r for r in reviews if r.get('product_id') == product_id]
            if user_id:
                reviews = [r for r in reviews if r.get('user_id') == user_id]
            if rating:
                reviews = [r for r in reviews if r.get('rating') == rating]
            
            total = len(reviews)
            reviews = reviews[offset:offset + limit]
            
            return {
                "reviews": reviews,
                "total": total,
                "limit": limit,
                "offset": offset,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.post("/api/v1/reviews")
        async def create_review(review: ReviewRequest):
            """Create new review (simulates real-time review submission)"""
            # Validate rating
            if review.rating < 1 or review.rating > 5:
                raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
            
            new_review = {
                "review_id": f"REV_{len(self.reviews_data) + 1:08d}",
                "user_id": review.user_id,
                "product_id": review.product_id,
                "rating": review.rating,
                "title": review.title,
                "review_text": review.review_text,
                "verified_purchase": review.verified_purchase,
                "review_date": datetime.now().isoformat(),
                "helpful_votes": 0,
                "total_votes": 0,
                "created_at": datetime.now().isoformat()
            }
            
            self.reviews_data.append(new_review)
            
            # Save to file (simulating real-time data persistence)
            self._save_real_time_review(new_review)
            
            return {"message": "Review created", "review": new_review}
        
        @self.app.get("/api/v1/users")
        async def get_users(
            limit: int = Query(default=100, le=1000),
            offset: int = Query(default=0, ge=0)
        ):
            """Get users (limited data for privacy)"""
            users = []
            for user in self.users_data[offset:offset + limit]:
                # Return limited user data (privacy protection)
                limited_user = {
                    "user_id": user.get("user_id"),
                    "username": user.get("username"),
                    "registration_date": user.get("registration_date"),
                    "is_verified": user.get("is_verified"),
                    "total_orders": user.get("total_orders")
                }
                users.append(limited_user)
            
            return {
                "users": users,
                "total": len(self.users_data),
                "limit": limit,
                "offset": offset,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/api/v1/analytics/summary")
        async def get_analytics_summary():
            """Get analytics summary (simulates business intelligence API)"""
            if not self.reviews_data:
                return {"message": "No data available"}
            
            # Calculate basic analytics
            total_reviews = len(self.reviews_data)
            avg_rating = sum(r.get('rating', 0) for r in self.reviews_data) / total_reviews if total_reviews > 0 else 0
            
            rating_distribution = {}
            for i in range(1, 6):
                rating_distribution[f"rating_{i}"] = len([r for r in self.reviews_data if r.get('rating') == i])
            
            return {
                "total_reviews": total_reviews,
                "average_rating": round(avg_rating, 2),
                "rating_distribution": rating_distribution,
                "total_products": len(self.products_data),
                "total_users": len(self.users_data),
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/api/v1/stream/reviews")
        async def stream_reviews():
            """Simulate real-time review stream"""
            # Generate a random review for streaming simulation
            if not self.products_data or not self.users_data:
                return {"message": "No base data available for streaming"}
            
            random_product = random.choice(self.products_data)
            random_user = random.choice(self.users_data)
            
            streaming_review = {
                "review_id": f"STREAM_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}",
                "user_id": random_user["user_id"],
                "product_id": random_product["product_id"],
                "rating": random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.10, 0.20, 0.35, 0.30])[0],
                "title": "Real-time review",
                "review_text": "This is a simulated real-time review for streaming demonstration.",
                "timestamp": datetime.now().isoformat(),
                "source": "streaming_api"
            }
            
            return streaming_review
        
        @self.app.get("/api/v1/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "data_counts": {
                    "users": len(self.users_data),
                    "products": len(self.products_data),
                    "reviews": len(self.reviews_data)
                }
            }
    
    def _save_real_time_review(self, review: Dict):
        """Save real-time review to file (simulates streaming data persistence)"""
        try:
            os.makedirs('data/raw/reviews/streaming', exist_ok=True)
            
            # Save individual review file (simulates event-based storage)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data/raw/reviews/streaming/review_{timestamp}_{review['review_id']}.json"
            
            with open(filename, 'w') as f:
                json.dump(review, f, indent=2)
                
        except Exception as e:
            print(f"Error saving real-time review: {e}")
    
    def run(self, host: str = "127.0.0.1", port: int = 8000):
        """Run the API simulator"""
        print(f"Starting E-commerce API Simulator on http://{host}:{port}")
        print("Available endpoints:")
        print("  - GET  /api/v1/products")
        print("  - POST /api/v1/products") 
        print("  - GET  /api/v1/reviews")
        print("  - POST /api/v1/reviews")
        print("  - GET  /api/v1/users")
        print("  - GET  /api/v1/analytics/summary")
        print("  - GET  /api/v1/stream/reviews")
        print("  - GET  /api/v1/health")
        
        uvicorn.run(self.app, host=host, port=port)


class APIDataCollector:
    """Collects data from the API simulator (demonstrates API-based ingestion)"""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
    
    def collect_products(self, category: str = None) -> List[Dict]:
        """Collect products from API"""
        import requests
        
        url = f"{self.base_url}/api/v1/products"
        params = {}
        if category:
            params['category'] = category
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()['products']
        except Exception as e:
            print(f"Error collecting products: {e}")
            return []
    
    def collect_reviews(self, product_id: str = None) -> List[Dict]:
        """Collect reviews from API"""
        import requests
        
        url = f"{self.base_url}/api/v1/reviews"
        params = {}
        if product_id:
            params['product_id'] = product_id
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()['reviews']
        except Exception as e:
            print(f"Error collecting reviews: {e}")
            return []
    
    def submit_review(self, review_data: Dict) -> bool:
        """Submit new review via API"""
        import requests
        
        url = f"{self.base_url}/api/v1/reviews"
        
        try:
            response = requests.post(url, json=review_data)
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Error submitting review: {e}")
            return False


if __name__ == "__main__":
    # Run the API simulator
    simulator = EcommerceAPISimulator()
    simulator.run()