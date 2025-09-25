"""
Streaming Data Simulator for Product Review Analysis
Simulates real-time streaming data sources (Kafka-like, WebSocket, etc.)
Demonstrates streaming data generation and real-time processing patterns
"""

import asyncio
import websockets
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, AsyncGenerator
import threading
import queue
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StreamingEvent:
    """Base class for streaming events"""
    event_id: str
    event_type: str
    timestamp: str
    source: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ReviewEvent(StreamingEvent):
    """Real-time review submission event"""
    user_id: str
    product_id: str
    rating: int
    review_text: str
    verified_purchase: bool
    device_type: str
    location: str


@dataclass
class UserActivityEvent(StreamingEvent):
    """User activity tracking event"""
    user_id: str
    session_id: str
    action_type: str
    page_url: str
    duration_seconds: int
    device_info: Dict
    referrer: str


@dataclass
class ProductViewEvent(StreamingEvent):
    """Product view tracking event"""
    user_id: str
    product_id: str
    category: str
    view_duration: int
    search_term: Optional[str]
    price: float
    in_stock: bool


@dataclass
class PurchaseEvent(StreamingEvent):
    """Purchase transaction event"""
    user_id: str
    order_id: str
    product_id: str
    quantity: int
    unit_price: float
    total_amount: float
    payment_method: str
    shipping_address: str


class StreamingDataGenerator:
    """Generates various types of streaming events"""
    
    def __init__(self):
        self.is_running = False
        self.event_queue = queue.Queue()
        self.subscribers = []
        
        # Configuration for event generation
        self.event_rates = {
            'review': 0.1,      # 10% of events
            'user_activity': 0.4,  # 40% of events
            'product_view': 0.35,   # 35% of events
            'purchase': 0.15     # 15% of events
        }
        
        # Sample data for realistic event generation
        self.sample_data = self._load_sample_data()
    
    def _load_sample_data(self) -> Dict:
        """Load sample data for realistic event generation"""
        return {
            'user_ids': [f"USER_{i:06d}" for i in range(1, 10001)],
            'product_ids': [f"PROD_{i:06d}" for i in range(1, 5001)],
            'categories': ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Beauty'],
            'devices': ['mobile', 'desktop', 'tablet'],
            'locations': ['NY', 'CA', 'TX', 'FL', 'IL', 'WA', 'PA', 'OH'],
            'payment_methods': ['credit_card', 'debit_card', 'paypal', 'apple_pay', 'google_pay'],
            'review_texts': [
                "Great product, highly recommend!",
                "Good value for money",
                "Not as expected, poor quality",
                "Excellent customer service",
                "Fast delivery, product as described",
                "Could be better for the price",
                "Amazing quality, will buy again",
                "Disappointed with the purchase"
            ],
            'page_urls': [
                '/home', '/products', '/search', '/cart', '/checkout',
                '/profile', '/orders', '/reviews', '/help', '/contact'
            ],
            'action_types': [
                'page_view', 'product_click', 'search', 'add_to_cart',
                'remove_from_cart', 'checkout_start', 'checkout_complete',
                'review_submit', 'wishlist_add'
            ]
        }
    
    def generate_review_event(self) -> ReviewEvent:
        """Generate a realistic review event"""
        event_id = f"REV_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        return ReviewEvent(
            event_id=event_id,
            event_type="review_submitted",
            timestamp=datetime.now().isoformat(),
            source="review_service",
            user_id=random.choice(self.sample_data['user_ids']),
            product_id=random.choice(self.sample_data['product_ids']),
            rating=random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.10, 0.20, 0.35, 0.30])[0],
            review_text=random.choice(self.sample_data['review_texts']),
            verified_purchase=random.choice([True, False]),
            device_type=random.choice(self.sample_data['devices']),
            location=random.choice(self.sample_data['locations'])
        )
    
    def generate_user_activity_event(self) -> UserActivityEvent:
        """Generate a user activity event"""
        event_id = f"ACT_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        return UserActivityEvent(
            event_id=event_id,
            event_type="user_activity",
            timestamp=datetime.now().isoformat(),
            source="web_analytics",
            user_id=random.choice(self.sample_data['user_ids']),
            session_id=f"sess_{random.randint(100000, 999999)}",
            action_type=random.choice(self.sample_data['action_types']),
            page_url=random.choice(self.sample_data['page_urls']),
            duration_seconds=random.randint(1, 300),
            device_info={
                'type': random.choice(self.sample_data['devices']),
                'os': random.choice(['iOS', 'Android', 'Windows', 'macOS']),
                'browser': random.choice(['Chrome', 'Safari', 'Firefox', 'Edge'])
            },
            referrer=random.choice(['google.com', 'facebook.com', 'direct', 'email'])
        )
    
    def generate_product_view_event(self) -> ProductViewEvent:
        """Generate a product view event"""
        event_id = f"VIEW_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        
        return ProductViewEvent(
            event_id=event_id,
            event_type="product_viewed",
            timestamp=datetime.now().isoformat(),
            source="product_service",
            user_id=random.choice(self.sample_data['user_ids']),
            product_id=random.choice(self.sample_data['product_ids']),
            category=random.choice(self.sample_data['categories']),
            view_duration=random.randint(5, 180),
            search_term=f"search term {random.randint(1, 100)}" if random.random() > 0.6 else None,
            price=round(random.uniform(10.0, 500.0), 2),
            in_stock=random.choice([True, False])
        )
    
    def generate_purchase_event(self) -> PurchaseEvent:
        """Generate a purchase event"""
        event_id = f"PUR_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        quantity = random.randint(1, 5)
        unit_price = round(random.uniform(10.0, 200.0), 2)
        total_amount = round(quantity * unit_price, 2)
        
        return PurchaseEvent(
            event_id=event_id,
            event_type="purchase_completed",
            timestamp=datetime.now().isoformat(),
            source="payment_service",
            user_id=random.choice(self.sample_data['user_ids']),
            order_id=f"ORD_{int(time.time())}_{random.randint(100, 999)}",
            product_id=random.choice(self.sample_data['product_ids']),
            quantity=quantity,
            unit_price=unit_price,
            total_amount=total_amount,
            payment_method=random.choice(self.sample_data['payment_methods']),
            shipping_address=f"Address {random.randint(1, 1000)}"
        )
    
    def generate_random_event(self) -> StreamingEvent:
        """Generate a random event based on configured rates"""
        event_type = random.choices(
            list(self.event_rates.keys()),
            weights=list(self.event_rates.values())
        )[0]
        
        if event_type == 'review':
            return self.generate_review_event()
        elif event_type == 'user_activity':
            return self.generate_user_activity_event()
        elif event_type == 'product_view':
            return self.generate_product_view_event()
        elif event_type == 'purchase':
            return self.generate_purchase_event()
    
    async def start_streaming(self, events_per_second: float = 10.0):
        """Start generating streaming events"""
        self.is_running = True
        logger.info(f"Starting streaming data generation at {events_per_second} events/second")
        
        interval = 1.0 / events_per_second
        
        while self.is_running:
            try:
                event = self.generate_random_event()
                
                # Add to queue for batch processing
                self.event_queue.put(event)
                
                # Notify subscribers (WebSocket clients, etc.)
                await self._notify_subscribers(event)
                
                # Save to file for persistence
                await self._save_event_to_file(event)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error generating streaming event: {e}")
                await asyncio.sleep(1)
    
    async def _notify_subscribers(self, event: StreamingEvent):
        """Notify all subscribers about new event"""
        if self.subscribers:
            event_json = json.dumps(event.to_dict())
            disconnected = []
            
            for websocket in self.subscribers:
                try:
                    await websocket.send(event_json)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.append(websocket)
                except Exception as e:
                    logger.error(f"Error sending event to subscriber: {e}")
                    disconnected.append(websocket)
            
            # Remove disconnected clients
            for ws in disconnected:
                self.subscribers.remove(ws)
    
    async def _save_event_to_file(self, event: StreamingEvent):
        """Save event to file for batch processing simulation"""
        try:
            # Create directory structure by date and event type
            date_str = datetime.now().strftime('%Y%m%d')
            hour_str = datetime.now().strftime('%H')
            
            output_dir = Path(f"data/raw/streaming/{event.event_type}/{date_str}")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to hourly files
            filename = f"{event.event_type}_{date_str}_{hour_str}.jsonl"
            filepath = output_dir / filename
            
            # Append to JSONL file
            with open(filepath, 'a') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
                
        except Exception as e:
            logger.error(f"Error saving event to file: {e}")
    
    def stop_streaming(self):
        """Stop streaming data generation"""
        self.is_running = False
        logger.info("Stopping streaming data generation")
    
    def add_subscriber(self, websocket):
        """Add WebSocket subscriber"""
        self.subscribers.append(websocket)
        logger.info(f"Added subscriber. Total subscribers: {len(self.subscribers)}")
    
    def remove_subscriber(self, websocket):
        """Remove WebSocket subscriber"""
        if websocket in self.subscribers:
            self.subscribers.remove(websocket)
            logger.info(f"Removed subscriber. Total subscribers: {len(self.subscribers)}")


class WebSocketServer:
    """WebSocket server for real-time event streaming"""
    
    def __init__(self, generator: StreamingDataGenerator, host: str = "localhost", port: int = 8765):
        self.generator = generator
        self.host = host
        self.port = port
    
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connections"""
        logger.info(f"New client connected from {websocket.remote_address}")
        
        # Add client to subscribers
        self.generator.add_subscriber(websocket)
        
        try:
            # Send welcome message
            welcome_msg = {
                "type": "welcome",
                "message": "Connected to streaming data feed",
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(welcome_msg))
            
            # Keep connection alive and handle client messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, data)
                except json.JSONDecodeError:
                    error_msg = {
                        "type": "error",
                        "message": "Invalid JSON format",
                        "timestamp": datetime.now().isoformat()
                    }
                    await websocket.send(json.dumps(error_msg))
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {websocket.remote_address} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {websocket.remote_address}: {e}")
        finally:
            # Remove client from subscribers
            self.generator.remove_subscriber(websocket)
    
    async def _handle_client_message(self, websocket, data: Dict):
        """Handle messages from clients"""
        message_type = data.get('type')
        
        if message_type == 'ping':
            pong_msg = {
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(pong_msg))
        
        elif message_type == 'subscribe':
            # Client can subscribe to specific event types
            event_types = data.get('event_types', [])
            response = {
                "type": "subscription_confirmed",
                "event_types": event_types,
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(response))
        
        elif message_type == 'get_stats':
            # Send current statistics
            stats = {
                "type": "stats",
                "active_subscribers": len(self.generator.subscribers),
                "queue_size": self.generator.event_queue.qsize(),
                "is_running": self.generator.is_running,
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send(json.dumps(stats))
    
    async def start_server(self):
        """Start the WebSocket server"""
        logger.info(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        
        async with websockets.serve(self.handle_client, self.host, self.port):
            logger.info("WebSocket server started successfully")
            await asyncio.Future()  # Run forever


class StreamingDataConsumer:
    """Consumes and processes streaming data"""
    
    def __init__(self, generator: StreamingDataGenerator):
        self.generator = generator
        self.processed_events = 0
        self.is_consuming = False
    
    async def start_consuming(self, batch_size: int = 100, batch_interval: float = 5.0):
        """Start consuming events from the queue"""
        self.is_consuming = True
        logger.info(f"Starting event consumer with batch size {batch_size}")
        
        while self.is_consuming:
            try:
                events = []
                
                # Collect events for batch processing
                start_time = time.time()
                while len(events) < batch_size and (time.time() - start_time) < batch_interval:
                    try:
                        event = self.generator.event_queue.get_nowait()
                        events.append(event)
                    except queue.Empty:
                        await asyncio.sleep(0.1)
                
                if events:
                    await self._process_batch(events)
                    self.processed_events += len(events)
                    logger.info(f"Processed batch of {len(events)} events. Total processed: {self.processed_events}")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in event consumer: {e}")
                await asyncio.sleep(5)
    
    async def _process_batch(self, events: List[StreamingEvent]):
        """Process a batch of events"""
        try:
            # Group events by type for different processing
            events_by_type = {}
            for event in events:
                event_type = event.event_type
                if event_type not in events_by_type:
                    events_by_type[event_type] = []
                events_by_type[event_type].append(event)
            
            # Process each event type
            for event_type, type_events in events_by_type.items():
                await self._process_events_by_type(event_type, type_events)
                
        except Exception as e:
            logger.error(f"Error processing event batch: {e}")
    
    async def _process_events_by_type(self, event_type: str, events: List[StreamingEvent]):
        """Process events of a specific type"""
        # Save processed events to different output formats
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as JSON for immediate processing
        output_dir = Path(f"data/processed/streaming/{event_type}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        json_file = output_dir / f"batch_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump([event.to_dict() for event in events], f, indent=2)
        
        # Simulate real-time analytics
        if event_type == "review_submitted":
            await self._process_review_analytics(events)
        elif event_type == "purchase_completed":
            await self._process_purchase_analytics(events)
    
    async def _process_review_analytics(self, events: List[ReviewEvent]):
        """Process review events for real-time analytics"""
        # Calculate real-time metrics
        total_reviews = len(events)
        avg_rating = sum(event.rating for event in events) / total_reviews if total_reviews > 0 else 0
        
        rating_distribution = {}
        for i in range(1, 6):
            rating_distribution[f"rating_{i}"] = len([e for e in events if e.rating == i])
        
        analytics = {
            "timestamp": datetime.now().isoformat(),
            "total_reviews": total_reviews,
            "average_rating": round(avg_rating, 2),
            "rating_distribution": rating_distribution,
            "device_breakdown": {},
            "location_breakdown": {}
        }
        
        # Device and location analytics
        for event in events:
            device = event.device_type
            location = event.location
            
            analytics["device_breakdown"][device] = analytics["device_breakdown"].get(device, 0) + 1
            analytics["location_breakdown"][location] = analytics["location_breakdown"].get(location, 0) + 1
        
        # Save analytics
        analytics_dir = Path("data/processed/analytics/real_time")
        analytics_dir.mkdir(parents=True, exist_ok=True)
        
        analytics_file = analytics_dir / f"review_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analytics_file, 'w') as f:
            json.dump(analytics, f, indent=2)
    
    async def _process_purchase_analytics(self, events: List[PurchaseEvent]):
        """Process purchase events for real-time analytics"""
        total_revenue = sum(event.total_amount for event in events)
        total_orders = len(events)
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
        
        analytics = {
            "timestamp": datetime.now().isoformat(),
            "total_orders": total_orders,
            "total_revenue": round(total_revenue, 2),
            "average_order_value": round(avg_order_value, 2),
            "payment_method_breakdown": {}
        }
        
        # Payment method analytics
        for event in events:
            method = event.payment_method
            analytics["payment_method_breakdown"][method] = analytics["payment_method_breakdown"].get(method, 0) + 1
        
        # Save analytics
        analytics_dir = Path("data/processed/analytics/real_time")
        analytics_dir.mkdir(parents=True, exist_ok=True)
        
        analytics_file = analytics_dir / f"purchase_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analytics_file, 'w') as f:
            json.dump(analytics, f, indent=2)
    
    def stop_consuming(self):
        """Stop consuming events"""
        self.is_consuming = False
        logger.info("Stopping event consumer")


class StreamingSimulationManager:
    """Manages the complete streaming simulation"""
    
    def __init__(self):
        self.generator = StreamingDataGenerator()
        self.consumer = StreamingDataConsumer(self.generator)
        self.websocket_server = WebSocketServer(self.generator)
        self.tasks = []
    
    async def start_simulation(self, 
                             events_per_second: float = 10.0,
                             enable_websocket: bool = True,
                             enable_consumer: bool = True):
        """Start the complete streaming simulation"""
        logger.info("Starting streaming simulation...")
        
        # Start event generation
        generation_task = asyncio.create_task(
            self.generator.start_streaming(events_per_second)
        )
        self.tasks.append(generation_task)
        
        # Start event consumer
        if enable_consumer:
            consumer_task = asyncio.create_task(
                self.consumer.start_consuming()
            )
            self.tasks.append(consumer_task)
        
        # Start WebSocket server
        if enable_websocket:
            websocket_task = asyncio.create_task(
                self.websocket_server.start_server()
            )
            self.tasks.append(websocket_task)
        
        logger.info("Streaming simulation started successfully")
        
        # Wait for all tasks
        try:
            await asyncio.gather(*self.tasks)
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            await self.stop_simulation()
    
    async def stop_simulation(self):
        """Stop the streaming simulation"""
        logger.info("Stopping streaming simulation...")
        
        # Stop components
        self.generator.stop_streaming()
        self.consumer.stop_consuming()
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info("Streaming simulation stopped")


async def main():
    """Main function to run the streaming simulation"""
    manager = StreamingSimulationManager()
    
    try:
        await manager.start_simulation(
            events_per_second=5.0,  # Generate 5 events per second
            enable_websocket=True,
            enable_consumer=True
        )
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Error in simulation: {e}")


if __name__ == "__main__":
    # Run the streaming simulation
    asyncio.run(main())