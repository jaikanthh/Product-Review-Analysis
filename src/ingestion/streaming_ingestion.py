"""
Streaming Data Ingestion Pipeline
Handles real-time data ingestion from various streaming sources
Implements buffering, windowing, and real-time processing capabilities
"""

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque
import logging
import yaml
from pathlib import Path
import threading
import time
import queue
import hashlib
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import aiohttp
from textblob import TextBlob
import sqlite3

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StreamingEvent:
    """Represents a single streaming event"""
    event_id: str
    event_type: str
    timestamp: datetime
    data: Dict[str, Any]
    source: str
    processed: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'event_id': self.event_id,
            'event_type': self.event_type,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'source': self.source,
            'processed': self.processed
        }


@dataclass
class StreamingWindow:
    """Represents a time-based window for stream processing"""
    window_id: str
    start_time: datetime
    end_time: datetime
    events: List[StreamingEvent] = field(default_factory=list)
    processed: bool = False
    
    def add_event(self, event: StreamingEvent):
        """Add event to window if it falls within time range"""
        if self.start_time <= event.timestamp <= self.end_time:
            self.events.append(event)
            return True
        return False
    
    def get_event_count(self) -> int:
        return len(self.events)
    
    def get_events_by_type(self, event_type: str) -> List[StreamingEvent]:
        return [e for e in self.events if e.event_type == event_type]


class StreamingBuffer:
    """Thread-safe buffer for streaming events"""
    
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self._event_count = 0
    
    def add_event(self, event: StreamingEvent):
        """Add event to buffer"""
        with self.lock:
            self.buffer.append(event)
            self._event_count += 1
    
    def get_events(self, count: int = None) -> List[StreamingEvent]:
        """Get events from buffer"""
        with self.lock:
            if count is None:
                events = list(self.buffer)
                self.buffer.clear()
            else:
                events = []
                for _ in range(min(count, len(self.buffer))):
                    if self.buffer:
                        events.append(self.buffer.popleft())
            return events
    
    def get_size(self) -> int:
        with self.lock:
            return len(self.buffer)
    
    def get_total_events(self) -> int:
        return self._event_count


class RealTimeProcessor:
    """Processes streaming events in real-time"""
    
    def __init__(self):
        self.processors = {
            'review': self._process_review_event,
            'user_activity': self._process_user_activity_event,
            'product_view': self._process_product_view_event,
            'purchase': self._process_purchase_event
        }
    
    def process_event(self, event: StreamingEvent) -> Dict[str, Any]:
        """Process a single event and return enriched data"""
        processor = self.processors.get(event.event_type, self._process_generic_event)
        return processor(event)
    
    def _process_review_event(self, event: StreamingEvent) -> Dict[str, Any]:
        """Process review events with sentiment analysis"""
        data = event.data.copy()
        
        # Add sentiment analysis
        if 'review_text' in data:
            blob = TextBlob(data['review_text'])
            data['sentiment_polarity'] = blob.sentiment.polarity
            data['sentiment_subjectivity'] = blob.sentiment.subjectivity
            data['sentiment_label'] = 'positive' if blob.sentiment.polarity > 0.1 else 'negative' if blob.sentiment.polarity < -0.1 else 'neutral'
        
        # Add processing metadata
        data['processed_timestamp'] = datetime.now().isoformat()
        data['processing_latency_ms'] = (datetime.now() - event.timestamp).total_seconds() * 1000
        
        return data
    
    def _process_user_activity_event(self, event: StreamingEvent) -> Dict[str, Any]:
        """Process user activity events"""
        data = event.data.copy()
        
        # Add session tracking
        data['session_duration'] = data.get('session_duration', 0)
        data['activity_score'] = self._calculate_activity_score(data)
        data['processed_timestamp'] = datetime.now().isoformat()
        
        return data
    
    def _process_product_view_event(self, event: StreamingEvent) -> Dict[str, Any]:
        """Process product view events"""
        data = event.data.copy()
        
        # Add view tracking
        data['view_duration'] = data.get('view_duration', 0)
        data['bounce_rate'] = 1 if data['view_duration'] < 5 else 0
        data['processed_timestamp'] = datetime.now().isoformat()
        
        return data
    
    def _process_purchase_event(self, event: StreamingEvent) -> Dict[str, Any]:
        """Process purchase events"""
        data = event.data.copy()
        
        # Add purchase analytics
        data['revenue_impact'] = data.get('amount', 0) * data.get('quantity', 1)
        data['purchase_category'] = self._categorize_purchase(data)
        data['processed_timestamp'] = datetime.now().isoformat()
        
        return data
    
    def _process_generic_event(self, event: StreamingEvent) -> Dict[str, Any]:
        """Process generic events"""
        data = event.data.copy()
        data['processed_timestamp'] = datetime.now().isoformat()
        return data
    
    def _calculate_activity_score(self, data: Dict) -> float:
        """Calculate user activity score"""
        base_score = 1.0
        if 'page_views' in data:
            base_score += data['page_views'] * 0.1
        if 'time_spent' in data:
            base_score += min(data['time_spent'] / 60, 5) * 0.2  # Max 5 minutes
        return min(base_score, 10.0)
    
    def _categorize_purchase(self, data: Dict) -> str:
        """Categorize purchase based on amount"""
        amount = data.get('amount', 0)
        if amount < 25:
            return 'low_value'
        elif amount < 100:
            return 'medium_value'
        else:
            return 'high_value'


class WindowManager:
    """Manages time-based windows for stream processing"""
    
    def __init__(self, window_size_seconds: int = 60, slide_interval_seconds: int = 30):
        self.window_size = timedelta(seconds=window_size_seconds)
        self.slide_interval = timedelta(seconds=slide_interval_seconds)
        self.windows: Dict[str, StreamingWindow] = {}
        self.lock = threading.Lock()
    
    def add_event_to_windows(self, event: StreamingEvent):
        """Add event to appropriate windows"""
        with self.lock:
            current_time = event.timestamp
            
            # Create new windows if needed
            self._create_windows_for_time(current_time)
            
            # Add event to all applicable windows
            for window in self.windows.values():
                if not window.processed:
                    window.add_event(event)
    
    def _create_windows_for_time(self, timestamp: datetime):
        """Create windows that should contain the given timestamp"""
        # Round down to slide interval
        base_time = timestamp.replace(second=0, microsecond=0)
        base_time = base_time.replace(minute=(base_time.minute // (self.slide_interval.seconds // 60)) * (self.slide_interval.seconds // 60))
        
        # Create windows around this time
        for i in range(-2, 3):  # Create 5 windows around current time
            window_start = base_time + (i * self.slide_interval)
            window_end = window_start + self.window_size
            window_id = f"window_{window_start.strftime('%Y%m%d_%H%M%S')}"
            
            if window_id not in self.windows:
                self.windows[window_id] = StreamingWindow(
                    window_id=window_id,
                    start_time=window_start,
                    end_time=window_end
                )
    
    def get_completed_windows(self) -> List[StreamingWindow]:
        """Get windows that are ready for processing"""
        with self.lock:
            current_time = datetime.now()
            completed = []
            
            for window_id, window in list(self.windows.items()):
                if not window.processed and current_time > window.end_time:
                    window.processed = True
                    completed.append(window)
                    # Remove old windows to prevent memory buildup
                    if current_time - window.end_time > timedelta(hours=1):
                        del self.windows[window_id]
            
            return completed
    
    def get_window_stats(self) -> Dict:
        """Get statistics about current windows"""
        with self.lock:
            total_windows = len(self.windows)
            processed_windows = sum(1 for w in self.windows.values() if w.processed)
            total_events = sum(w.get_event_count() for w in self.windows.values())
            
            return {
                'total_windows': total_windows,
                'processed_windows': processed_windows,
                'active_windows': total_windows - processed_windows,
                'total_events_in_windows': total_events
            }


class WebSocketIngestionHandler:
    """Handles WebSocket-based streaming ingestion"""
    
    def __init__(self, buffer: StreamingBuffer, processor: RealTimeProcessor):
        self.buffer = buffer
        self.processor = processor
        self.is_running = False
        self.connection = None
    
    async def connect_and_consume(self, websocket_url: str):
        """Connect to WebSocket and consume events"""
        self.is_running = True
        logger.info(f"Connecting to WebSocket: {websocket_url}")
        
        try:
            async with websockets.connect(websocket_url) as websocket:
                self.connection = websocket
                logger.info("WebSocket connection established")
                
                async for message in websocket:
                    if not self.is_running:
                        break
                    
                    try:
                        # Parse incoming message
                        data = json.loads(message)
                        
                        # Create streaming event
                        event = StreamingEvent(
                            event_id=data.get('event_id', hashlib.md5(message.encode()).hexdigest()[:8]),
                            event_type=data.get('event_type', 'unknown'),
                            timestamp=datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat())),
                            data=data.get('data', data),
                            source='websocket'
                        )
                        
                        # Add to buffer
                        self.buffer.add_event(event)
                        
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing WebSocket message: {e}")
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")
        
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            self.is_running = False
            logger.info("WebSocket connection closed")
    
    def stop(self):
        """Stop WebSocket consumption"""
        self.is_running = False


class APIPollingHandler:
    """Handles API polling for streaming-like ingestion"""
    
    def __init__(self, buffer: StreamingBuffer, processor: RealTimeProcessor):
        self.buffer = buffer
        self.processor = processor
        self.is_running = False
        self.session = None
    
    async def start_polling(self, api_url: str, poll_interval_seconds: int = 5):
        """Start polling API for new data"""
        self.is_running = True
        logger.info(f"Starting API polling: {api_url} (interval: {poll_interval_seconds}s)")
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            last_timestamp = datetime.now() - timedelta(hours=1)
            
            while self.is_running:
                try:
                    # Poll API with timestamp filter
                    params = {
                        'since': last_timestamp.isoformat(),
                        'limit': 100
                    }
                    
                    async with session.get(api_url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            # Process response data
                            records = data if isinstance(data, list) else data.get('data', [])
                            
                            for record in records:
                                event = StreamingEvent(
                                    event_id=record.get('id', hashlib.md5(str(record).encode()).hexdigest()[:8]),
                                    event_type=record.get('type', 'api_data'),
                                    timestamp=datetime.fromisoformat(record.get('timestamp', datetime.now().isoformat())),
                                    data=record,
                                    source='api_polling'
                                )
                                
                                self.buffer.add_event(event)
                                
                                # Update last timestamp
                                if event.timestamp > last_timestamp:
                                    last_timestamp = event.timestamp
                        
                        else:
                            logger.warning(f"API polling returned status {response.status}")
                
                except Exception as e:
                    logger.error(f"Error in API polling: {e}")
                
                # Wait before next poll
                await asyncio.sleep(poll_interval_seconds)
        
        logger.info("API polling stopped")
    
    def stop(self):
        """Stop API polling"""
        self.is_running = False


class StreamingDataSink:
    """Handles output of processed streaming data"""
    
    def __init__(self, output_path: str = "data/processed/streaming"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize output files
        self.raw_events_file = self.output_path / "raw_events.jsonl"
        self.processed_events_file = self.output_path / "processed_events.jsonl"
        self.window_aggregates_file = self.output_path / "window_aggregates.jsonl"
        
        # Initialize database for real-time queries
        self.db_path = self.output_path / "streaming_data.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for real-time data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS streaming_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT,
                timestamp TEXT,
                source TEXT,
                data TEXT,
                processed_data TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS window_aggregates (
                window_id TEXT PRIMARY KEY,
                start_time TEXT,
                end_time TEXT,
                event_count INTEGER,
                event_types TEXT,
                aggregates TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    async def save_raw_event(self, event: StreamingEvent):
        """Save raw event to file"""
        try:
            async with aiofiles.open(self.raw_events_file, 'a') as f:
                await f.write(json.dumps(event.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Error saving raw event: {e}")
    
    async def save_processed_event(self, event: StreamingEvent, processed_data: Dict):
        """Save processed event to file and database"""
        try:
            # Save to file
            output_data = {
                'event_id': event.event_id,
                'event_type': event.event_type,
                'timestamp': event.timestamp.isoformat(),
                'source': event.source,
                'original_data': event.data,
                'processed_data': processed_data
            }
            
            async with aiofiles.open(self.processed_events_file, 'a') as f:
                await f.write(json.dumps(output_data) + '\n')
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO streaming_events 
                (event_id, event_type, timestamp, source, data, processed_data)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.event_type,
                event.timestamp.isoformat(),
                event.source,
                json.dumps(event.data),
                json.dumps(processed_data)
            ))
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving processed event: {e}")
    
    async def save_window_aggregate(self, window: StreamingWindow, aggregates: Dict):
        """Save window aggregates"""
        try:
            output_data = {
                'window_id': window.window_id,
                'start_time': window.start_time.isoformat(),
                'end_time': window.end_time.isoformat(),
                'event_count': window.get_event_count(),
                'event_types': list(set(e.event_type for e in window.events)),
                'aggregates': aggregates
            }
            
            # Save to file
            async with aiofiles.open(self.window_aggregates_file, 'a') as f:
                await f.write(json.dumps(output_data) + '\n')
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO window_aggregates 
                (window_id, start_time, end_time, event_count, event_types, aggregates)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                window.window_id,
                window.start_time.isoformat(),
                window.end_time.isoformat(),
                window.get_event_count(),
                json.dumps(output_data['event_types']),
                json.dumps(aggregates)
            ))
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving window aggregate: {e}")


class StreamingIngestionPipeline:
    """Main streaming ingestion pipeline orchestrator"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.buffer = StreamingBuffer(max_size=50000)
        self.processor = RealTimeProcessor()
        self.window_manager = WindowManager(
            window_size_seconds=self.config.get('streaming', {}).get('window_size_seconds', 60),
            slide_interval_seconds=self.config.get('streaming', {}).get('slide_interval_seconds', 30)
        )
        self.data_sink = StreamingDataSink()
        
        # Initialize handlers
        self.websocket_handler = WebSocketIngestionHandler(self.buffer, self.processor)
        self.api_handler = APIPollingHandler(self.buffer, self.processor)
        
        # Control flags
        self.is_running = False
        self.processing_task = None
        self.window_processing_task = None
        
        # Statistics
        self.stats = {
            'events_received': 0,
            'events_processed': 0,
            'windows_processed': 0,
            'start_time': None
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return {}
    
    async def start_pipeline(self, websocket_url: str = None, api_url: str = None):
        """Start the streaming ingestion pipeline"""
        self.is_running = True
        self.stats['start_time'] = datetime.now()
        
        logger.info("🚀 Starting Streaming Ingestion Pipeline")
        
        # Start processing tasks
        self.processing_task = asyncio.create_task(self._process_events_continuously())
        self.window_processing_task = asyncio.create_task(self._process_windows_continuously())
        
        # Start data sources
        tasks = []
        
        if websocket_url:
            websocket_task = asyncio.create_task(
                self.websocket_handler.connect_and_consume(websocket_url)
            )
            tasks.append(websocket_task)
        
        if api_url:
            api_task = asyncio.create_task(
                self.api_handler.start_polling(api_url, poll_interval_seconds=5)
            )
            tasks.append(api_task)
        
        # Wait for all tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _process_events_continuously(self):
        """Continuously process events from buffer"""
        logger.info("Started continuous event processing")
        
        while self.is_running:
            try:
                # Get events from buffer
                events = self.buffer.get_events(count=100)
                
                if events:
                    for event in events:
                        # Save raw event
                        await self.data_sink.save_raw_event(event)
                        
                        # Process event
                        processed_data = self.processor.process_event(event)
                        
                        # Save processed event
                        await self.data_sink.save_processed_event(event, processed_data)
                        
                        # Add to window manager
                        self.window_manager.add_event_to_windows(event)
                        
                        # Update statistics
                        self.stats['events_received'] += 1
                        self.stats['events_processed'] += 1
                
                # Small delay to prevent busy waiting
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in event processing: {e}")
                await asyncio.sleep(1)
        
        logger.info("Stopped continuous event processing")
    
    async def _process_windows_continuously(self):
        """Continuously process completed windows"""
        logger.info("Started continuous window processing")
        
        while self.is_running:
            try:
                # Get completed windows
                completed_windows = self.window_manager.get_completed_windows()
                
                for window in completed_windows:
                    # Calculate aggregates
                    aggregates = self._calculate_window_aggregates(window)
                    
                    # Save window aggregates
                    await self.data_sink.save_window_aggregate(window, aggregates)
                    
                    # Update statistics
                    self.stats['windows_processed'] += 1
                    
                    logger.info(f"Processed window {window.window_id} with {window.get_event_count()} events")
                
                # Wait before checking for more windows
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in window processing: {e}")
                await asyncio.sleep(5)
        
        logger.info("Stopped continuous window processing")
    
    def _calculate_window_aggregates(self, window: StreamingWindow) -> Dict:
        """Calculate aggregates for a window"""
        events = window.events
        
        if not events:
            return {}
        
        # Basic aggregates
        aggregates = {
            'total_events': len(events),
            'event_types': {},
            'time_range': {
                'start': window.start_time.isoformat(),
                'end': window.end_time.isoformat()
            }
        }
        
        # Count by event type
        for event in events:
            event_type = event.event_type
            if event_type not in aggregates['event_types']:
                aggregates['event_types'][event_type] = 0
            aggregates['event_types'][event_type] += 1
        
        # Event-specific aggregates
        review_events = window.get_events_by_type('review')
        if review_events:
            ratings = [e.data.get('rating', 0) for e in review_events if 'rating' in e.data]
            if ratings:
                aggregates['review_stats'] = {
                    'count': len(review_events),
                    'avg_rating': np.mean(ratings),
                    'min_rating': min(ratings),
                    'max_rating': max(ratings)
                }
        
        purchase_events = window.get_events_by_type('purchase')
        if purchase_events:
            amounts = [e.data.get('amount', 0) for e in purchase_events if 'amount' in e.data]
            if amounts:
                aggregates['purchase_stats'] = {
                    'count': len(purchase_events),
                    'total_revenue': sum(amounts),
                    'avg_order_value': np.mean(amounts),
                    'min_order': min(amounts),
                    'max_order': max(amounts)
                }
        
        return aggregates
    
    async def stop_pipeline(self):
        """Stop the streaming ingestion pipeline"""
        logger.info("🛑 Stopping Streaming Ingestion Pipeline")
        
        self.is_running = False
        
        # Stop handlers
        self.websocket_handler.stop()
        self.api_handler.stop()
        
        # Cancel tasks
        if self.processing_task:
            self.processing_task.cancel()
        if self.window_processing_task:
            self.window_processing_task.cancel()
        
        # Process remaining events
        remaining_events = self.buffer.get_events()
        if remaining_events:
            logger.info(f"Processing {len(remaining_events)} remaining events")
            for event in remaining_events:
                processed_data = self.processor.process_event(event)
                await self.data_sink.save_processed_event(event, processed_data)
        
        logger.info("✅ Streaming ingestion pipeline stopped")
    
    def get_pipeline_stats(self) -> Dict:
        """Get pipeline statistics"""
        current_time = datetime.now()
        runtime = (current_time - self.stats['start_time']).total_seconds() if self.stats['start_time'] else 0
        
        buffer_stats = {
            'buffer_size': self.buffer.get_size(),
            'total_events_received': self.buffer.get_total_events()
        }
        
        window_stats = self.window_manager.get_window_stats()
        
        return {
            'runtime_seconds': runtime,
            'events_per_second': self.stats['events_processed'] / runtime if runtime > 0 else 0,
            'buffer': buffer_stats,
            'windows': window_stats,
            'processing': self.stats,
            'is_running': self.is_running
        }


async def main():
    """Main function to demonstrate streaming ingestion pipeline"""
    logger.info("🚀 Starting Streaming Ingestion Pipeline Demo")
    
    # Initialize pipeline
    pipeline = StreamingIngestionPipeline()
    
    try:
        # Start pipeline with WebSocket source
        websocket_url = "ws://127.0.0.1:8765"
        api_url = "http://127.0.0.1:8000/api/v1/reviews"
        
        # Run for a limited time for demo
        pipeline_task = asyncio.create_task(
            pipeline.start_pipeline(websocket_url=websocket_url, api_url=api_url)
        )
        
        # Let it run for 2 minutes
        await asyncio.sleep(120)
        
        # Stop pipeline
        await pipeline.stop_pipeline()
        
        # Print statistics
        stats = pipeline.get_pipeline_stats()
        
        print("\n" + "="*60)
        print("📊 STREAMING INGESTION PIPELINE SUMMARY")
        print("="*60)
        print(f"Runtime: {stats['runtime_seconds']:.1f} seconds")
        print(f"Events Processed: {stats['processing']['events_processed']:,}")
        print(f"Events per Second: {stats['events_per_second']:.2f}")
        print(f"Windows Processed: {stats['processing']['windows_processed']}")
        print(f"Buffer Size: {stats['buffer']['buffer_size']}")
        print(f"Total Events Received: {stats['buffer']['total_events_received']:,}")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("⚠️  Pipeline interrupted by user")
        await pipeline.stop_pipeline()
    except Exception as e:
        logger.error(f"💥 Error in streaming pipeline: {e}")
        await pipeline.stop_pipeline()


if __name__ == "__main__":
    asyncio.run(main())