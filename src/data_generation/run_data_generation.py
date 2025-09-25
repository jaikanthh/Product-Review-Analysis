"""
Unified Data Generation Runner
Orchestrates all data generation components for the Product Review Analysis project
Demonstrates comprehensive data source simulation across different systems
"""

import asyncio
import threading
import time
import logging
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Import all data generation modules
from generate_reviews import ReviewDataGenerator
from api_simulator import EcommerceAPISimulator, APIDataCollector
from database_simulator import SQLiteSimulator, DatabaseExtractor
from file_simulator import FileDataSimulator, FileDataProcessor
from streaming_simulator import StreamingSimulationManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataGenerationOrchestrator:
    """Orchestrates all data generation components"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Initialize all generators
        self.review_generator = ReviewDataGenerator()
        self.api_simulator = EcommerceAPISimulator()
        self.database_simulator = SQLiteSimulator()
        self.file_simulator = FileDataSimulator()
        self.streaming_manager = StreamingSimulationManager()
        
        # Track generation status
        self.generation_status = {
            'basic_reviews': False,
            'api_simulation': False,
            'database_simulation': False,
            'file_simulation': False,
            'streaming_simulation': False
        }
        
        self.start_time = None
        self.api_server_thread = None
    
    def generate_basic_review_data(self):
        """Generate basic review data using the original generator"""
        logger.info("=== Starting Basic Review Data Generation ===")
        
        try:
            # Generate sample data
            self.review_generator.generate_sample_data()
            
            # Save in multiple formats
            self.review_generator.save_data_multiple_formats()
            
            self.generation_status['basic_reviews'] = True
            logger.info("✅ Basic review data generation completed")
            
        except Exception as e:
            logger.error(f"❌ Error in basic review data generation: {e}")
            raise
    
    def start_api_simulation(self):
        """Start API simulation in a separate thread"""
        logger.info("=== Starting API Simulation ===")
        
        try:
            # Start API server in a separate thread
            def run_api_server():
                self.api_simulator.run(host="127.0.0.1", port=8000)
            
            self.api_server_thread = threading.Thread(target=run_api_server, daemon=True)
            self.api_server_thread.start()
            
            # Wait a moment for server to start
            time.sleep(3)
            
            # Test API endpoints
            collector = APIDataCollector()
            
            # Collect some data to verify API is working
            products = collector.collect_products()
            reviews = collector.collect_reviews()
            
            logger.info(f"API simulation started - Products: {len(products)}, Reviews: {len(reviews)}")
            
            self.generation_status['api_simulation'] = True
            logger.info("✅ API simulation started successfully")
            
        except Exception as e:
            logger.error(f"❌ Error starting API simulation: {e}")
            raise
    
    def generate_database_data(self):
        """Generate database-based data sources"""
        logger.info("=== Starting Database Data Generation ===")
        
        try:
            # Populate SQLite database
            self.database_simulator.populate_sample_data(
                num_users=2000,
                num_products=1000,
                num_orders=5000
            )
            
            # Extract data for demonstration
            extractor = DatabaseExtractor()
            extracted_data = extractor.extract_all_data()
            extractor.save_extracted_data(extracted_data)
            
            self.generation_status['database_simulation'] = True
            logger.info("✅ Database data generation completed")
            
        except Exception as e:
            logger.error(f"❌ Error in database data generation: {e}")
            raise
    
    def generate_file_data(self):
        """Generate file-based data sources"""
        logger.info("=== Starting File Data Generation ===")
        
        try:
            # Generate all types of files
            self.file_simulator.generate_all_files()
            
            # Validate generated files
            processor = FileDataProcessor()
            validation_results = processor.validate_all_files()
            
            # Save validation results
            with open("data/raw/files/validation_results.json", 'w') as f:
                json.dump(validation_results, f, indent=2)
            
            self.generation_status['file_simulation'] = True
            logger.info("✅ File data generation completed")
            
        except Exception as e:
            logger.error(f"❌ Error in file data generation: {e}")
            raise
    
    async def start_streaming_simulation(self, duration_minutes: int = 5):
        """Start streaming simulation for a specified duration"""
        logger.info(f"=== Starting Streaming Simulation for {duration_minutes} minutes ===")
        
        try:
            # Start streaming simulation
            streaming_task = asyncio.create_task(
                self.streaming_manager.start_simulation(
                    events_per_second=3.0,  # Moderate rate for demo
                    enable_websocket=True,
                    enable_consumer=True
                )
            )
            
            # Let it run for specified duration
            await asyncio.sleep(duration_minutes * 60)
            
            # Stop streaming
            await self.streaming_manager.stop_simulation()
            
            self.generation_status['streaming_simulation'] = True
            logger.info("✅ Streaming simulation completed")
            
        except Exception as e:
            logger.error(f"❌ Error in streaming simulation: {e}")
            raise
    
    def generate_comprehensive_summary(self) -> Dict:
        """Generate a comprehensive summary of all generated data"""
        logger.info("=== Generating Comprehensive Data Summary ===")
        
        summary = {
            'generation_timestamp': datetime.now().isoformat(),
            'generation_duration_minutes': (datetime.now() - self.start_time).total_seconds() / 60 if self.start_time else 0,
            'status': self.generation_status,
            'data_sources': {},
            'file_inventory': {},
            'statistics': {}
        }
        
        try:
            # Basic review data summary
            if self.generation_status['basic_reviews']:
                basic_data_path = Path("data/raw/reviews")
                if basic_data_path.exists():
                    files = list(basic_data_path.glob("*"))
                    summary['data_sources']['basic_reviews'] = {
                        'files_generated': len(files),
                        'file_list': [f.name for f in files],
                        'total_size_mb': sum(f.stat().st_size for f in files if f.is_file()) / (1024 * 1024)
                    }
            
            # Database data summary
            if self.generation_status['database_simulation']:
                db_info = self.database_simulator.get_table_info()
                summary['data_sources']['database'] = {
                    'tables': db_info,
                    'database_file': 'data/raw/databases/ecommerce.db'
                }
            
            # File data summary
            if self.generation_status['file_simulation']:
                file_inventory = self.file_simulator.get_file_inventory()
                summary['data_sources']['files'] = file_inventory
            
            # Streaming data summary
            if self.generation_status['streaming_simulation']:
                streaming_path = Path("data/raw/streaming")
                if streaming_path.exists():
                    streaming_files = list(streaming_path.rglob("*.jsonl"))
                    summary['data_sources']['streaming'] = {
                        'event_files': len(streaming_files),
                        'total_events': self._count_streaming_events(streaming_files),
                        'event_types': list(set(f.parent.parent.name for f in streaming_files))
                    }
            
            # API simulation summary
            if self.generation_status['api_simulation']:
                summary['data_sources']['api'] = {
                    'status': 'running',
                    'endpoint': 'http://127.0.0.1:8000',
                    'available_endpoints': [
                        '/api/v1/products',
                        '/api/v1/reviews',
                        '/api/v1/users',
                        '/api/v1/analytics/summary'
                    ]
                }
            
            # Overall statistics
            summary['statistics'] = {
                'total_data_sources': sum(1 for status in self.generation_status.values() if status),
                'estimated_total_records': self._estimate_total_records(),
                'data_formats_generated': ['CSV', 'JSON', 'Parquet', 'XML', 'SQLite', 'JSONL'],
                'storage_systems_simulated': ['File System', 'Relational DB', 'Document DB', 'Streaming', 'API']
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            summary['summary_error'] = str(e)
        
        return summary
    
    def _count_streaming_events(self, files: List[Path]) -> int:
        """Count total events in streaming files"""
        total_events = 0
        for file in files:
            try:
                with open(file, 'r') as f:
                    total_events += sum(1 for line in f if line.strip())
            except Exception:
                continue
        return total_events
    
    def _estimate_total_records(self) -> int:
        """Estimate total number of records generated"""
        total = 0
        
        # Basic reviews
        if self.generation_status['basic_reviews']:
            total += 10000  # Estimated from review generator
        
        # Database records
        if self.generation_status['database_simulation']:
            db_info = self.database_simulator.get_table_info()
            total += sum(db_info.values())
        
        # File records (estimated)
        if self.generation_status['file_simulation']:
            total += 20000  # Estimated from file generator
        
        # Streaming events
        if self.generation_status['streaming_simulation']:
            streaming_path = Path("data/raw/streaming")
            if streaming_path.exists():
                streaming_files = list(streaming_path.rglob("*.jsonl"))
                total += self._count_streaming_events(streaming_files)
        
        return total
    
    def save_summary(self, summary: Dict):
        """Save generation summary to file"""
        summary_path = Path("data/generation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Generation summary saved to {summary_path}")
    
    async def run_complete_generation(self, streaming_duration_minutes: int = 3):
        """Run complete data generation process"""
        logger.info("🚀 Starting Complete Data Generation Process")
        self.start_time = datetime.now()
        
        try:
            # Step 1: Generate basic review data
            self.generate_basic_review_data()
            
            # Step 2: Start API simulation
            self.start_api_simulation()
            
            # Step 3: Generate database data
            self.generate_database_data()
            
            # Step 4: Generate file-based data
            self.generate_file_data()
            
            # Step 5: Run streaming simulation
            await self.start_streaming_simulation(streaming_duration_minutes)
            
            # Step 6: Generate and save summary
            summary = self.generate_comprehensive_summary()
            self.save_summary(summary)
            
            logger.info("🎉 Complete data generation process finished successfully!")
            logger.info(f"📈 Total records generated: {summary['statistics']['estimated_total_records']:,}")
            logger.info(f"⏱️  Total duration: {summary['generation_duration_minutes']:.2f} minutes")
            
            return summary
            
        except Exception as e:
            logger.error(f"💥 Error in complete data generation: {e}")
            raise
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up resources...")
        
        # Stop API server if running
        if self.api_server_thread and self.api_server_thread.is_alive():
            logger.info("Stopping API server...")
            # Note: In a production environment, you'd want a more graceful shutdown
        
        logger.info("✅ Cleanup completed")


async def main():
    """Main function to run the complete data generation"""
    orchestrator = DataGenerationOrchestrator()
    
    try:
        # Run complete generation process
        summary = await orchestrator.run_complete_generation(streaming_duration_minutes=2)
        
        # Print final summary
        print("\n" + "="*80)
        print("🎯 DATA GENERATION COMPLETE!")
        print("="*80)
        print(f"📊 Total Data Sources: {summary['statistics']['total_data_sources']}")
        print(f"📈 Total Records: {summary['statistics']['estimated_total_records']:,}")
        print(f"⏱️  Duration: {summary['generation_duration_minutes']:.2f} minutes")
        print(f"💾 Data Formats: {', '.join(summary['statistics']['data_formats_generated'])}")
        print(f"🏗️  Storage Systems: {', '.join(summary['statistics']['storage_systems_simulated'])}")
        print("\n📁 Generated Data Locations:")
        for source, details in summary['data_sources'].items():
            print(f"  • {source.title()}: {details}")
        print("\n🔗 API Endpoints (if running):")
        print("  • Main API: http://127.0.0.1:8000")
        print("  • WebSocket: ws://127.0.0.1:8765")
        print("  • Health Check: http://127.0.0.1:8000/api/v1/health")
        print("\n📋 Summary saved to: data/generation_summary.json")
        print("="*80)
        
    except KeyboardInterrupt:
        logger.info("⚠️  Generation interrupted by user")
    except Exception as e:
        logger.error(f"💥 Error in data generation: {e}")
    finally:
        orchestrator.cleanup()


if __name__ == "__main__":
    # Run the complete data generation process
    asyncio.run(main())