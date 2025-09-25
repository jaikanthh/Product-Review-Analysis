"""
Unified Data Ingestion Orchestrator
Manages both batch and streaming ingestion pipelines
Provides comprehensive data ingestion capabilities for the Product Review Analysis platform
"""

import asyncio
import logging
import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import threading
import time

# Import our custom ingestion modules
from .batch_ingestion import BatchIngestionPipeline
from .streaming_ingestion import StreamingIngestionPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IngestionConfig:
    """Configuration for ingestion orchestrator"""
    batch_enabled: bool = True
    streaming_enabled: bool = True
    batch_schedule_interval_hours: int = 6
    streaming_sources: List[str] = None
    batch_sources: List[str] = None
    output_path: str = "data/processed"
    monitoring_enabled: bool = True
    
    def __post_init__(self):
        if self.streaming_sources is None:
            self.streaming_sources = []
        if self.batch_sources is None:
            self.batch_sources = []


class IngestionMonitor:
    """Monitors ingestion pipeline health and performance"""
    
    def __init__(self, output_path: str = "data/processed/monitoring"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'batch_jobs': [],
            'streaming_stats': [],
            'errors': [],
            'performance': []
        }
        
        self.start_time = datetime.now()
    
    def log_batch_job(self, job_info: Dict):
        """Log batch job execution"""
        job_info['timestamp'] = datetime.now().isoformat()
        self.metrics['batch_jobs'].append(job_info)
        logger.info(f"Batch job logged: {job_info.get('job_id', 'unknown')}")
    
    def log_streaming_stats(self, stats: Dict):
        """Log streaming pipeline statistics"""
        stats['timestamp'] = datetime.now().isoformat()
        self.metrics['streaming_stats'].append(stats)
    
    def log_error(self, error_info: Dict):
        """Log error information"""
        error_info['timestamp'] = datetime.now().isoformat()
        self.metrics['errors'].append(error_info)
        logger.error(f"Error logged: {error_info}")
    
    def log_performance(self, perf_info: Dict):
        """Log performance metrics"""
        perf_info['timestamp'] = datetime.now().isoformat()
        self.metrics['performance'].append(perf_info)
    
    def save_metrics(self):
        """Save metrics to file"""
        metrics_file = self.output_path / f"ingestion_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Add summary statistics
        summary = self.get_summary()
        output_data = {
            'summary': summary,
            'detailed_metrics': self.metrics,
            'generated_at': datetime.now().isoformat()
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Metrics saved to {metrics_file}")
    
    def get_summary(self) -> Dict:
        """Get summary of ingestion metrics"""
        runtime = (datetime.now() - self.start_time).total_seconds()
        
        batch_summary = {
            'total_jobs': len(self.metrics['batch_jobs']),
            'successful_jobs': len([j for j in self.metrics['batch_jobs'] if j.get('status') == 'completed']),
            'failed_jobs': len([j for j in self.metrics['batch_jobs'] if j.get('status') == 'failed']),
        }
        
        streaming_summary = {
            'total_snapshots': len(self.metrics['streaming_stats']),
            'latest_stats': self.metrics['streaming_stats'][-1] if self.metrics['streaming_stats'] else {}
        }
        
        error_summary = {
            'total_errors': len(self.metrics['errors']),
            'error_types': list(set(e.get('type', 'unknown') for e in self.metrics['errors']))
        }
        
        return {
            'runtime_seconds': runtime,
            'batch': batch_summary,
            'streaming': streaming_summary,
            'errors': error_summary,
            'performance_snapshots': len(self.metrics['performance'])
        }


class IngestionOrchestrator:
    """Main orchestrator for all ingestion pipelines"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.ingestion_config = self._parse_ingestion_config()
        
        # Initialize components
        self.batch_pipeline = BatchIngestionPipeline(config_path) if self.ingestion_config.batch_enabled else None
        self.streaming_pipeline = StreamingIngestionPipeline(config_path) if self.ingestion_config.streaming_enabled else None
        self.monitor = IngestionMonitor() if self.ingestion_config.monitoring_enabled else None
        
        # Control flags
        self.is_running = False
        self.batch_scheduler_task = None
        self.streaming_task = None
        self.monitoring_task = None
        
        # Statistics
        self.orchestrator_stats = {
            'start_time': None,
            'batch_runs': 0,
            'streaming_uptime': 0,
            'total_data_processed': 0
        }
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return {}
    
    def _parse_ingestion_config(self) -> IngestionConfig:
        """Parse ingestion configuration from main config"""
        ingestion_config = self.config.get('ingestion', {})
        
        return IngestionConfig(
            batch_enabled=ingestion_config.get('batch_enabled', True),
            streaming_enabled=ingestion_config.get('streaming_enabled', True),
            batch_schedule_interval_hours=ingestion_config.get('batch_schedule_interval_hours', 6),
            streaming_sources=ingestion_config.get('streaming_sources', []),
            batch_sources=ingestion_config.get('batch_sources', []),
            output_path=ingestion_config.get('output_path', 'data/processed'),
            monitoring_enabled=ingestion_config.get('monitoring_enabled', True)
        )
    
    async def start_orchestrator(self):
        """Start the complete ingestion orchestrator"""
        self.is_running = True
        self.orchestrator_stats['start_time'] = datetime.now()
        
        logger.info("🚀 Starting Data Ingestion Orchestrator")
        logger.info(f"Batch Enabled: {self.ingestion_config.batch_enabled}")
        logger.info(f"Streaming Enabled: {self.ingestion_config.streaming_enabled}")
        logger.info(f"Monitoring Enabled: {self.ingestion_config.monitoring_enabled}")
        
        tasks = []
        
        # Start batch scheduler
        if self.ingestion_config.batch_enabled and self.batch_pipeline:
            self.batch_scheduler_task = asyncio.create_task(self._run_batch_scheduler())
            tasks.append(self.batch_scheduler_task)
        
        # Start streaming pipeline
        if self.ingestion_config.streaming_enabled and self.streaming_pipeline:
            self.streaming_task = asyncio.create_task(self._run_streaming_pipeline())
            tasks.append(self.streaming_task)
        
        # Start monitoring
        if self.ingestion_config.monitoring_enabled and self.monitor:
            self.monitoring_task = asyncio.create_task(self._run_monitoring())
            tasks.append(self.monitoring_task)
        
        # Wait for all tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _run_batch_scheduler(self):
        """Run batch ingestion on a schedule"""
        logger.info(f"Starting batch scheduler (interval: {self.ingestion_config.batch_schedule_interval_hours}h)")
        
        # Run initial batch
        await self._execute_batch_ingestion()
        
        while self.is_running:
            try:
                # Wait for next scheduled run
                await asyncio.sleep(self.ingestion_config.batch_schedule_interval_hours * 3600)
                
                if self.is_running:
                    await self._execute_batch_ingestion()
                
            except Exception as e:
                logger.error(f"Error in batch scheduler: {e}")
                if self.monitor:
                    self.monitor.log_error({
                        'type': 'batch_scheduler_error',
                        'message': str(e),
                        'component': 'batch_scheduler'
                    })
                await asyncio.sleep(300)  # Wait 5 minutes before retry
        
        logger.info("Batch scheduler stopped")
    
    async def _execute_batch_ingestion(self):
        """Execute a batch ingestion run"""
        logger.info("🔄 Starting batch ingestion run")
        start_time = datetime.now()
        
        try:
            # Create batch jobs based on configuration
            self._create_batch_jobs()
            
            # Execute all batch jobs
            results = self.batch_pipeline.execute_all_jobs(max_workers=4)
            
            # Get summary
            summary = self.batch_pipeline.get_ingestion_summary()
            
            # Update statistics
            self.orchestrator_stats['batch_runs'] += 1
            self.orchestrator_stats['total_data_processed'] += summary['total_records_processed']
            
            # Log to monitor
            if self.monitor:
                self.monitor.log_batch_job({
                    'run_id': f"batch_run_{self.orchestrator_stats['batch_runs']}",
                    'status': 'completed',
                    'duration_seconds': (datetime.now() - start_time).total_seconds(),
                    'jobs_executed': summary['total_jobs'],
                    'records_processed': summary['total_records_processed'],
                    'success_rate': summary['success_rate']
                })
            
            logger.info(f"✅ Batch ingestion completed: {summary['total_records_processed']:,} records processed")
            
        except Exception as e:
            logger.error(f"❌ Batch ingestion failed: {e}")
            if self.monitor:
                self.monitor.log_error({
                    'type': 'batch_ingestion_error',
                    'message': str(e),
                    'component': 'batch_ingestion'
                })
    
    def _create_batch_jobs(self):
        """Create batch ingestion jobs based on configuration"""
        # Default batch sources if not configured
        default_sources = [
            ("file", "data/raw/reviews/reviews_2024.csv", "data/processed/batch/reviews_csv", "csv"),
            ("file", "data/raw/reviews/reviews_2024.json", "data/processed/batch/reviews_json", "json"),
            ("file", "data/raw/files/products.xml", "data/processed/batch/products_xml", "xml"),
            ("database", "data/raw/databases/ecommerce.db", "data/processed/batch/database_tables", "sqlite"),
        ]
        
        # Use configured sources or defaults
        sources = self.ingestion_config.batch_sources if self.ingestion_config.batch_sources else default_sources
        
        for source_type, source_path, target_path, format_type in sources:
            try:
                # Add timestamp to target path to avoid conflicts
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                timestamped_target = f"{target_path}_{timestamp}"
                
                self.batch_pipeline.create_job(source_type, source_path, timestamped_target, format_type)
            except Exception as e:
                logger.warning(f"Could not create batch job for {source_path}: {e}")
    
    async def _run_streaming_pipeline(self):
        """Run streaming ingestion pipeline"""
        logger.info("Starting streaming pipeline")
        
        try:
            # Default streaming sources
            websocket_url = "ws://127.0.0.1:8765"
            api_url = "http://127.0.0.1:8000/api/v1/reviews"
            
            # Start streaming pipeline
            await self.streaming_pipeline.start_pipeline(
                websocket_url=websocket_url,
                api_url=api_url
            )
            
        except Exception as e:
            logger.error(f"Streaming pipeline error: {e}")
            if self.monitor:
                self.monitor.log_error({
                    'type': 'streaming_pipeline_error',
                    'message': str(e),
                    'component': 'streaming_pipeline'
                })
    
    async def _run_monitoring(self):
        """Run monitoring and metrics collection"""
        logger.info("Starting monitoring")
        
        while self.is_running:
            try:
                # Collect batch pipeline stats
                if self.batch_pipeline:
                    batch_summary = self.batch_pipeline.get_ingestion_summary()
                    self.monitor.log_performance({
                        'component': 'batch_pipeline',
                        'metrics': batch_summary
                    })
                
                # Collect streaming pipeline stats
                if self.streaming_pipeline:
                    streaming_stats = self.streaming_pipeline.get_pipeline_stats()
                    self.monitor.log_streaming_stats(streaming_stats)
                
                # Collect orchestrator stats
                orchestrator_runtime = (datetime.now() - self.orchestrator_stats['start_time']).total_seconds()
                self.monitor.log_performance({
                    'component': 'orchestrator',
                    'metrics': {
                        'runtime_seconds': orchestrator_runtime,
                        'batch_runs': self.orchestrator_stats['batch_runs'],
                        'total_data_processed': self.orchestrator_stats['total_data_processed']
                    }
                })
                
                # Save metrics periodically
                self.monitor.save_metrics()
                
                # Wait before next monitoring cycle
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
        
        logger.info("Monitoring stopped")
    
    async def stop_orchestrator(self):
        """Stop the ingestion orchestrator"""
        logger.info("🛑 Stopping Data Ingestion Orchestrator")
        
        self.is_running = False
        
        # Stop streaming pipeline
        if self.streaming_pipeline:
            await self.streaming_pipeline.stop_pipeline()
        
        # Cancel tasks
        tasks_to_cancel = [
            self.batch_scheduler_task,
            self.streaming_task,
            self.monitoring_task
        ]
        
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Save final metrics
        if self.monitor:
            self.monitor.save_metrics()
        
        logger.info("✅ Data Ingestion Orchestrator stopped")
    
    def get_orchestrator_status(self) -> Dict:
        """Get current status of the orchestrator"""
        runtime = (datetime.now() - self.orchestrator_stats['start_time']).total_seconds() if self.orchestrator_stats['start_time'] else 0
        
        status = {
            'is_running': self.is_running,
            'runtime_seconds': runtime,
            'configuration': {
                'batch_enabled': self.ingestion_config.batch_enabled,
                'streaming_enabled': self.ingestion_config.streaming_enabled,
                'monitoring_enabled': self.ingestion_config.monitoring_enabled,
                'batch_interval_hours': self.ingestion_config.batch_schedule_interval_hours
            },
            'statistics': self.orchestrator_stats.copy()
        }
        
        # Add component status
        if self.batch_pipeline:
            status['batch_pipeline'] = self.batch_pipeline.get_ingestion_summary()
        
        if self.streaming_pipeline:
            status['streaming_pipeline'] = self.streaming_pipeline.get_pipeline_stats()
        
        if self.monitor:
            status['monitoring'] = self.monitor.get_summary()
        
        return status


async def main():
    """Main function to demonstrate the ingestion orchestrator"""
    logger.info("🚀 Starting Data Ingestion Orchestrator Demo")
    
    # Initialize orchestrator
    orchestrator = IngestionOrchestrator()
    
    try:
        # Start orchestrator
        orchestrator_task = asyncio.create_task(orchestrator.start_orchestrator())
        
        # Let it run for a demo period (5 minutes)
        await asyncio.sleep(300)
        
        # Stop orchestrator
        await orchestrator.stop_orchestrator()
        
        # Print final status
        status = orchestrator.get_orchestrator_status()
        
        print("\n" + "="*80)
        print("📊 DATA INGESTION ORCHESTRATOR SUMMARY")
        print("="*80)
        print(f"Runtime: {status['runtime_seconds']:.1f} seconds")
        print(f"Batch Runs: {status['statistics']['batch_runs']}")
        print(f"Total Data Processed: {status['statistics']['total_data_processed']:,} records")
        print("\n🔧 Configuration:")
        for key, value in status['configuration'].items():
            print(f"  • {key}: {value}")
        
        if 'batch_pipeline' in status:
            batch_stats = status['batch_pipeline']
            print(f"\n📦 Batch Pipeline:")
            print(f"  • Total Jobs: {batch_stats['total_jobs']}")
            print(f"  • Success Rate: {batch_stats['success_rate']:.1%}")
            print(f"  • Records Processed: {batch_stats['total_records_processed']:,}")
        
        if 'streaming_pipeline' in status:
            streaming_stats = status['streaming_pipeline']
            print(f"\n🌊 Streaming Pipeline:")
            print(f"  • Events Processed: {streaming_stats['processing']['events_processed']:,}")
            print(f"  • Events per Second: {streaming_stats['events_per_second']:.2f}")
            print(f"  • Windows Processed: {streaming_stats['processing']['windows_processed']}")
        
        if 'monitoring' in status:
            monitoring_stats = status['monitoring']
            print(f"\n📈 Monitoring:")
            print(f"  • Total Errors: {monitoring_stats['errors']['total_errors']}")
            print(f"  • Performance Snapshots: {monitoring_stats['performance_snapshots']}")
        
        print("="*80)
        
    except KeyboardInterrupt:
        logger.info("⚠️  Orchestrator interrupted by user")
        await orchestrator.stop_orchestrator()
    except Exception as e:
        logger.error(f"💥 Error in orchestrator: {e}")
        await orchestrator.stop_orchestrator()


if __name__ == "__main__":
    asyncio.run(main())