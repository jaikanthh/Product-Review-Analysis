"""
Batch Data Ingestion Pipeline
Handles batch processing of data from various sources including files, databases, and APIs
Implements data quality checks, error handling, and monitoring
"""

import pandas as pd
import numpy as np
import json
import sqlite3
import logging
import yaml
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import requests
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow.parquet as pq
import xml.etree.ElementTree as ET

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class IngestionJob:
    """Represents a single ingestion job"""
    job_id: str
    source_type: str
    source_path: str
    target_path: str
    format_type: str
    status: str = "pending"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    records_processed: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class DataQualityChecker:
    """Handles data quality validation for ingested data"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        self.config = self._load_config(config_path)
        self.quality_rules = self.config.get('data_quality', {}).get('validation_rules', {})
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return {}
    
    def validate_dataframe(self, df: pd.DataFrame, data_type: str) -> Tuple[bool, List[str]]:
        """Validate a DataFrame against quality rules"""
        errors = []
        
        # Check for None DataFrame
        if df is None:
            errors.append("DataFrame is None")
            return False, errors
        
        # Basic checks
        if df.empty:
            errors.append("DataFrame is empty")
            return False, errors
        
        # Check for required columns based on data type
        required_columns = self.quality_rules.get(data_type, {}).get('required_columns', [])
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check for null values in critical columns
        critical_columns = self.quality_rules.get(data_type, {}).get('non_null_columns', [])
        for col in critical_columns:
            if col in df.columns and df[col].isnull().any():
                null_count = df[col].isnull().sum()
                errors.append(f"Column '{col}' has {null_count} null values")
        
        # Check data types
        expected_types = self.quality_rules.get(data_type, {}).get('column_types', {})
        for col, expected_type in expected_types.items():
            if col in df.columns:
                if expected_type == 'numeric' and not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column '{col}' should be numeric")
                elif expected_type == 'datetime' and not pd.api.types.is_datetime64_any_dtype(df[col]):
                    errors.append(f"Column '{col}' should be datetime")
        
        # Check value ranges
        value_ranges = self.quality_rules.get(data_type, {}).get('value_ranges', {})
        for col, range_config in value_ranges.items():
            if col in df.columns:
                min_val, max_val = range_config.get('min'), range_config.get('max')
                if min_val is not None and (df[col] < min_val).any():
                    errors.append(f"Column '{col}' has values below minimum {min_val}")
                if max_val is not None and (df[col] > max_val).any():
                    errors.append(f"Column '{col}' has values above maximum {max_val}")
        
        # Check for duplicates
        if self.quality_rules.get(data_type, {}).get('check_duplicates', False):
            duplicate_count = df.duplicated().sum()
            if duplicate_count > 0:
                errors.append(f"Found {duplicate_count} duplicate records")
        
        return len(errors) == 0, errors
    
    def clean_dataframe(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Apply basic cleaning to DataFrame"""
        cleaned_df = df.copy()
        
        # Remove duplicates if configured
        if self.quality_rules.get(data_type, {}).get('remove_duplicates', False):
            initial_count = len(cleaned_df)
            cleaned_df = cleaned_df.drop_duplicates()
            removed_count = initial_count - len(cleaned_df)
            if removed_count > 0:
                logger.info(f"Removed {removed_count} duplicate records")
        
        # Fill null values for specific columns
        fill_rules = self.quality_rules.get(data_type, {}).get('fill_null_values', {})
        for col, fill_value in fill_rules.items():
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna(fill_value)
        
        # Convert data types
        type_conversions = self.quality_rules.get(data_type, {}).get('type_conversions', {})
        for col, target_type in type_conversions.items():
            if col in cleaned_df.columns:
                try:
                    if target_type == 'datetime':
                        cleaned_df[col] = pd.to_datetime(cleaned_df[col])
                    elif target_type == 'numeric':
                        cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                    elif target_type == 'string':
                        cleaned_df[col] = cleaned_df[col].astype(str)
                except Exception as e:
                    logger.warning(f"Could not convert column {col} to {target_type}: {e}")
        
        return cleaned_df


class FileIngestionHandler:
    """Handles ingestion from various file formats"""
    
    def __init__(self, quality_checker: DataQualityChecker):
        self.quality_checker = quality_checker
    
    def ingest_csv(self, file_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """Ingest data from CSV file"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully read CSV file: {file_path} ({len(df)} records)")
            return df, []
        except Exception as e:
            error_msg = f"Error reading CSV file {file_path}: {e}"
            logger.error(error_msg)
            return pd.DataFrame(), [error_msg]
    
    def ingest_json(self, file_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """Ingest data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Try to find the main data array
                for key, value in data.items():
                    if isinstance(value, list) and len(value) > 0:
                        df = pd.DataFrame(value)
                        break
                else:
                    df = pd.DataFrame([data])
            else:
                df = pd.DataFrame([data])
            
            logger.info(f"Successfully read JSON file: {file_path} ({len(df)} records)")
            return df, []
        except Exception as e:
            error_msg = f"Error reading JSON file {file_path}: {e}"
            logger.error(error_msg)
            return pd.DataFrame(), [error_msg]
    
    def ingest_parquet(self, file_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """Ingest data from Parquet file"""
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"Successfully read Parquet file: {file_path} ({len(df)} records)")
            return df, []
        except Exception as e:
            error_msg = f"Error reading Parquet file {file_path}: {e}"
            logger.error(error_msg)
            return pd.DataFrame(), [error_msg]
    
    def ingest_xml(self, file_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """Ingest data from XML file"""
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # Extract data based on XML structure
            records = []
            for child in root:
                record = {}
                for elem in child:
                    record[elem.tag] = elem.text
                records.append(record)
            
            df = pd.DataFrame(records)
            logger.info(f"Successfully read XML file: {file_path} ({len(df)} records)")
            return df, []
        except Exception as e:
            error_msg = f"Error reading XML file {file_path}: {e}"
            logger.error(error_msg)
            return pd.DataFrame(), [error_msg]
    
    def ingest_delimited(self, file_path: str, delimiter: str = '|') -> Tuple[pd.DataFrame, List[str]]:
        """Ingest data from delimited file"""
        try:
            df = pd.read_csv(file_path, delimiter=delimiter)
            logger.info(f"Successfully read delimited file: {file_path} ({len(df)} records)")
            return df, []
        except Exception as e:
            error_msg = f"Error reading delimited file {file_path}: {e}"
            logger.error(error_msg)
            return pd.DataFrame(), [error_msg]


class DatabaseIngestionHandler:
    """Handles ingestion from database sources"""
    
    def __init__(self, quality_checker: DataQualityChecker):
        self.quality_checker = quality_checker
    
    def ingest_sqlite(self, db_path: str, table_name: str = None) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """Ingest data from SQLite database"""
        try:
            conn = sqlite3.connect(db_path)
            
            # Get all tables if no specific table requested
            if table_name is None:
                tables_query = "SELECT name FROM sqlite_master WHERE type='table';"
                tables = pd.read_sql_query(tables_query, conn)['name'].tolist()
            else:
                tables = [table_name]
            
            dataframes = {}
            errors = []
            
            for table in tables:
                try:
                    query = f"SELECT * FROM {table}"
                    df = pd.read_sql_query(query, conn)
                    dataframes[table] = df
                    logger.info(f"Successfully read table {table}: {len(df)} records")
                except Exception as e:
                    error_msg = f"Error reading table {table}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            conn.close()
            return dataframes, errors
            
        except Exception as e:
            error_msg = f"Error connecting to SQLite database {db_path}: {e}"
            logger.error(error_msg)
            return {}, [error_msg]


class APIIngestionHandler:
    """Handles ingestion from API sources"""
    
    def __init__(self, quality_checker: DataQualityChecker):
        self.quality_checker = quality_checker
        self.session = requests.Session()
    
    def ingest_api_endpoint(self, url: str, params: Dict = None, headers: Dict = None) -> Tuple[pd.DataFrame, List[str]]:
        """Ingest data from API endpoint"""
        try:
            response = self.session.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Handle different response structures
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # Look for common data keys
                for key in ['data', 'results', 'items', 'records']:
                    if key in data and isinstance(data[key], list):
                        df = pd.DataFrame(data[key])
                        break
                else:
                    # If no standard key found, use the entire response
                    df = pd.DataFrame([data])
            else:
                df = pd.DataFrame([data])
            
            logger.info(f"Successfully ingested from API {url}: {len(df)} records")
            return df, []
            
        except Exception as e:
            error_msg = f"Error ingesting from API {url}: {e}"
            logger.error(error_msg)
            return pd.DataFrame(), [error_msg]
    
    def ingest_paginated_api(self, base_url: str, page_param: str = 'page', 
                           max_pages: int = 10, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        """Ingest data from paginated API"""
        all_data = []
        errors = []
        
        for page in range(1, max_pages + 1):
            params = kwargs.get('params', {}).copy()
            params[page_param] = page
            
            df, page_errors = self.ingest_api_endpoint(
                base_url, 
                params=params, 
                headers=kwargs.get('headers')
            )
            
            if not df.empty:
                all_data.append(df)
            
            errors.extend(page_errors)
            
            # Break if no data returned (end of pages)
            if df.empty:
                break
            
            # Rate limiting
            time.sleep(0.1)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Successfully ingested {len(combined_df)} records from paginated API")
            return combined_df, errors
        else:
            return pd.DataFrame(), errors


class BatchIngestionPipeline:
    """Main batch ingestion pipeline orchestrator"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        self.config = self._load_config(config_path)
        self.quality_checker = DataQualityChecker(config_path)
        self.file_handler = FileIngestionHandler(self.quality_checker)
        self.db_handler = DatabaseIngestionHandler(self.quality_checker)
        self.api_handler = APIIngestionHandler(self.quality_checker)
        
        # Initialize tracking
        self.jobs: List[IngestionJob] = []
        self.processed_data: Dict[str, pd.DataFrame] = {}
        
        # Create output directories
        self.output_base = Path("data/processed/batch")
        self.output_base.mkdir(parents=True, exist_ok=True)
        
        self.error_log_path = self.output_base / "ingestion_errors.log"
        self.job_log_path = self.output_base / "ingestion_jobs.json"
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}. Using defaults.")
            return {}
    
    def create_job(self, source_type: str, source_path: str, target_path: str, 
                   format_type: str) -> str:
        """Create a new ingestion job"""
        job_id = hashlib.md5(f"{source_type}_{source_path}_{datetime.now()}".encode()).hexdigest()[:8]
        
        job = IngestionJob(
            job_id=job_id,
            source_type=source_type,
            source_path=source_path,
            target_path=target_path,
            format_type=format_type
        )
        
        self.jobs.append(job)
        logger.info(f"Created ingestion job {job_id}: {source_type} -> {target_path}")
        return job_id
    
    def execute_job(self, job_id: str) -> bool:
        """Execute a specific ingestion job"""
        job = next((j for j in self.jobs if j.job_id == job_id), None)
        if not job:
            logger.error(f"Job {job_id} not found")
            return False
        
        job.status = "running"
        job.start_time = datetime.now()
        
        try:
            logger.info(f"Executing job {job_id}: {job.source_type} from {job.source_path}")
            
            # Route to appropriate handler
            if job.source_type == "file":
                success = self._execute_file_job(job)
            elif job.source_type == "database":
                success = self._execute_database_job(job)
            elif job.source_type == "api":
                success = self._execute_api_job(job)
            else:
                job.errors.append(f"Unknown source type: {job.source_type}")
                success = False
            
            job.status = "completed" if success else "failed"
            job.end_time = datetime.now()
            
            logger.info(f"Job {job_id} {job.status}. Processed {job.records_processed} records")
            return success
            
        except Exception as e:
            job.status = "failed"
            job.end_time = datetime.now()
            job.errors.append(f"Unexpected error: {e}")
            logger.error(f"Job {job_id} failed with error: {e}")
            return False
    
    def _execute_file_job(self, job: IngestionJob) -> bool:
        """Execute file-based ingestion job"""
        try:
            # Determine file format and ingest
            if job.format_type.lower() == "csv":
                df, errors = self.file_handler.ingest_csv(job.source_path)
            elif job.format_type.lower() == "json":
                df, errors = self.file_handler.ingest_json(job.source_path)
            elif job.format_type.lower() == "parquet":
                df, errors = self.file_handler.ingest_parquet(job.source_path)
            elif job.format_type.lower() == "xml":
                df, errors = self.file_handler.ingest_xml(job.source_path)
            elif job.format_type.lower() == "delimited":
                df, errors = self.file_handler.ingest_delimited(job.source_path)
            else:
                errors = [f"Unsupported file format: {job.format_type}"]
                df = pd.DataFrame()
            
            job.errors.extend(errors)
            
            if not df.empty:
                # Apply data quality checks and cleaning
                is_valid, quality_errors = self.quality_checker.validate_dataframe(df, "reviews")
                job.errors.extend(quality_errors)
                
                if is_valid or self.config.get('ingestion', {}).get('allow_quality_issues', True):
                    cleaned_df = self.quality_checker.clean_dataframe(df, "reviews")
                    
                    # Save processed data
                    self._save_processed_data(cleaned_df, job.target_path)
                    job.records_processed = len(cleaned_df)
                    self.processed_data[job.job_id] = cleaned_df
                    
                    return True
            
            return False
            
        except Exception as e:
            job.errors.append(f"File job execution error: {e}")
            return False
    
    def _execute_database_job(self, job: IngestionJob) -> bool:
        """Execute database-based ingestion job"""
        try:
            if job.format_type.lower() == "sqlite":
                dataframes, errors = self.db_handler.ingest_sqlite(job.source_path)
                job.errors.extend(errors)
                
                if dataframes:
                    # Process each table
                    total_records = 0
                    for table_name, df in dataframes.items():
                        if not df.empty:
                            # Apply data quality checks
                            is_valid, quality_errors = self.quality_checker.validate_dataframe(df, table_name)
                            job.errors.extend(quality_errors)
                            
                            if is_valid or self.config.get('ingestion', {}).get('allow_quality_issues', True):
                                cleaned_df = self.quality_checker.clean_dataframe(df, table_name)
                                
                                # Save each table separately
                                table_target_path = f"{job.target_path}_{table_name}.parquet"
                                self._save_processed_data(cleaned_df, table_target_path)
                                total_records += len(cleaned_df)
                                self.processed_data[f"{job.job_id}_{table_name}"] = cleaned_df
                    
                    job.records_processed = total_records
                    return total_records > 0
            
            return False
            
        except Exception as e:
            job.errors.append(f"Database job execution error: {e}")
            return False
    
    def _execute_api_job(self, job: IngestionJob) -> bool:
        """Execute API-based ingestion job"""
        try:
            # Check if it's a paginated API
            if "paginated" in job.format_type.lower():
                df, errors = self.api_handler.ingest_paginated_api(job.source_path)
            else:
                df, errors = self.api_handler.ingest_api_endpoint(job.source_path)
            
            job.errors.extend(errors)
            
            if not df.empty:
                # Apply data quality checks
                is_valid, quality_errors = self.quality_checker.validate_dataframe(df, "api_data")
                job.errors.extend(quality_errors)
                
                if is_valid or self.config.get('ingestion', {}).get('allow_quality_issues', True):
                    cleaned_df = self.quality_checker.clean_dataframe(df, "api_data")
                    
                    # Save processed data
                    self._save_processed_data(cleaned_df, job.target_path)
                    job.records_processed = len(cleaned_df)
                    self.processed_data[job.job_id] = cleaned_df
                    
                    return True
            
            return False
            
        except Exception as e:
            job.errors.append(f"API job execution error: {e}")
            return False
    
    def _save_processed_data(self, df: pd.DataFrame, target_path: str):
        """Save processed data to target location"""
        try:
            # Ensure target directory exists
            target_file = Path(target_path)
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as Parquet for efficiency
            if not target_path.endswith('.parquet'):
                target_path += '.parquet'
            
            df.to_parquet(target_path, index=False)
            logger.info(f"Saved {len(df)} records to {target_path}")
            
        except Exception as e:
            logger.error(f"Error saving data to {target_path}: {e}")
            raise
    
    def execute_all_jobs(self, max_workers: int = 4) -> Dict[str, bool]:
        """Execute all pending jobs with parallel processing"""
        pending_jobs = [job for job in self.jobs if job.status == "pending"]
        
        if not pending_jobs:
            logger.info("No pending jobs to execute")
            return {}
        
        logger.info(f"Executing {len(pending_jobs)} jobs with {max_workers} workers")
        
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self.execute_job, job.job_id): job.job_id 
                for job in pending_jobs
            }
            
            # Collect results
            for future in as_completed(future_to_job):
                job_id = future_to_job[future]
                try:
                    success = future.result()
                    results[job_id] = success
                except Exception as e:
                    logger.error(f"Job {job_id} failed with exception: {e}")
                    results[job_id] = False
        
        # Save job logs
        self._save_job_logs()
        
        return results
    
    def _save_job_logs(self):
        """Save job execution logs"""
        job_data = []
        for job in self.jobs:
            job_dict = {
                'job_id': job.job_id,
                'source_type': job.source_type,
                'source_path': job.source_path,
                'target_path': job.target_path,
                'format_type': job.format_type,
                'status': job.status,
                'start_time': job.start_time.isoformat() if job.start_time else None,
                'end_time': job.end_time.isoformat() if job.end_time else None,
                'records_processed': job.records_processed,
                'errors': job.errors
            }
            job_data.append(job_dict)
        
        with open(self.job_log_path, 'w') as f:
            json.dump(job_data, f, indent=2)
        
        logger.info(f"Job logs saved to {self.job_log_path}")
    
    def get_ingestion_summary(self) -> Dict:
        """Get summary of ingestion pipeline execution"""
        total_jobs = len(self.jobs)
        completed_jobs = len([j for j in self.jobs if j.status == "completed"])
        failed_jobs = len([j for j in self.jobs if j.status == "failed"])
        total_records = sum(j.records_processed for j in self.jobs)
        
        return {
            'total_jobs': total_jobs,
            'completed_jobs': completed_jobs,
            'failed_jobs': failed_jobs,
            'success_rate': completed_jobs / total_jobs if total_jobs > 0 else 0,
            'total_records_processed': total_records,
            'processed_datasets': list(self.processed_data.keys()),
            'execution_timestamp': datetime.now().isoformat()
        }


def main():
    """Main function to demonstrate batch ingestion pipeline"""
    logger.info("🚀 Starting Batch Ingestion Pipeline Demo")
    
    # Initialize pipeline
    pipeline = BatchIngestionPipeline()
    
    # Create sample ingestion jobs
    jobs = [
        # File-based sources
        ("file", "data/raw/reviews/reviews_2024.csv", "data/processed/batch/reviews_csv", "csv"),
        ("file", "data/raw/reviews/reviews_2024.json", "data/processed/batch/reviews_json", "json"),
        ("file", "data/raw/files/products.xml", "data/processed/batch/products_xml", "xml"),
        
        # Database sources
        ("database", "data/raw/databases/ecommerce.db", "data/processed/batch/database_tables", "sqlite"),
        
        # API sources (if available)
        ("api", "http://127.0.0.1:8000/api/v1/products", "data/processed/batch/api_products", "json"),
        ("api", "http://127.0.0.1:8000/api/v1/reviews", "data/processed/batch/api_reviews", "json"),
    ]
    
    # Create jobs
    job_ids = []
    for source_type, source_path, target_path, format_type in jobs:
        try:
            job_id = pipeline.create_job(source_type, source_path, target_path, format_type)
            job_ids.append(job_id)
        except Exception as e:
            logger.warning(f"Could not create job for {source_path}: {e}")
    
    # Execute all jobs
    results = pipeline.execute_all_jobs(max_workers=2)
    
    # Print summary
    summary = pipeline.get_ingestion_summary()
    
    print("\n" + "="*60)
    print("📊 BATCH INGESTION PIPELINE SUMMARY")
    print("="*60)
    print(f"Total Jobs: {summary['total_jobs']}")
    print(f"Completed: {summary['completed_jobs']}")
    print(f"Failed: {summary['failed_jobs']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Total Records Processed: {summary['total_records_processed']:,}")
    print(f"Processed Datasets: {len(summary['processed_datasets'])}")
    print("="*60)
    
    logger.info("✅ Batch ingestion pipeline demo completed")


if __name__ == "__main__":
    main()